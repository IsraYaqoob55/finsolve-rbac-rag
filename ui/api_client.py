"""Typed HTTP client for the FinSolve API.

The UI never imports backend code: it only knows the HTTP contract. This is
how a real frontend works, and it means the UI and API can be deployed and
scaled separately.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx


class ApiError(Exception):
    """Something went wrong talking to the API; ``message`` is safe to show users."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class AuthError(ApiError):
    """Credentials are wrong, or the session expired: the user must log in again."""


@dataclass(frozen=True)
class Profile:
    username: str
    full_name: str
    role: str
    role_display_name: str
    role_description: str
    departments: list[tuple[str, str]]  # (name, description)
    suggested_questions: list[str]


@dataclass(frozen=True)
class Source:
    number: int
    document: str
    title: str
    section: str
    department: str
    quarter: str | None
    score: float
    excerpt: str
    cited: bool


@dataclass
class ChatResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    model: str | None = None
    latency_ms: int = 0
    search_query: str = ""
    route: str = "docs"


@dataclass(frozen=True)
class Token:
    text: str


@dataclass(frozen=True)
class Done:
    result: ChatResult


StreamEvent = Token | Done


def _parse_result(data: dict[str, Any]) -> ChatResult:
    return ChatResult(
        answer=data["answer"],
        sources=[Source(**s) for s in data.get("sources", [])],
        model=data.get("model"),
        latency_ms=data.get("latency_ms", 0),
        search_query=data.get("search_query", ""),
        route=data.get("route", "docs"),
    )


def iter_sse(lines: Iterator[str]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Parse a Server-Sent Events stream into (event, data) pairs."""
    event: str = "message"
    data_lines: list[str] = []
    for line in lines:
        if line == "":  # a blank line ends one event
            if data_lines:
                yield event, json.loads("\n".join(data_lines))
            event, data_lines = "message", []
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if data_lines:
        yield event, json.loads("\n".join(data_lines))


class FinSolveClient:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout: float = 90,
        transport: httpx.BaseTransport | None = None,  # tests inject a mock transport
    ) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self.token = token

    def close(self) -> None:
        self._http.close()

    # ---------- helpers ----------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @staticmethod
    def _raise_for(response: httpx.Response) -> None:
        if response.status_code == 401:
            raise AuthError("Your session has expired. Please log in again.", retryable=False)
        if response.status_code == 403:
            raise ApiError("Your role is not allowed to do this.", retryable=False)
        if response.status_code == 503:
            detail = response.json().get("detail", "The assistant is temporarily unavailable.")
            raise ApiError(str(detail))
        if response.is_error:
            raise ApiError(f"Unexpected server error ({response.status_code}).")

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._http.request(method, path, headers=self._headers(), **kwargs)
        except httpx.TransportError as exc:
            raise ApiError("Cannot reach the FinSolve API. Is the backend running?") from exc
        self._raise_for(response)
        return response

    # ---------- endpoints ----------

    def login(self, username: str, password: str) -> str:
        try:
            response = self._http.post(
                "/auth/login", data={"username": username, "password": password}
            )
        except httpx.TransportError as exc:
            raise ApiError("Cannot reach the FinSolve API. Is the backend running?") from exc
        if response.status_code == 401:
            raise AuthError("Incorrect username or password.", retryable=False)
        self._raise_for(response)
        self.token = str(response.json()["access_token"])
        return self.token

    def me(self) -> Profile:
        data = self._request("GET", "/auth/me").json()
        return Profile(
            username=data["username"],
            full_name=data["full_name"],
            role=data["role"],
            role_display_name=data["role_display_name"],
            role_description=data["role_description"],
            departments=[(d["name"], d["description"]) for d in data["accessible_departments"]],
            suggested_questions=data.get("suggested_questions", []),
        )

    def chat(self, question: str, history: list[dict[str, str]]) -> ChatResult:
        response = self._request("POST", "/chat", json={"question": question, "history": history})
        return _parse_result(response.json())

    def chat_stream(self, question: str, history: list[dict[str, str]]) -> Iterator[StreamEvent]:
        """Yield Token events as the answer is generated, then one Done event."""
        payload = {"question": question, "history": history}
        try:
            with self._http.stream(
                "POST", "/chat/stream", json=payload, headers=self._headers()
            ) as response:
                if response.is_error:
                    response.read()
                    self._raise_for(response)
                for event, data in iter_sse(response.iter_lines()):
                    if event == "token":
                        yield Token(data["text"])
                    elif event == "done":
                        yield Done(_parse_result(data))
                    elif event == "error":
                        raise ApiError(
                            data.get("detail", "The assistant failed."), data["retryable"]
                        )
        except httpx.TransportError as exc:
            raise ApiError("Lost connection to the FinSolve API.") from exc
