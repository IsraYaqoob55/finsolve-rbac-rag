"""FinSolve Assistant: the chat UI.

Run with:
    streamlit run ui/streamlit_app.py

Streamlit re-runs this whole script on every interaction; anything that must
survive between runs (token, profile, messages) lives in ``st.session_state``.
"""

from collections.abc import Iterator
from typing import Any

import streamlit as st

from ui.api_client import ApiError, AuthError, ChatResult, Done, FinSolveClient, Token
from ui.components import (
    render_answer_footer,
    render_demo_accounts,
    render_login_header,
    render_sidebar,
    render_sources,
)
from ui.config import API_BASE_URL, HISTORY_MESSAGES

st.set_page_config(page_title="FinSolve Assistant", page_icon="💬", layout="centered")

# Wide answer tables scroll inside their own box instead of stretching the page.
st.html(
    """<style>
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { overflow-x: auto; }
    [data-testid="stChatMessage"] table { font-size: 0.875rem; }
    [data-testid="stChatMessage"] th, [data-testid="stChatMessage"] td {
        padding: 0.35rem 0.5rem; min-width: 6rem;
    }
    </style>"""
)

state: Any = st.session_state  # typed loosely: session_state is a dynamic mapping


def reset_session(message: str | None = None) -> None:
    for key in ("token", "profile", "messages", "pending"):
        state.pop(key, None)
    if message:
        state["flash"] = message


def client() -> FinSolveClient:
    return FinSolveClient(API_BASE_URL, token=state.get("token"))


# ---------------------------------------------------------------- login


def login_view() -> None:
    render_login_header()
    if flash := state.pop("flash", None):
        st.info(flash)
    with st.form("login", border=True):
        username = st.text_input("Username", autocomplete="username")
        password = st.text_input("Password", type="password", autocomplete="current-password")
        submitted = st.form_submit_button("Log in", type="primary", use_container_width=True)
    if submitted:
        if not username or not password:
            st.warning("Please enter your username and password.")
        else:
            api = client()
            try:
                state["token"] = api.login(username.strip(), password)
                state["profile"] = api.me()
                state["messages"] = []
                st.rerun()
            except ApiError as exc:
                state.pop("token", None)
                st.error(exc.message)
            finally:
                api.close()
    render_demo_accounts()


# ---------------------------------------------------------------- chat


def stream_answer(question: str, history: list[dict[str, str]]) -> ChatResult | None:
    """Stream tokens into the page; return the final result (or None on error)."""
    holder: dict[str, ChatResult] = {}
    api = client()

    def tokens() -> Iterator[str]:
        for event in api.chat_stream(question, history):
            if isinstance(event, Token):
                yield event.text
            elif isinstance(event, Done):
                holder["result"] = event.result

    try:
        with st.spinner("Searching your documents…", show_time=False):
            stream = tokens()
            first = next(stream, "")  # wait for retrieval + first token under the spinner

        def full() -> Iterator[str]:
            yield first
            yield from stream

        st.write_stream(full)
    except AuthError as exc:
        reset_session(exc.message)
        st.rerun()
    except ApiError as exc:
        st.error(exc.message + (" Please try again." if exc.retryable else ""))
        return None
    finally:
        api.close()
    return holder.get("result")


def history_payload() -> list[dict[str, str]]:
    """Previous turns to send for follow-up questions (most recent only)."""
    turns = [{"role": m["role"], "content": m["content"]} for m in state["messages"]]
    return turns[-HISTORY_MESSAGES:] if HISTORY_MESSAGES else []


def render_history() -> None:
    for message in state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            result: ChatResult | None = message.get("result")
            if result is not None:
                render_sources(result, state.get("show_all_sources", False))
                render_answer_footer(result)


def render_welcome(suggestions: list[str]) -> None:
    profile = state["profile"]
    st.markdown(f"#### Hi {profile.full_name.split()[0]}, what would you like to know?")
    if suggestions:
        choice = st.pills(
            "Try one of these", suggestions, selection_mode="single", key="suggestion"
        )
        if choice:
            state["pending"] = choice
            state.pop("suggestion", None)
            st.rerun()


def chat_view() -> None:
    profile = state["profile"]
    new_chat, logout = render_sidebar(profile)
    if logout:
        reset_session("You have been logged out.")
        st.rerun()
    if new_chat:
        state["messages"] = []
        st.rerun()

    st.markdown("## 💬 FinSolve Assistant")
    if not state["messages"]:
        render_welcome(profile.suggested_questions)
    render_history()

    typed = st.chat_input(f"Ask a question ({profile.role_display_name} access)…")
    question = typed or state.pop("pending", None)
    if not question:
        return

    history = history_payload()
    state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        result = stream_answer(question, history)
        if result is None:
            state["messages"].pop()  # let the user retry the same question
            return
        render_sources(result, state.get("show_all_sources", False))
        render_answer_footer(result)
    state["messages"].append({"role": "assistant", "content": result.answer, "result": result})


if "token" in state and "profile" in state:
    chat_view()
else:
    login_view()
