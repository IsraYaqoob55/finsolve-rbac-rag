"""HTTP middleware: request IDs, access logs, security headers, safe errors.

Written as plain ASGI middleware (a function wrapping the app) rather than
Starlette's ``BaseHTTPMiddleware``, so streamed responses (``/chat/stream``)
pass through untouched and the request ID stays set while they stream.
"""

import json
import logging
import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.request_context import new_request_id, reset_request_id, set_request_id

access_log = logging.getLogger("access")
logger = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "x-content-type-options": "nosniff",  # browsers must not guess content types
    "x-frame-options": "DENY",  # the API is never shown inside a frame
    "referrer-policy": "no-referrer",
}


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        request_id = new_request_id(headers.get(b"x-request-id", b"").decode("latin-1"))
        token = set_request_id(request_id)
        started = time.perf_counter()
        status_code = 500  # if the app crashes before responding
        response_started = False

        async def send_with_headers(message: Message) -> None:
            nonlocal status_code, response_started
            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                extra = [(b"x-request-id", request_id.encode())] + [
                    (k.encode(), v.encode()) for k, v in SECURITY_HEADERS.items()
                ]
                message["headers"] = list(message.get("headers", [])) + extra
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            # Never send stack traces or internal messages to clients: log them,
            # and give the client the request ID to quote when reporting it.
            logger.exception("Unhandled error", extra={"path": scope["path"]})
            if response_started:  # headers already sent: we can only end the stream
                raise
            body = json.dumps({"detail": "Internal server error", "request_id": request_id})
            await send_with_headers(
                {
                    "type": "http.response.start",
                    "status": 500,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body.encode()})
        finally:
            # Health probes run every few seconds; keep them out of normal logs.
            level = logging.DEBUG if scope["path"].startswith("/health") else logging.INFO
            access_log.log(
                level,
                "%s %s %d",
                scope["method"],
                scope["path"],
                status_code,
                extra={"duration_ms": int((time.perf_counter() - started) * 1000)},
            )
            reset_request_id(token)
