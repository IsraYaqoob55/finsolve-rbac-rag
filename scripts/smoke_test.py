"""Smoke test for a running deployment (used by CI after `docker compose up`).

    python scripts/smoke_test.py [--api URL] [--ui URL] [--timeout SECONDS]

Checks, without spending LLM tokens:
1. the API becomes ready (index built, users seeded) and the UI is up;
2. a demo user can log in and sees the right departments;
3. RBAC holds end to end: an Employee's search returns only general documents.
Standard library only, so it runs anywhere Python does.
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def request(url: str, data: bytes | None = None, headers: dict[str, str] | None = None) -> Any:
    req = urllib.request.Request(url, data=data, headers=headers or {})  # noqa: S310 - our URLs
    with urllib.request.urlopen(req, timeout=10) as response:  # noqa: S310
        return json.loads(response.read() or b"null")


def wait_until_ready(url: str, timeout: float) -> None:
    """Poll until ``url`` answers 200 (any body: Streamlit's health check is plain text)."""
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
                print(f"ready: {url} -> {response.read().decode()[:200]}")
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            last = str(exc)
        time.sleep(3)
    sys.exit(f"FAIL: {url} not ready after {timeout:.0f}s ({last})")


def login(api: str, username: str, password: str) -> str:
    form = urllib.parse.urlencode({"username": username, "password": password}).encode()
    body = request(f"{api}/auth/login", form, {"Content-Type": "application/x-www-form-urlencoded"})
    return str(body["access_token"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--ui", default="http://localhost:8501")
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    password = os.environ.get("SEED_USER_PASSWORD", "FinSolveDemo@2024")

    wait_until_ready(f"{args.api}/health/ready", args.timeout)
    wait_until_ready(f"{args.ui}/_stcore/health", 60)

    ceo = login(args.api, "tony_ceo", password)
    me = request(f"{args.api}/auth/me", headers={"Authorization": f"Bearer {ceo}"})
    departments = {d["name"] for d in me["accessible_departments"]}
    expected = {"engineering", "finance", "general", "hr", "marketing"}
    if departments != expected:
        sys.exit(f"FAIL: C-Level sees {sorted(departments)}")
    print("ok: C-Level login sees all departments")

    employee = login(args.api, "alice_emp", password)
    results = request(
        f"{args.api}/search",
        json.dumps({"query": "quarterly revenue and employee salaries", "limit": 10}).encode(),
        {"Authorization": f"Bearer {employee}", "Content-Type": "application/json"},
    )["results"]
    seen = {r["source"]["department"] for r in results}
    if not results or seen != {"general"}:
        sys.exit(f"FAIL: Employee search returned departments {sorted(seen)}")
    print(f"ok: Employee search returned {len(results)} results, all from 'general'")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
