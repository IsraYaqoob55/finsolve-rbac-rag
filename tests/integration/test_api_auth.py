"""API tests for login, /auth/me and token handling."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ROLE_USERS, TEST_PASSWORD, SessionServices, auth_header, login


def test_login_returns_bearer_token(client: TestClient) -> None:
    response = client.post("/auth/login", data={"username": "t_finance", "password": TEST_PASSWORD})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"  # noqa: S105
    assert body["role"] == "finance"


@pytest.mark.parametrize(("username", "password"), [("t_finance", "wrong-pass"), ("ghost", "x")])
def test_bad_login_gives_same_generic_error(
    client: TestClient, username: str, password: str
) -> None:
    response = client.post("/auth/login", data={"username": username, "password": password})
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"


def test_me_shows_role_and_accessible_departments(client: TestClient) -> None:
    token = login(client, "t_finance")
    body = client.get("/auth/me", headers=auth_header(token)).json()
    assert body["role"] == "finance"
    assert [d["name"] for d in body["accessible_departments"]] == [
        "finance",
        "general",
        "marketing",
    ]


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer not-a-jwt"}, {"Authorization": "Basic dXNlcjpwYXNz"}],
)
def test_protected_endpoints_reject_missing_or_bad_tokens(
    client: TestClient, headers: dict[str, str]
) -> None:
    assert client.get("/auth/me", headers=headers).status_code == 401
    assert client.post("/search", json={"query": "revenue"}, headers=headers).status_code == 401


def test_disabling_user_revokes_existing_token(
    client: TestClient, services: SessionServices
) -> None:
    services.users.create("temp_user", "Temp", "employee", TEST_PASSWORD)
    token = login(client, "temp_user")
    assert client.get("/auth/me", headers=auth_header(token)).status_code == 200

    services.users.set_disabled("temp_user", True)

    assert client.get("/auth/me", headers=auth_header(token)).status_code == 401


def test_role_change_applies_to_existing_token(
    client: TestClient, services: SessionServices
) -> None:
    services.users.create("promo_user", "Promo", "employee", TEST_PASSWORD)
    token = login(client, "promo_user")  # token says role=employee

    services.users.set_role("promo_user", "hr")

    body = client.get("/auth/me", headers=auth_header(token)).json()
    assert body["role"] == "hr"  # the database wins over the token claim


def test_user_with_role_missing_from_policy_is_forbidden(
    client: TestClient, services: SessionServices
) -> None:
    services.users.create("ghost_role", "Ghost", "contractor", TEST_PASSWORD)
    token = login(client, "ghost_role")
    response = client.post("/search", json={"query": "policy"}, headers=auth_header(token))
    assert response.status_code == 403


def test_all_seeded_roles_can_log_in(client: TestClient) -> None:
    for username in ROLE_USERS.values():
        assert login(client, username)
