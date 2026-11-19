"""Tests for the role policy and the RBAC guard."""

from dataclasses import dataclass
from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT
from app.rbac.guards import enforce_scope
from app.rbac.policy import AccessScope, RolePolicy, UnknownRoleError

ROLES_FILE = PROJECT_ROOT / "config" / "roles.yaml"


@pytest.fixture(scope="module")
def policy() -> RolePolicy:
    return RolePolicy.from_yaml(ROLES_FILE)


def test_wildcard_expands_to_every_registered_department(policy: RolePolicy) -> None:
    scope = policy.scope_for("c_level")
    assert scope.departments == frozenset(policy.departments)
    assert "*" not in scope.departments


def test_employee_sees_only_general(policy: RolePolicy) -> None:
    assert policy.scope_for("employee").departments == {"general"}


def test_unknown_role_raises_instead_of_granting_access(policy: RolePolicy) -> None:
    with pytest.raises(UnknownRoleError):
        policy.scope_for("superuser")


def test_registry_matches_data_folders(policy: RolePolicy) -> None:
    """Every data folder must be a registered department and vice versa.

    Otherwise a new folder's documents would be invisible to everyone, or a
    typo in the registry would silently match nothing.
    """
    folders = {p.name for p in (PROJECT_ROOT / "data").iterdir() if p.is_dir()}
    assert folders == set(policy.departments)


def test_role_with_unknown_department_fails_at_load(tmp_path: Path) -> None:
    bad = tmp_path / "roles.yaml"
    bad.write_text(
        "departments: {general: x}\n"
        "roles: {intern: {display_name: Intern, departments: [general, finanse]}}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="finanse"):
        RolePolicy.from_yaml(bad)


def test_role_with_no_departments_fails_at_load(tmp_path: Path) -> None:
    bad = tmp_path / "roles.yaml"
    bad.write_text(
        "departments: {general: x}\nroles: {intern: {display_name: Intern, departments: []}}\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError):
        RolePolicy.from_yaml(bad)


# ---------- guard ----------


@dataclass(frozen=True)
class _Chunk:
    department: str


@dataclass(frozen=True)
class _Hit:
    chunk: _Chunk


def test_guard_drops_out_of_scope_hits(caplog: pytest.LogCaptureFixture) -> None:
    scope = AccessScope("marketing", "Marketing", frozenset({"marketing", "general"}))
    hits = [_Hit(_Chunk("marketing")), _Hit(_Chunk("hr")), _Hit(_Chunk("general"))]

    allowed = enforce_scope(hits, scope)

    assert [h.chunk.department for h in allowed] == ["marketing", "general"]
    assert "RBAC guard blocked" in caplog.text
