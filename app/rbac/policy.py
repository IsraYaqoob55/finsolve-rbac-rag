"""Role policy: which departments each role may read.

Loaded once from ``config/roles.yaml`` and validated strictly, so a broken
policy stops the app at startup instead of silently granting or denying access.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

WILDCARD = "*"


class UnknownRoleError(KeyError):
    """Raised when asked about a role that is not defined in the policy."""


class RoleDefinition(BaseModel):
    display_name: str
    description: str = ""
    departments: list[str] = Field(min_length=1)


class PolicyFile(BaseModel):
    """Schema of roles.yaml."""

    departments: dict[str, str] = Field(min_length=1)
    roles: dict[str, RoleDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def _roles_reference_known_departments(self) -> "PolicyFile":
        known = set(self.departments)
        for name, role in self.roles.items():
            unknown = set(role.departments) - known - {WILDCARD}
            if unknown:
                raise ValueError(f"role '{name}' references unknown departments: {sorted(unknown)}")
        return self


@dataclass(frozen=True)
class AccessScope:
    """What one role may see. Always an explicit, finite set of departments."""

    role: str
    display_name: str
    departments: frozenset[str]

    def can_access(self, department: str) -> bool:
        return department in self.departments


class RolePolicy:
    """Answers "what may this role see?"."""

    def __init__(self, policy: PolicyFile) -> None:
        self._policy = policy
        all_departments = frozenset(policy.departments)
        self._scopes = {
            name: AccessScope(
                role=name,
                display_name=role.display_name,
                departments=(
                    all_departments if WILDCARD in role.departments else frozenset(role.departments)
                ),
            )
            for name, role in policy.roles.items()
        }

    @classmethod
    def from_yaml(cls, path: Path) -> "RolePolicy":
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            return cls(PolicyFile.model_validate(raw))
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            raise RuntimeError(f"Invalid role policy file {path}: {exc}") from exc

    @property
    def roles(self) -> list[str]:
        return list(self._scopes)

    @property
    def departments(self) -> dict[str, str]:
        return dict(self._policy.departments)

    def describe(self, role: str) -> RoleDefinition:
        if role not in self._policy.roles:
            raise UnknownRoleError(role)
        return self._policy.roles[role]

    def scope_for(self, role: str) -> AccessScope:
        """Resolve a role to its access scope. Unknown roles are an error, never "all"."""
        try:
            return self._scopes[role]
        except KeyError:
            raise UnknownRoleError(role) from None
