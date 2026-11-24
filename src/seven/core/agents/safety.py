
"""Safety domain models for autonomous agent execution."""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID


class PermissionLevel(StrEnum):
    """Define the permission levels available to agent tools."""

    DENIED = "denied"
    REQUIRES_APPROVAL = "requires_approval"
    ALLOWED = "allowed"


class SafetyDecision(StrEnum):
    """Represent the outcome of a safety policy evaluation."""

    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    DENY = "deny"


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolPermission:
    """Define a permission granted to a specific tool."""

    tool_name: str
    level: PermissionLevel
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty.")

        if not isinstance(self.level, PermissionLevel):
            raise TypeError("level must be a PermissionLevel instance.")

        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("reason must be a string or None.")


@dataclass(frozen=True, slots=True, kw_only=True)
class SafetyPolicyResult:
    """Represent the result of evaluating a tool execution request."""

    decision: SafetyDecision
    tool_name: str
    reason: str
    task_id: UUID | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, SafetyDecision):
            raise TypeError("decision must be a SafetyDecision instance.")

        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty.")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason cannot be empty.")

        if self.task_id is not None and not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID or None.")

        metadata = self.metadata or {}

        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        if not all(isinstance(key, str) for key in metadata):
            raise TypeError("metadata keys must be strings.")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(metadata)),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolPermissionSet:
    """Provide an immutable collection of tool permissions."""

    permissions: tuple[ToolPermission, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.permissions, tuple):
            raise TypeError("permissions must be a tuple.")

        if not all(
            isinstance(permission, ToolPermission)
            for permission in self.permissions
        ):
            raise TypeError(
                "permissions must contain ToolPermission instances."
            )

        names = [permission.tool_name for permission in self.permissions]

        if len(names) != len(set(names)):
            raise ValueError("Tool permissions cannot contain duplicate names.")

    def get(self, tool_name: str) -> ToolPermission | None:
        """Return the permission assigned to a tool."""
        return next(
            (
                permission
                for permission in self.permissions
                if permission.tool_name == tool_name
            ),
            None,
        )
