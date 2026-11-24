
"""Safety policy evaluation for autonomous agent tools."""

from seven.core.agents.safety import (
    PermissionLevel,
    SafetyDecision,
    SafetyPolicyResult,
    ToolPermissionSet,
)
from seven.core.agents.tools.context import ToolExecutionContext
from seven.core.agents.tools.models import ToolDefinition


class SafetyPolicyEngine:
    """Evaluate whether an agent tool may execute."""

    def __init__(
        self,
        permissions: ToolPermissionSet | None = None,
        default_permission: PermissionLevel = PermissionLevel.REQUIRES_APPROVAL,
    ) -> None:
        if permissions is not None and not isinstance(
            permissions,
            ToolPermissionSet,
        ):
            raise TypeError(
                "permissions must be a ToolPermissionSet instance or None."
            )

        if not isinstance(default_permission, PermissionLevel):
            raise TypeError(
                "default_permission must be a PermissionLevel instance."
            )

        self._permissions = permissions or ToolPermissionSet()
        self._default_permission = default_permission

    @property
    def permissions(self) -> ToolPermissionSet:
        """Return the configured permission set."""
        return self._permissions

    @property
    def default_permission(self) -> PermissionLevel:
        """Return the default permission level."""
        return self._default_permission

    def evaluate(
        self,
        definition: ToolDefinition,
        context: ToolExecutionContext | None = None,
    ) -> SafetyPolicyResult:
        """Evaluate a tool execution request against configured permissions."""
        if not isinstance(definition, ToolDefinition):
            raise TypeError("definition must be a ToolDefinition instance.")

        if context is not None and not isinstance(
            context,
            ToolExecutionContext,
        ):
            raise TypeError(
                "context must be a ToolExecutionContext instance or None."
            )

        permission = self._permissions.get(definition.name)

        level = (
            permission.level
            if permission is not None
            else self._default_permission
        )

        if definition.requires_confirmation:
            level = self._restrict_permission(
                level,
                PermissionLevel.REQUIRES_APPROVAL,
            )

        reason = (
            permission.reason
            if permission is not None and permission.reason
            else self._default_reason(level)
        )

        decision = {
            PermissionLevel.ALLOWED: SafetyDecision.ALLOW,
            PermissionLevel.REQUIRES_APPROVAL: SafetyDecision.REQUIRE_APPROVAL,
            PermissionLevel.DENIED: SafetyDecision.DENY,
        }[level]

        return SafetyPolicyResult(
            decision=decision,
            tool_name=definition.name,
            reason=reason,
            task_id=context.task_id if context is not None else None,
            metadata={
                "permission_level": level.value,
                "requires_confirmation": definition.requires_confirmation,
            },
        )

    @staticmethod
    def _restrict_permission(
        current: PermissionLevel,
        required: PermissionLevel,
    ) -> PermissionLevel:
        """Return the more restrictive permission level."""
        rank = {
            PermissionLevel.ALLOWED: 0,
            PermissionLevel.REQUIRES_APPROVAL: 1,
            PermissionLevel.DENIED: 2,
        }

        return current if rank[current] >= rank[required] else required

    @staticmethod
    def _default_reason(level: PermissionLevel) -> str:
        """Return a descriptive reason for a policy decision."""
        reasons = {
            PermissionLevel.ALLOWED: "Tool execution is permitted.",
            PermissionLevel.REQUIRES_APPROVAL: (
                "User approval is required before tool execution."
            ),
            PermissionLevel.DENIED: "Tool execution is denied by policy.",
        }

        return reasons[level]
