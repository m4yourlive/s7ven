
"""Tests for agent safety policy models and evaluation."""

from uuid import uuid4

import pytest

from seven.core.agents.safety import (
    PermissionLevel,
    SafetyDecision,
    SafetyPolicyResult,
    ToolPermission,
    ToolPermissionSet,
)
from seven.core.agents.safety_policy import SafetyPolicyEngine
from seven.core.agents.tools.context import ToolExecutionContext
from seven.core.agents.tools.models import ToolDefinition


def create_definition(
    name: str = "test_tool",
    requires_confirmation: bool = False,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="A test tool.",
        requires_confirmation=requires_confirmation,
    )


def test_permission_model_initialization():
    permission = ToolPermission(
        tool_name="test_tool",
        level=PermissionLevel.ALLOWED,
    )

    assert permission.tool_name == "test_tool"
    assert permission.level is PermissionLevel.ALLOWED


def test_permission_model_rejects_empty_tool_name():
    with pytest.raises(ValueError):
        ToolPermission(
            tool_name=" ",
            level=PermissionLevel.ALLOWED,
        )


def test_permission_model_rejects_invalid_level():
    with pytest.raises(TypeError):
        ToolPermission(
            tool_name="test_tool",
            level="allowed",
        )


def test_permission_set_rejects_duplicate_tool_names():
    with pytest.raises(ValueError):
        ToolPermissionSet(
            permissions=(
                ToolPermission(
                    tool_name="test_tool",
                    level=PermissionLevel.ALLOWED,
                ),
                ToolPermission(
                    tool_name="test_tool",
                    level=PermissionLevel.DENIED,
                ),
            )
        )


def test_permission_set_get_returns_matching_permission():
    permission = ToolPermission(
        tool_name="test_tool",
        level=PermissionLevel.ALLOWED,
    )
    permissions = ToolPermissionSet(permissions=(permission,))

    assert permissions.get("test_tool") is permission
    assert permissions.get("missing_tool") is None


def test_safety_policy_defaults_to_approval():
    engine = SafetyPolicyEngine()

    result = engine.evaluate(create_definition())

    assert result.decision is SafetyDecision.REQUIRE_APPROVAL
    assert result.metadata["permission_level"] == "requires_approval"


def test_safety_policy_allows_explicitly_permitted_tool():
    permissions = ToolPermissionSet(
        permissions=(
            ToolPermission(
                tool_name="test_tool",
                level=PermissionLevel.ALLOWED,
            ),
        )
    )
    engine = SafetyPolicyEngine(permissions=permissions)

    result = engine.evaluate(create_definition())

    assert result.decision is SafetyDecision.ALLOW


def test_safety_policy_denies_explicitly_denied_tool():
    permissions = ToolPermissionSet(
        permissions=(
            ToolPermission(
                tool_name="test_tool",
                level=PermissionLevel.DENIED,
            ),
        )
    )
    engine = SafetyPolicyEngine(permissions=permissions)

    result = engine.evaluate(create_definition())

    assert result.decision is SafetyDecision.DENY


def test_confirmation_requirement_cannot_be_overridden_by_allow_permission():
    permissions = ToolPermissionSet(
        permissions=(
            ToolPermission(
                tool_name="test_tool",
                level=PermissionLevel.ALLOWED,
            ),
        )
    )
    engine = SafetyPolicyEngine(permissions=permissions)

    result = engine.evaluate(
        create_definition(requires_confirmation=True)
    )

    assert result.decision is SafetyDecision.REQUIRE_APPROVAL


def test_denied_permission_cannot_be_overridden_by_confirmation_requirement():
    permissions = ToolPermissionSet(
        permissions=(
            ToolPermission(
                tool_name="test_tool",
                level=PermissionLevel.DENIED,
            ),
        )
    )
    engine = SafetyPolicyEngine(permissions=permissions)

    result = engine.evaluate(
        create_definition(requires_confirmation=True)
    )

    assert result.decision is SafetyDecision.DENY


def test_safety_policy_includes_task_id():
    task_id = uuid4()
    context = ToolExecutionContext(task_id=task_id)

    result = SafetyPolicyEngine().evaluate(
        create_definition(),
        context,
    )

    assert result.task_id == task_id


def test_safety_policy_result_freezes_metadata():
    metadata = {"source": "test"}

    result = SafetyPolicyResult(
        decision=SafetyDecision.ALLOW,
        tool_name="test_tool",
        reason="Allowed.",
        metadata=metadata,
    )

    metadata["source"] = "modified"

    assert result.metadata["source"] == "test"

    with pytest.raises(TypeError):
        result.metadata["source"] = "modified"


def test_safety_policy_rejects_invalid_definition():
    engine = SafetyPolicyEngine()

    with pytest.raises(TypeError):
        engine.evaluate(object())


def test_safety_policy_rejects_invalid_context():
    engine = SafetyPolicyEngine()

    with pytest.raises(TypeError):
        engine.evaluate(
            create_definition(),
            object(),
        )
