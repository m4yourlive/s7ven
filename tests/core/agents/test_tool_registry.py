"""Tests for the autonomous agent tool registry."""

import pytest

from seven.core.agents.tools import (
    Tool,
    ToolAlreadyRegisteredError,
    ToolDefinition,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistrationError,
)


class MockTool(Tool):
    """Concrete tool implementation for registry testing."""

    def __init__(self, name: str = "mock_tool") -> None:
        self._definition = ToolDefinition(
            name=name,
            description="A mock tool for registry testing.",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, call):
        from seven.core.agents.tools import ToolResult

        return ToolResult(
            call_id=call.call_id,
            tool_name=self.definition.name,
            success=True,
            output="Completed.",
        )


def test_registry_starts_empty():
    registry = ToolRegistry()

    assert registry.count == 0
    assert registry.list_tools() == ()
    assert registry.list_definitions() == ()


def test_register_tool():
    registry = ToolRegistry()
    tool = MockTool()

    registry.register(tool)

    assert registry.count == 1
    assert registry.contains("mock_tool")


def test_register_rejects_non_tool_object():
    registry = ToolRegistry()

    with pytest.raises(TypeError):
        registry.register(object())


def test_register_rejects_duplicate_names():
    registry = ToolRegistry()
    registry.register(MockTool())

    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(MockTool())


def test_register_rejects_invalid_definition():
    class InvalidTool(Tool):
        @property
        def definition(self):
            return "invalid"

        async def execute(self, call):
            raise NotImplementedError

    registry = ToolRegistry()

    with pytest.raises(ToolRegistrationError):
        registry.register(InvalidTool())


def test_get_registered_tool():
    registry = ToolRegistry()
    tool = MockTool()
    registry.register(tool)

    assert registry.get("mock_tool") is tool


def test_get_unknown_tool_raises_error():
    registry = ToolRegistry()

    with pytest.raises(ToolNotFoundError):
        registry.get("unknown_tool")


def test_get_definition():
    registry = ToolRegistry()
    registry.register(MockTool())

    definition = registry.get_definition("mock_tool")

    assert isinstance(definition, ToolDefinition)
    assert definition.name == "mock_tool"


def test_contains_registered_tool():
    registry = ToolRegistry()
    registry.register(MockTool())

    assert registry.contains("mock_tool") is True
    assert registry.contains("unknown_tool") is False


def test_unregister_registered_tool():
    registry = ToolRegistry()
    registry.register(MockTool())

    assert registry.unregister("mock_tool") is True
    assert registry.count == 0
    assert registry.contains("mock_tool") is False


def test_unregister_unknown_tool_returns_false():
    registry = ToolRegistry()

    assert registry.unregister("unknown_tool") is False


def test_list_tools_returns_all_registered_tools():
    registry = ToolRegistry()
    first = MockTool("first_tool")
    second = MockTool("second_tool")

    registry.register(first)
    registry.register(second)

    tools = registry.list_tools()

    assert len(tools) == 2
    assert tools == (first, second)


def test_list_definitions_returns_all_definitions():
    registry = ToolRegistry()
    registry.register(MockTool("first_tool"))
    registry.register(MockTool("second_tool"))

    definitions = registry.list_definitions()

    assert len(definitions) == 2
    assert definitions[0].name == "first_tool"
    assert definitions[1].name == "second_tool"


def test_clear_removes_all_tools():
    registry = ToolRegistry()
    registry.register(MockTool("first_tool"))
    registry.register(MockTool("second_tool"))

    registry.clear()

    assert registry.count == 0
    assert registry.list_tools() == ()


@pytest.mark.parametrize("method_name", ["get", "contains", "unregister"])
@pytest.mark.parametrize("invalid_name", ["", " ", None, 123])
def test_registry_rejects_invalid_tool_names(method_name, invalid_name):
    registry = ToolRegistry()
    method = getattr(registry, method_name)

    with pytest.raises(ValueError):
        method(invalid_name)