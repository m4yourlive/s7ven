"""Lifecycle events for autonomous agent tool execution."""

from dataclasses import dataclass
from uuid import UUID

from seven.core.agents.tools.context import ToolExecutionContext
from seven.core.agents.tools.models import ToolCall, ToolResult
from seven.core.events import Event


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionStarted(Event):
    """Emitted when a tool execution begins."""

    call: ToolCall
    context: ToolExecutionContext


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionCompleted(Event):
    """Emitted when a tool execution succeeds."""

    result: ToolResult
    execution_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionFailed(Event):
    """Emitted when a tool execution fails."""

    result: ToolResult
    execution_id: UUID


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionTimedOut(Event):
    """Emitted when a tool execution exceeds its timeout."""

    result: ToolResult
    execution_id: UUID