
"""Execution engine for autonomous agent tools."""

import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID

from seven.core.agents.tools.context import ToolExecutionContext
from seven.core.agents.tools.events import (
    ToolExecutionCompleted,
    ToolExecutionFailed,
    ToolExecutionStarted,
    ToolExecutionTimedOut,
)
from seven.core.agents.tools.exceptions import ToolArgumentValidationError
from seven.core.agents.tools.lifecycle import (
    ToolExecutionStatus,
    ToolExecutionTracker,
)
from seven.core.agents.tools.models import ToolCall, ToolResult
from seven.core.agents.tools.registry import ToolRegistry
from seven.core.agents.tools.validator import ToolArgumentValidator
from seven.core.events import EventBus


class ToolExecutionEngine:
    """Execute registered tools and track their execution lifecycle."""

    def __init__(
        self,
        registry: ToolRegistry,
        validator: ToolArgumentValidator | None = None,
        max_concurrency: int = 5,
        tracker: ToolExecutionTracker | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("registry must be a ToolRegistry instance.")

        if validator is not None and not isinstance(
            validator, ToolArgumentValidator
        ):
            raise TypeError(
                "validator must be a ToolArgumentValidator instance."
            )

        if (
            not isinstance(max_concurrency, int)
            or isinstance(max_concurrency, bool)
        ):
            raise TypeError("max_concurrency must be an integer.")

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be greater than zero.")

        if tracker is not None and not isinstance(
            tracker, ToolExecutionTracker
        ):
            raise TypeError(
                "tracker must be a ToolExecutionTracker instance."
            )

        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus instance or None.")

        self._registry = registry
        self._validator = validator or ToolArgumentValidator()
        self._max_concurrency = max_concurrency
        self._tracker = tracker or ToolExecutionTracker()
        self._event_bus = event_bus

    @property
    def registry(self) -> ToolRegistry:
        """Return the tool registry."""
        return self._registry

    @property
    def validator(self) -> ToolArgumentValidator:
        """Return the argument validator."""
        return self._validator

    @property
    def max_concurrency(self) -> int:
        """Return the maximum concurrent execution count."""
        return self._max_concurrency

    @property
    def tracker(self) -> ToolExecutionTracker:
        """Return the execution tracker."""
        return self._tracker

    @property
    def metrics(self):
        """Return aggregated execution metrics."""
        return self._tracker.metrics

    @property
    def event_bus(self) -> EventBus | None:
        """Return the configured event bus."""
        return self._event_bus

    def _publish(self, event: object) -> None:
        """Publish an event without interrupting tool execution."""
        if self._event_bus is None:
            return

        try:
            self._event_bus.publish(event)
        except Exception:
            pass

    def _publish_result(
        self,
        result: ToolResult,
        execution_id: UUID,
        *,
        timed_out: bool = False,
    ) -> None:
        """Publish the terminal event for an execution."""
        if timed_out:
            event = ToolExecutionTimedOut(
                result=result,
                execution_id=execution_id,
            )
        elif result.success:
            event = ToolExecutionCompleted(
                result=result,
                execution_id=execution_id,
            )
        else:
            event = ToolExecutionFailed(
                result=result,
                execution_id=execution_id,
            )

        self._publish(event)

    def _finish_tracking(
        self,
        execution_id: UUID,
        status: ToolExecutionStatus,
        error: str | None = None,
    ) -> None:
        """Finalize execution tracking."""
        self._tracker.finish(
            execution_id,
            status,
            error=error,
        )

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """Validate, execute, track, and publish a tool call."""
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall instance.")

        if context is None:
            context = ToolExecutionContext()
        elif not isinstance(context, ToolExecutionContext):
            raise TypeError(
                "context must be a ToolExecutionContext instance or None."
            )

        execution_id = context.execution_id
        execution_metadata = self._context_metadata(context)

        self._tracker.start(call, context)

        self._publish(
            ToolExecutionStarted(
                call=call,
                context=context,
            )
        )

        try:
            result, timed_out = await self._execute_call(
                call,
                context,
                execution_metadata,
            )

            if timed_out:
                status = ToolExecutionStatus.TIMED_OUT
            elif result.success:
                status = ToolExecutionStatus.SUCCEEDED
            else:
                status = ToolExecutionStatus.FAILED

            self._finish_tracking(
                execution_id,
                status,
                error=result.error,
            )

            self._publish_result(
                result,
                execution_id,
                timed_out=timed_out,
            )

            return result

        except asyncio.CancelledError:
            self._finish_tracking(
                execution_id,
                ToolExecutionStatus.CANCELLED,
                error="Tool execution was cancelled.",
            )
            raise

    async def _execute_call(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        execution_metadata: dict[str, Any],
    ) -> tuple[ToolResult, bool]:
        """Execute one tool call and normalize its result."""
        try:
            tool = self._registry.get(call.tool_name)
        except Exception as error:
            return (
                self._failure_result(
                    call,
                    str(error),
                    execution_metadata,
                ),
                False,
            )

        try:
            self._validator.validate(
                arguments=call.arguments,
                schema=tool.definition.parameters,
            )
        except ToolArgumentValidationError as error:
            return (
                self._failure_result(
                    call,
                    f"Invalid tool arguments: {error}",
                    {
                        **execution_metadata,
                        "validation_error": True,
                    },
                ),
                False,
            )

        try:
            result = await asyncio.wait_for(
                tool.execute(call),
                timeout=tool.definition.timeout_seconds,
            )

            if not isinstance(result, ToolResult):
                return (
                    self._failure_result(
                        call,
                        "Tool execution must return a ToolResult instance.",
                        execution_metadata,
                    ),
                    False,
                )

            if result.call_id != call.call_id:
                return (
                    self._failure_result(
                        call,
                        "Tool result call ID does not match the original call.",
                        execution_metadata,
                    ),
                    False,
                )

            if result.tool_name != call.tool_name:
                return (
                    self._failure_result(
                        call,
                        "Tool result name does not match the requested tool.",
                        execution_metadata,
                    ),
                    False,
                )

            normalized = ToolResult(
                call_id=result.call_id,
                tool_name=result.tool_name,
                success=result.success,
                output=result.output,
                error=result.error,
                metadata={
                    **dict(result.metadata),
                    **execution_metadata,
                },
            )

            return normalized, False

        except TimeoutError:
            return (
                self._failure_result(
                    call,
                    (
                        f"Tool '{call.tool_name}' exceeded its execution "
                        f"timeout of {tool.definition.timeout_seconds} seconds."
                    ),
                    {
                        **execution_metadata,
                        "timeout": True,
                    },
                ),
                True,
            )

        except asyncio.CancelledError:
            raise

        except Exception as error:
            return (
                self._failure_result(
                    call,
                    f"Tool execution failed: {error}",
                    execution_metadata,
                ),
                False,
            )

    async def execute_many(
        self,
        calls: Sequence[ToolCall],
        context: ToolExecutionContext | None = None,
    ) -> tuple[ToolResult, ...]:
        """Execute multiple calls with bounded concurrency."""
        if not isinstance(calls, Sequence) or isinstance(
            calls, (str, bytes)
        ):
            raise TypeError(
                "calls must be a sequence of ToolCall instances."
            )

        if not all(isinstance(call, ToolCall) for call in calls):
            raise TypeError(
                "Every item in calls must be a ToolCall instance."
            )

        if context is not None and not isinstance(
            context, ToolExecutionContext
        ):
            raise TypeError(
                "context must be a ToolExecutionContext instance or None."
            )

        if not calls:
            return ()

        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def execute_one(call: ToolCall) -> ToolResult:
            execution_context = ToolExecutionContext(
                task_id=context.task_id if context else None,
                metadata=context.metadata if context else {},
            )

            async with semaphore:
                return await self.execute(call, execution_context)

        results = await asyncio.gather(
            *(execute_one(call) for call in calls)
        )

        return tuple(results)

    @staticmethod
    def _context_metadata(
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Convert execution context into result metadata."""
        return {
            "execution_context": {
                "execution_id": str(context.execution_id),
                "task_id": (
                    str(context.task_id)
                    if context.task_id is not None
                    else None
                ),
                "metadata": dict(context.metadata),
            }
        }

    @staticmethod
    def _failure_result(
        call: ToolCall,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create a standardized failed result."""
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=False,
            error=error,
            metadata=metadata or {},
        )
