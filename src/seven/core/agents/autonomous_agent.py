
"""Core orchestration service for the S7VEN autonomous agent."""

import asyncio
from collections.abc import Mapping
from uuid import UUID

from seven.core.agents.events import (
    AgentPlanCreated,
    AgentPlanningStarted,
    AgentStepCompleted,
    AgentStepStarted,
    AgentTaskCancelled,
    AgentTaskCompleted,
    AgentTaskFailed,
    AgentTaskStarted,
)
from seven.core.agents.exceptions import (
    AgentExecutionError,
    AgentPlanningError,
    AgentTaskError,
)
from seven.core.agents.interfaces import Executor, Planner
from seven.core.agents.models import AgentResult, AgentTask
from seven.core.agents.state import AgentState, AgentStatus
from seven.core.events import Event, EventBus


class AutonomousAgent:
    """Orchestrate autonomous tasks and publish lifecycle events."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        event_bus: EventBus | None = None,
    ) -> None:
        if not isinstance(planner, Planner):
            raise TypeError("planner must implement the Planner interface.")
        if not isinstance(executor, Executor):
            raise TypeError("executor must implement the Executor interface.")
        if event_bus is not None and not isinstance(event_bus, EventBus):
            raise TypeError("event_bus must be an EventBus instance or None.")

        self._planner = planner
        self._executor = executor
        self._event_bus = event_bus
        self._states: dict[UUID, AgentState] = {}
        self._active_tasks: dict[UUID, asyncio.Task[AgentResult]] = {}
        self._lock = asyncio.Lock()

    @property
    def planner(self) -> Planner:
        return self._planner

    @property
    def executor(self) -> Executor:
        return self._executor

    @property
    def event_bus(self) -> EventBus | None:
        return self._event_bus

    @property
    def active_count(self) -> int:
        return len(self._active_tasks)

    def get_state(self, task_id: UUID) -> AgentState | None:
        return self._states.get(task_id)

    def _publish(self, event: Event) -> None:
        """Publish an event when an event bus is configured."""
        if self._event_bus is not None:
            self._event_bus.publish(event)

    async def run(self, task: AgentTask) -> AgentResult:
        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask instance.")

        async with self._lock:
            if task.task_id in self._states:
                raise AgentTaskError(
                    f"Task '{task.task_id}' has already been submitted."
                )

            state = AgentState(task_id=task.task_id)
            self._states[task.task_id] = state

            current_task = asyncio.current_task()
            if current_task is not None:
                self._active_tasks[task.task_id] = current_task

        self._publish(AgentTaskStarted(task_id=task.task_id))

        try:
            return await self._execute_task(task, state)
        finally:
            async with self._lock:
                self._active_tasks.pop(task.task_id, None)

    async def cancel(self, task_id: UUID) -> bool:
        async with self._lock:
            running_task = self._active_tasks.get(task_id)
            state = self._states.get(task_id)

            if running_task is None or state is None:
                return False

            if running_task.done():
                return False

            running_task.cancel()
            return True

    async def _execute_task(
        self,
        task: AgentTask,
        state: AgentState,
    ) -> AgentResult:
        try:
            state.transition(AgentStatus.PLANNING)
            self._publish(
                AgentPlanningStarted(task_id=task.task_id)
            )

            plan = await self._planner.plan(task)

            if plan.task_id != task.task_id:
                raise AgentPlanningError(
                    "The planner returned a plan for a different task."
                )

            state.set_plan(plan)
            self._publish(
                AgentPlanCreated(
                    task_id=task.task_id,
                    plan=plan,
                )
            )

            state.transition(AgentStatus.RUNNING)

        except asyncio.CancelledError:
            state.cancel()
            self._publish(
                AgentTaskCancelled(task_id=task.task_id)
            )
            raise

        except Exception as error:
            result = self._build_failure_result(
                task=task,
                state=state,
                error=AgentPlanningError(
                    f"Task planning failed: {error}"
                ),
            )
            state.result = result
            state.error = result.error
            state.transition(AgentStatus.FAILED)

            self._publish(
                AgentTaskFailed(
                    task_id=task.task_id,
                    result=result,
                )
            )
            return result

        outputs: dict[UUID, str] = {}
        completed_step_ids: list[UUID] = []
        current_step_id: UUID | None = None

        try:
            for step in sorted(
                plan.steps,
                key=lambda item: item.position,
            ):
                current_step_id = step.step_id
                state.start_step(step.step_id)

                self._publish(
                    AgentStepStarted(
                        task_id=task.task_id,
                        step=step,
                    )
                )

                step_output = await self._executor.execute(
                    task,
                    step,
                    outputs,
                )

                if not isinstance(step_output, str):
                    raise AgentExecutionError(
                        "Executor output must be a string."
                    )

                outputs[step.step_id] = step_output
                completed_step_ids.append(step.step_id)
                state.complete_step(step.step_id)

                self._publish(
                    AgentStepCompleted(
                        task_id=task.task_id,
                        step=step,
                        output=step_output,
                    )
                )

            result = AgentResult(
                task_id=task.task_id,
                success=True,
                output=outputs[completed_step_ids[-1]],
                completed_step_ids=tuple(completed_step_ids),
                metadata={
                    "step_count": len(plan.steps),
                    "outputs": dict(outputs),
                },
            )

            state.finish(result)

            self._publish(
                AgentTaskCompleted(
                    task_id=task.task_id,
                    result=result,
                )
            )

            return result

        except asyncio.CancelledError:
            state.cancel()
            self._publish(
                AgentTaskCancelled(task_id=task.task_id)
            )
            raise

        except Exception as error:
            result = self._build_failure_result(
                task=task,
                state=state,
                error=AgentExecutionError(
                    f"Task execution failed: {error}"
                ),
                completed_step_ids=completed_step_ids,
                failed_step_id=current_step_id,
                outputs=outputs,
            )

            state.finish(result)

            self._publish(
                AgentTaskFailed(
                    task_id=task.task_id,
                    result=result,
                )
            )

            return result

    @staticmethod
    def _build_failure_result(
        task: AgentTask,
        state: AgentState,
        error: AgentTaskError,
        completed_step_ids: list[UUID] | None = None,
        failed_step_id: UUID | None = None,
        outputs: Mapping[UUID, str] | None = None,
    ) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            success=False,
            output="",
            completed_step_ids=tuple(completed_step_ids or ()),
            failed_step_id=failed_step_id,
            error=str(error),
            metadata={
                "status": state.status.value,
                "outputs": dict(outputs or {}),
            },
        )
