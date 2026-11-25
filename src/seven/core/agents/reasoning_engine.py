
"""Autonomous reasoning loop with integrated safety enforcement."""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from seven.core.ai.engine import AIEngine
from seven.core.ai.models import (
    AIMessage,
    FunctionDefinition,
    GenerationRequest,
    MessageRole,
    ToolChoice,
)
from seven.core.agents.budget import (
    ExecutionBudget,
    ExecutionBudgetExceededError,
)
from seven.core.agents.budget_manager import ExecutionBudgetManager
from seven.core.agents.confirmation_manager import ConfirmationManager
from seven.core.agents.models import AgentTask
from seven.core.agents.reasoning import (
    ReasoningResult,
    ReasoningStatus,
    ReasoningStep,
)
from seven.core.agents.reasoning_config import ReasoningConfig
from seven.core.agents.reasoning_exceptions import (
    ReasoningProviderError,
)
from seven.core.agents.safety import SafetyDecision
from seven.core.agents.safety_policy import SafetyPolicyEngine
from seven.core.agents.tools.context import ToolExecutionContext
from seven.core.agents.tools.engine import ToolExecutionEngine
from seven.core.agents.tools.models import ToolCall, ToolResult


class ReasoningEngine:
    """Coordinate AI reasoning and safely controlled tool execution."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an autonomous AI agent. "
        "Analyze the user's goal, use available tools when necessary, "
        "and provide a clear final answer when the task is complete. "
        "Do not claim that an action was completed unless a tool result "
        "confirms its successful execution. "
        "Never attempt to bypass tool permissions or user confirmations."
    )

    def __init__(
        self,
        ai_engine: AIEngine,
        tool_execution_engine: ToolExecutionEngine,
        model: str,
        config: ReasoningConfig | None = None,
        system_prompt: str | None = None,
        safety_policy: SafetyPolicyEngine | None = None,
        confirmation_manager: ConfirmationManager | None = None,
        execution_budget: ExecutionBudget | None = None,
        confirmation_timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(ai_engine, AIEngine):
            raise TypeError("ai_engine must be an AIEngine instance.")

        if not isinstance(tool_execution_engine, ToolExecutionEngine):
            raise TypeError(
                "tool_execution_engine must be a ToolExecutionEngine instance."
            )

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string.")

        if config is not None and not isinstance(config, ReasoningConfig):
            raise TypeError(
                "config must be a ReasoningConfig instance or None."
            )

        if system_prompt is not None and (
            not isinstance(system_prompt, str) or not system_prompt.strip()
        ):
            raise ValueError(
                "system_prompt must be a non-empty string or None."
            )

        if safety_policy is not None and not isinstance(
            safety_policy,
            SafetyPolicyEngine,
        ):
            raise TypeError(
                "safety_policy must be a SafetyPolicyEngine instance or None."
            )

        if confirmation_manager is not None and not isinstance(
            confirmation_manager,
            ConfirmationManager,
        ):
            raise TypeError(
                "confirmation_manager must be a ConfirmationManager instance "
                "or None."
            )

        if execution_budget is not None and not isinstance(
            execution_budget,
            ExecutionBudget,
        ):
            raise TypeError(
                "execution_budget must be an ExecutionBudget instance or None."
            )

        if (
            not isinstance(confirmation_timeout_seconds, (int, float))
            or isinstance(confirmation_timeout_seconds, bool)
        ):
            raise TypeError("confirmation_timeout_seconds must be a number.")

        if (
            confirmation_timeout_seconds <= 0
            or confirmation_timeout_seconds == float("inf")
            or confirmation_timeout_seconds != confirmation_timeout_seconds
        ):
            raise ValueError(
                "confirmation_timeout_seconds must be finite and positive."
            )

        self._ai_engine = ai_engine
        self._tool_execution_engine = tool_execution_engine
        self._model = model
        self._config = config or ReasoningConfig()
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        self._safety_policy = safety_policy or SafetyPolicyEngine()
        self._confirmation_manager = (
            confirmation_manager or ConfirmationManager()
        )
        self._confirmation_timeout_seconds = float(
            confirmation_timeout_seconds
        )

        self._execution_budget = execution_budget or ExecutionBudget(
            max_tool_calls=self._config.max_tool_calls,
            max_execution_seconds=self._config.timeout_seconds,
        )

    @property
    def ai_engine(self) -> AIEngine:
        """Return the configured AI engine."""
        return self._ai_engine

    @property
    def tool_execution_engine(self) -> ToolExecutionEngine:
        """Return the configured tool execution engine."""
        return self._tool_execution_engine

    @property
    def config(self) -> ReasoningConfig:
        """Return the reasoning configuration."""
        return self._config

    @property
    def model(self) -> str:
        """Return the configured model identifier."""
        return self._model

    @property
    def safety_policy(self) -> SafetyPolicyEngine:
        """Return the active safety policy engine."""
        return self._safety_policy

    @property
    def confirmation_manager(self) -> ConfirmationManager:
        """Return the confirmation manager used by this engine."""
        return self._confirmation_manager

    @property
    def execution_budget(self) -> ExecutionBudget:
        """Return the configured execution budget."""
        return self._execution_budget

    async def reason(self, task: AgentTask) -> ReasoningResult:
        """Execute an autonomous reasoning loop for a task."""
        if not isinstance(task, AgentTask):
            raise TypeError("task must be an AgentTask instance.")

        steps: list[ReasoningStep] = []
        budget_manager = ExecutionBudgetManager(
            budget=self._execution_budget,
            task_id=task.task_id,
        )

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                return await self._run_loop(
                    task=task,
                    steps=steps,
                    budget_manager=budget_manager,
                )

        except TimeoutError:
            return self._build_result(
                task=task,
                status=ReasoningStatus.FAILED,
                output="",
                steps=steps,
                error=(
                    "Reasoning exceeded its configured timeout of "
                    f"{self._config.timeout_seconds} seconds."
                ),
            )

        except asyncio.CancelledError:
            raise

        except ReasoningProviderError as error:
            return self._build_result(
                task=task,
                status=ReasoningStatus.FAILED,
                output="",
                steps=steps,
                error=str(error),
            )

    async def _run_loop(
        self,
        task: AgentTask,
        steps: list[ReasoningStep],
        budget_manager: ExecutionBudgetManager,
    ) -> ReasoningResult:
        """Run the iterative reasoning and tool execution cycle."""
        messages = [
            AIMessage(
                role=MessageRole.SYSTEM,
                content=self._system_prompt,
            ),
            *task.context,
            AIMessage(
                role=MessageRole.USER,
                content=task.goal,
            ),
        ]

        definitions = self._build_function_definitions()
        total_tool_calls = 0
        consecutive_failures = 0

        for iteration in range(1, self._config.max_iterations + 1):
            request = GenerationRequest(
                messages=tuple(messages),
                model=self._model,
                tools=definitions,
                tool_choice=ToolChoice.AUTO,
            )

            try:
                response = await self._ai_engine.generate(request)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise ReasoningProviderError(
                    f"AI generation failed: {error}"
                ) from error

            function_calls = tuple(response.tool_calls or ())
            assistant_content = response.content or ""

            messages.append(
                AIMessage(
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                    tool_calls=function_calls,
                )
            )

            step = ReasoningStep(
                iteration=iteration,
                assistant_content=assistant_content,
                tool_call_ids=tuple(
                    function_call.call_id
                    for function_call in function_calls
                ),
                created_at=datetime.now(timezone.utc),
            )
            steps.append(step)

            if not function_calls:
                if not assistant_content.strip():
                    return self._build_result(
                        task=task,
                        status=ReasoningStatus.FAILED,
                        output="",
                        steps=steps,
                        error="The AI provider returned an empty final response.",
                    )

                return self._build_result(
                    task=task,
                    status=ReasoningStatus.COMPLETED,
                    output=assistant_content,
                    steps=steps,
                )

            if len(function_calls) > self._config.max_tool_calls_per_iteration:
                return self._build_result(
                    task=task,
                    status=ReasoningStatus.LIMIT_REACHED,
                    output=(
                        "The reasoning loop reached its per-iteration "
                        "tool call limit."
                    ),
                    steps=steps,
                )

            if (
                total_tool_calls + len(function_calls)
                > self._config.max_tool_calls
            ):
                return self._build_result(
                    task=task,
                    status=ReasoningStatus.LIMIT_REACHED,
                    output="The reasoning loop reached its total tool call limit.",
                    steps=steps,
                )

            total_tool_calls += len(function_calls)

            results = await self._execute_function_calls(
                task=task,
                function_calls=function_calls,
                budget_manager=budget_manager,
            )

            for function_call, result in zip(
                function_calls,
                results,
                strict=True,
            ):
                messages.append(
                    AIMessage(
                        role=MessageRole.TOOL,
                        content=self._serialize_tool_result(result),
                        tool_call_id=function_call.call_id,
                    )
                )

                if result.success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1

                if (
                    consecutive_failures
                    >= self._config.max_consecutive_failures
                ):
                    return self._build_result(
                        task=task,
                        status=ReasoningStatus.FAILED,
                        output="",
                        steps=steps,
                        error=(
                            "The reasoning loop stopped after reaching "
                            "the consecutive tool failure limit."
                        ),
                    )

        return self._build_result(
            task=task,
            status=ReasoningStatus.LIMIT_REACHED,
            output="The reasoning loop reached its maximum iteration limit.",
            steps=steps,
        )

    def _build_function_definitions(
        self,
    ) -> tuple[FunctionDefinition, ...]:
        """Convert registered tool definitions into AI function definitions."""
        definitions = self._tool_execution_engine.registry.list_definitions()

        return tuple(
            FunctionDefinition(
                name=definition.name,
                description=definition.description,
                parameters=definition.parameters,
            )
            for definition in definitions
        )

    async def _execute_function_calls(
        self,
        task: AgentTask,
        function_calls: tuple[Any, ...],
        budget_manager: ExecutionBudgetManager,
    ) -> tuple[ToolResult, ...]:
        """Execute tool calls sequentially through safety enforcement."""
        context = ToolExecutionContext(task_id=task.task_id)
        results: list[ToolResult] = []

        for function_call in function_calls:
            call = ToolCall(
                tool_name=function_call.name,
                arguments=function_call.arguments,
                call_id=uuid4(),
            )

            result = await self._execute_single_tool_call(
                call=call,
                context=context,
                budget_manager=budget_manager,
            )
            results.append(result)

        return tuple(results)

    async def _execute_single_tool_call(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        budget_manager: ExecutionBudgetManager,
    ) -> ToolResult:
        """Apply permission, confirmation, and budget checks before execution."""
        registry = self._tool_execution_engine.registry

        try:
            definition = registry.get_definition(call.tool_name)
        except Exception as error:
            return self._failure_result(
                call,
                f"Tool definition lookup failed: {error}",
            )

        policy_result = self._safety_policy.evaluate(
            definition,
            context,
        )

        if policy_result.decision is SafetyDecision.DENY:
            return self._failure_result(
                call,
                f"Tool execution denied: {policy_result.reason}",
                {
                    "safety_decision": policy_result.decision.value,
                    "execution_blocked": True,
                },
            )

        if policy_result.decision is SafetyDecision.REQUIRE_APPROVAL:
            try:
                confirmation = self._confirmation_manager.create_request(
                    call,
                    task_id=context.task_id,
                    timeout_seconds=self._confirmation_timeout_seconds,
                )

                decision = (
                    await self._confirmation_manager.wait_for_decision(
                        confirmation.request_id
                    )
                )

            except asyncio.CancelledError:
                raise
            except Exception as error:
                return self._failure_result(
                    call,
                    f"Confirmation workflow failed: {error}",
                    {
                        "confirmation_required": True,
                        "execution_blocked": True,
                    },
                )

            if not decision.approved:
                return self._failure_result(
                    call,
                    (
                        "This tool requires user confirmation."
                        "Execution was not approved; action was not performed."
                    ),
                    {
                        "confirmation_required": True,
                        "confirmation_status": decision.status.value,
                        "confirmation_request_id": str(
                            confirmation.request_id
                        ),
                        "execution_blocked": True,
                    },
                )

            if not self._confirmation_manager.consume_approval(
                confirmation.request_id,
                call,
            ):
                return self._failure_result(
                    call,
                    "Confirmation approval could not be validated or consumed.",
                    {
                        "confirmation_required": True,
                        "execution_blocked": True,
                    },
                )

        try:
            timeout = budget_manager.get_tool_timeout()
            lease = budget_manager.start_tool_call(call.tool_name)
        except ExecutionBudgetExceededError as error:
            return self._failure_result(
                call,
                f"Execution budget exceeded: {error}",
                {
                    "budget_exceeded": True,
                    "execution_blocked": True,
                },
            )

        result: ToolResult | None = None
        budget_error: ExecutionBudgetExceededError | None = None

        try:
            result = await asyncio.wait_for(
                self._tool_execution_engine.execute(call, context),
                timeout=timeout,
            )

        except TimeoutError:
            result = self._failure_result(
                call,
                (
                    f"Tool '{call.tool_name}' exceeded its remaining "
                    "execution budget."
                ),
                {
                    "budget_exceeded": True,
                    "timeout": True,
                },
            )

        except asyncio.CancelledError:
            raise

        except Exception as error:
            result = self._failure_result(
                call,
                f"Tool execution failed: {error}",
            )

        finally:
            try:
                budget_manager.complete_tool_call(lease)
            except ExecutionBudgetExceededError as error:
                budget_error = error

        if budget_error is not None:
            return self._failure_result(
                call,
                f"Execution budget exceeded: {budget_error}",
                {
                    "budget_exceeded": True,
                    "execution_blocked": True,
                },
            )

        if result is None:
            return self._failure_result(
                call,
                "Tool execution did not produce a result.",
            )

        return result

    @staticmethod
    def _failure_result(
        call: ToolCall,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """Create a standardized failed tool result."""
        return ToolResult(
            call_id=call.call_id,
            tool_name=call.tool_name,
            success=False,
            error=error,
            metadata=metadata or {},
        )

    @staticmethod
    def _serialize_tool_result(result: ToolResult) -> str:
        """Serialize a tool result into a message for the AI provider."""
        payload: dict[str, Any] = {
            "tool_name": result.tool_name,
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "metadata": dict(result.metadata),
        }

        try:
            return json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            return json.dumps(
                {
                    "tool_name": result.tool_name,
                    "success": result.success,
                    "output": str(result.output),
                    "error": result.error,
                },
                ensure_ascii=False,
            )

    @staticmethod
    def _build_result(
        task: AgentTask,
        status: ReasoningStatus,
        output: str,
        steps: list[ReasoningStep],
        error: str | None = None,
    ) -> ReasoningResult:
        """Build a validated reasoning result."""
        return ReasoningResult(
            task_id=task.task_id,
            status=status,
            output=output,
            iterations=len(steps),
            steps=tuple(steps),
            metadata={
                "tool_execution_count": sum(
                    len(step.tool_call_ids)
                    for step in steps
                ),
            },
            error=error,
            completed_at=datetime.now(timezone.utc),
        )
