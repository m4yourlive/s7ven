
"""Execution safety policy for workflow automation."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class WorkflowExecutionPolicy:
    """Define safety limits for workflow execution."""

    max_steps: int = 100
    max_duration_seconds: float = 300.0
    allowed_actions: frozenset[str] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("Maximum steps must be a positive integer.")

        if (
            isinstance(self.max_duration_seconds, bool)
            or not isinstance(self.max_duration_seconds, (int, float))
            or not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("Maximum duration must be a positive finite number.")

        if self.allowed_actions is not None:
            if not isinstance(self.allowed_actions, (set, frozenset)):
                raise TypeError("Allowed actions must be a set or frozenset.")

            normalized_actions = frozenset(self.allowed_actions)

            if any(
                not isinstance(action, str) or not action.strip()
                for action in normalized_actions
            ):
                raise ValueError("Allowed actions must contain non-empty strings.")

            object.__setattr__(self, "allowed_actions", normalized_actions)
