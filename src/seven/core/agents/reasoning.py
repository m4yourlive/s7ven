
"""Domain models for the autonomous reasoning loop."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID


class ReasoningStatus(StrEnum):
    """Available reasoning loop states."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class ReasoningStep:
    """Record one iteration of the reasoning loop."""

    iteration: int
    assistant_content: str = ""
    tool_call_ids: tuple[str, ...] = ()
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if (
            not isinstance(self.iteration, int)
            or isinstance(self.iteration, bool)
        ):
            raise TypeError("iteration must be an integer.")

        if self.iteration < 1:
            raise ValueError("iteration must be greater than zero.")

        if not isinstance(self.assistant_content, str):
            raise TypeError("assistant_content must be a string.")

        if not isinstance(self.tool_call_ids, tuple):
            raise TypeError("tool_call_ids must be a tuple.")

        if not all(
            isinstance(call_id, str) and call_id.strip()
            for call_id in self.tool_call_ids
        ):
            raise ValueError(
                "tool_call_ids must contain non-empty strings."
            )

        if len(set(self.tool_call_ids)) != len(self.tool_call_ids):
            raise ValueError("tool_call_ids must be unique.")

        _validate_timestamp(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class ReasoningResult:
    """Represent the final outcome of a reasoning loop."""

    task_id: UUID
    status: ReasoningStatus
    output: str = ""
    iterations: int = 0
    steps: tuple[ReasoningStep, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None
    completed_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID.")

        if not isinstance(self.status, ReasoningStatus):
            raise TypeError("status must be a ReasoningStatus.")

        if not isinstance(self.output, str):
            raise TypeError("output must be a string.")

        if (
            not isinstance(self.iterations, int)
            or isinstance(self.iterations, bool)
        ):
            raise TypeError("iterations must be an integer.")

        if self.iterations < 0:
            raise ValueError("iterations cannot be negative.")

        if not isinstance(self.steps, tuple):
            raise TypeError("steps must be a tuple.")

        if not all(
            isinstance(step, ReasoningStep)
            for step in self.steps
        ):
            raise TypeError(
                "steps must contain ReasoningStep instances."
            )

        if len(self.steps) != self.iterations:
            raise ValueError(
                "iterations must match the number of reasoning steps."
            )

        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("error must be a string or None.")

        if self.status == ReasoningStatus.FAILED and not self.error:
            raise ValueError("Failed reasoning results require an error.")

        if self.status != ReasoningStatus.FAILED and self.error is not None:
            raise ValueError(
                "Only failed reasoning results can contain an error."
            )

        _validate_timestamp(self.completed_at, "completed_at")

        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata, "metadata"),
        )


def _validate_timestamp(value: datetime, field_name: str) -> None:
    """Validate a timezone-aware timestamp."""

    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime instance.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")


def _freeze_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    """Validate and freeze a mapping."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping.")

    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings.")

    return MappingProxyType(dict(value))
