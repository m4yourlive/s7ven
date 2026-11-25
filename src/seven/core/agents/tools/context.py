"""Execution context models for autonomous agent tools."""

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolExecutionContext:
    """Carry execution metadata across a tool invocation."""

    task_id: UUID | None = None
    execution_id: UUID = field(default_factory=uuid4)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.task_id is not None and not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID or None.")

        if not isinstance(self.execution_id, UUID):
            raise TypeError("execution_id must be a UUID.")

        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        if not all(isinstance(key, str) for key in self.metadata):
            raise TypeError("metadata keys must be strings.")

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )