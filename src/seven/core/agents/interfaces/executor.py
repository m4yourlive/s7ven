
"""Executor interface for autonomous agent step execution."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from uuid import UUID

from seven.core.agents.models import AgentStep, AgentTask


class Executor(ABC):
    """Define the contract for autonomous step execution."""

    @abstractmethod
    async def execute(
        self,
        task: AgentTask,
        step: AgentStep,
        context: Mapping[UUID, str],
    ) -> str:
        """Execute one planned step and return its output."""
        raise NotImplementedError
