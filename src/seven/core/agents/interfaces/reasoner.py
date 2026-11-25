
"""Reasoner interface for autonomous agent reasoning."""

from abc import ABC, abstractmethod

from seven.core.agents.models import AgentTask
from seven.core.agents.reasoning import ReasoningResult


class Reasoner(ABC):
    """Define the contract for autonomous task reasoning."""

    @abstractmethod
    async def reason(self, task: AgentTask) -> ReasoningResult:
        """Execute a reasoning loop for the supplied task."""
        raise NotImplementedError
