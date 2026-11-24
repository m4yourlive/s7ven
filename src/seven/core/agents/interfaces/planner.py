"""Planner interface for autonomous agent task decomposition."""

from abc import ABC, abstractmethod

from seven.core.agents.models import AgentPlan, AgentTask


class Planner(ABC):
    """Define the contract for autonomous task planning."""

    @abstractmethod
    async def plan(self, task: AgentTask) -> AgentPlan:
        """Decompose a task into an executable plan."""
        raise NotImplementedError