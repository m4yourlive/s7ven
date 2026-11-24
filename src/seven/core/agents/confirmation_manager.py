
"""Manage asynchronous user confirmation workflows."""

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID

from seven.core.agents.confirmation import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationStatus,
)
from seven.core.agents.tools.models import ToolCall


class ConfirmationManager:
    """Create, resolve, expire, and consume confirmation requests."""

    def __init__(self) -> None:
        self._requests: dict[UUID, ConfirmationRequest] = {}
        self._futures: dict[
            UUID,
            asyncio.Future[ConfirmationDecision],
        ] = {}

    @property
    def pending_count(self) -> int:
        """Return the number of pending confirmation requests."""
        return sum(
            request.status is ConfirmationStatus.PENDING
            for request in self._requests.values()
        )

    def create_request(
        self,
        call: ToolCall,
        task_id: UUID | None = None,
        timeout_seconds: float = 60.0,
    ) -> ConfirmationRequest:
        """Create a confirmation request for a specific tool call."""
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall instance.")

        if task_id is not None and not isinstance(task_id, UUID):
            raise TypeError("task_id must be a UUID or None.")

        if (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
        ):
            raise TypeError("timeout_seconds must be a number.")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero.")

        loop = asyncio.get_running_loop()
        now = datetime.now(timezone.utc)

        request = ConfirmationRequest(
            tool_name=call.tool_name,
            call_id=call.call_id,
            arguments=call.arguments,
            task_id=task_id,
            created_at=now,
            expires_at=now + timedelta(seconds=timeout_seconds),
        )

        self._requests[request.request_id] = request
        self._futures[request.request_id] = loop.create_future()

        return request

    def get_request(
        self,
        request_id: UUID,
    ) -> ConfirmationRequest:
        """Retrieve a confirmation request."""
        if not isinstance(request_id, UUID):
            raise TypeError("request_id must be a UUID.")

        request = self._requests.get(request_id)

        if request is None:
            raise KeyError("Confirmation request was not found.")

        return request

    def list_pending(self) -> tuple[ConfirmationRequest, ...]:
        """Return all pending confirmation requests."""
        return tuple(
            request
            for request in self._requests.values()
            if request.status is ConfirmationStatus.PENDING
        )

    def resolve(
        self,
        request_id: UUID,
        approved: bool,
        decided_by: str = "user",
        reason: str | None = None,
    ) -> ConfirmationDecision:
        """Resolve a pending request with an approval or rejection."""
        if not isinstance(request_id, UUID):
            raise TypeError("request_id must be a UUID.")

        if not isinstance(approved, bool):
            raise TypeError("approved must be a boolean.")

        if not isinstance(decided_by, str) or not decided_by.strip():
            raise ValueError("decided_by cannot be empty.")

        request = self.get_request(request_id)

        if request.status is not ConfirmationStatus.PENDING:
            raise ValueError("Confirmation request is no longer pending.")

        now = datetime.now(timezone.utc)

        if now >= request.expires_at:
            return self._expire_request(request)

        status = (
            ConfirmationStatus.APPROVED
            if approved
            else ConfirmationStatus.REJECTED
        )

        decision = ConfirmationDecision(
            request_id=request_id,
            status=status,
            decided_by=decided_by,
            decided_at=now,
            reason=reason,
        )

        self._requests[request_id] = replace(
            request,
            status=status,
        )

        future = self._futures[request_id]

        if not future.done():
            future.set_result(decision)

        return decision

    async def wait_for_decision(
        self,
        request_id: UUID,
    ) -> ConfirmationDecision:
        """Wait asynchronously for a user's confirmation decision."""
        request = self.get_request(request_id)

        future = self._futures[request_id]

        if future.done():
            return future.result()

        if request.status is not ConfirmationStatus.PENDING:
            raise ValueError("Confirmation request is no longer pending.")

        remaining = (
            request.expires_at - datetime.now(timezone.utc)
        ).total_seconds()

        if remaining <= 0:
            return self._expire_request(request)

        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=remaining,
            )
        except TimeoutError:
            if future.done():
                return future.result()

            return self._expire_request(
                self.get_request(request_id)
            )

    def consume_approval(
        self,
        request_id: UUID,
        call: ToolCall,
    ) -> bool:
        """Consume an approval exactly once for its original tool call."""
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall instance.")

        request = self.get_request(request_id)

        if request.status is not ConfirmationStatus.APPROVED:
            return False

        if (
            request.call_id != call.call_id
            or request.tool_name != call.tool_name
            or dict(request.arguments) != dict(call.arguments)
        ):
            return False

        self._requests[request_id] = replace(
            request,
            status=ConfirmationStatus.CONSUMED,
        )

        return True

    def _expire_request(
        self,
        request: ConfirmationRequest,
    ) -> ConfirmationDecision:
        """Expire a pending request."""
        if request.status is not ConfirmationStatus.PENDING:
            raise ValueError("Only pending requests can expire.")

        decision = ConfirmationDecision(
            request_id=request.request_id,
            status=ConfirmationStatus.EXPIRED,
            decided_by="system",
            reason="Confirmation request expired.",
        )

        self._requests[request.request_id] = replace(
            request,
            status=ConfirmationStatus.EXPIRED,
        )

        future = self._futures[request.request_id]

        if not future.done():
            future.set_result(decision)

        return decision
