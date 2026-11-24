
"""Tests for the user confirmation workflow."""

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from seven.core.agents.confirmation import (
    ConfirmationDecision,
    ConfirmationRequest,
    ConfirmationStatus,
)
from seven.core.agents.confirmation_manager import ConfirmationManager
from seven.core.agents.tools.models import ToolCall


def make_tool_call(
    *,
    tool_name: str = "send_message",
    arguments: dict | None = None,
) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        arguments=arguments or {"recipient": "user@example.com"},
        call_id=uuid4(),
    )


class TestConfirmationRequest:
    def test_request_creation(self) -> None:
        now = datetime.now(timezone.utc)

        request = ConfirmationRequest(
            tool_name="send_message",
            call_id=uuid4(),
            arguments={"recipient": "user@example.com"},
            created_at=now,
            expires_at=now + timedelta(seconds=60),
        )

        assert request.tool_name == "send_message"
        assert request.status is ConfirmationStatus.PENDING
        assert request.request_id is not None

    def test_arguments_are_immutable(self) -> None:
        request = ConfirmationRequest(
            tool_name="send_message",
            call_id=uuid4(),
            arguments={"recipient": "user@example.com"},
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
        )

        with pytest.raises(TypeError):
            request.arguments["recipient"] = "attacker@example.com"

    def test_rejects_invalid_tool_name(self) -> None:
        with pytest.raises(ValueError):
            ConfirmationRequest(
                tool_name=" ",
                call_id=uuid4(),
                arguments={},
                expires_at=datetime.now(timezone.utc) + timedelta(seconds=60),
            )

    def test_rejects_naive_expiration_datetime(self) -> None:
        with pytest.raises(ValueError):
            ConfirmationRequest(
                tool_name="send_message",
                call_id=uuid4(),
                arguments={},
                expires_at=datetime.now() + timedelta(seconds=60),
            )

    def test_rejects_expiration_before_creation(self) -> None:
        now = datetime.now(timezone.utc)

        with pytest.raises(ValueError):
            ConfirmationRequest(
                tool_name="send_message",
                call_id=uuid4(),
                arguments={},
                created_at=now,
                expires_at=now - timedelta(seconds=1),
            )


class TestConfirmationDecision:
    def test_approved_decision(self) -> None:
        decision = ConfirmationDecision(
            request_id=uuid4(),
            status=ConfirmationStatus.APPROVED,
            decided_by="user",
        )

        assert decision.approved is True

    def test_rejected_decision(self) -> None:
        decision = ConfirmationDecision(
            request_id=uuid4(),
            status=ConfirmationStatus.REJECTED,
            decided_by="user",
        )

        assert decision.approved is False

    def test_rejects_string_status(self) -> None:
        with pytest.raises(TypeError):
            ConfirmationDecision(
                request_id=uuid4(),
                status="approved",
                decided_by="user",
            )

    def test_rejects_pending_as_decision(self) -> None:
        with pytest.raises(ValueError):
            ConfirmationDecision(
                request_id=uuid4(),
                status=ConfirmationStatus.PENDING,
                decided_by="user",
            )


class TestConfirmationManager:
    @pytest.mark.asyncio
    async def test_create_request(self) -> None:
        manager = ConfirmationManager()
        call = make_tool_call()

        request = manager.create_request(call, timeout_seconds=60)

        assert request.call_id == call.call_id
        assert request.tool_name == call.tool_name
        assert dict(request.arguments) == dict(call.arguments)
        assert request.status is ConfirmationStatus.PENDING
        assert manager.pending_count == 1

    @pytest.mark.asyncio
    async def test_get_request(self) -> None:
        manager = ConfirmationManager()
        request = manager.create_request(make_tool_call())

        assert manager.get_request(request.request_id) == request

    @pytest.mark.asyncio
    async def test_list_pending(self) -> None:
        manager = ConfirmationManager()

        first = manager.create_request(make_tool_call())
        second = manager.create_request(make_tool_call())

        pending = manager.list_pending()

        assert len(pending) == 2
        assert first in pending
        assert second in pending

    @pytest.mark.asyncio
    async def test_approval_resolves_async_waiter(self) -> None:
        manager = ConfirmationManager()
        request = manager.create_request(make_tool_call())

        waiter = asyncio.create_task(
            manager.wait_for_decision(request.request_id)
        )

        decision = manager.resolve(
            request.request_id,
            approved=True,
            decided_by="user",
        )

        result = await waiter

        assert decision.status is ConfirmationStatus.APPROVED
        assert result == decision
        assert result.approved is True
        assert manager.pending_count == 0

    @pytest.mark.asyncio
    async def test_rejection_resolves_async_waiter(self) -> None:
        manager = ConfirmationManager()
        request = manager.create_request(make_tool_call())

        waiter = asyncio.create_task(
            manager.wait_for_decision(request.request_id)
        )

        decision = manager.resolve(
            request.request_id,
            approved=False,
            decided_by="user",
        )

        result = await waiter

        assert decision.status is ConfirmationStatus.REJECTED
        assert result == decision
        assert result.approved is False

    @pytest.mark.asyncio
    async def test_request_expires(self) -> None:
        manager = ConfirmationManager()

        request = manager.create_request(
            make_tool_call(),
            timeout_seconds=0.01,
        )

        decision = await manager.wait_for_decision(request.request_id)

        assert decision.status is ConfirmationStatus.EXPIRED
        assert manager.get_request(
            request.request_id
        ).status is ConfirmationStatus.EXPIRED
        assert manager.pending_count == 0

    @pytest.mark.asyncio
    async def test_expired_request_cannot_be_approved(self) -> None:
        manager = ConfirmationManager()

        request = manager.create_request(
            make_tool_call(),
            timeout_seconds=0.01,
        )

        await manager.wait_for_decision(request.request_id)

        with pytest.raises(ValueError):
            manager.resolve(request.request_id, approved=True)

    @pytest.mark.asyncio
    async def test_duplicate_resolution_is_rejected(self) -> None:
        manager = ConfirmationManager()
        request = manager.create_request(make_tool_call())

        manager.resolve(request.request_id, approved=True)

        with pytest.raises(ValueError):
            manager.resolve(request.request_id, approved=False)

    @pytest.mark.asyncio
    async def test_approval_can_only_be_consumed_once(self) -> None:
        manager = ConfirmationManager()
        call = make_tool_call()
        request = manager.create_request(call)

        manager.resolve(request.request_id, approved=True)

        assert manager.consume_approval(request.request_id, call) is True
        assert manager.consume_approval(request.request_id, call) is False

        assert manager.get_request(
            request.request_id
        ).status is ConfirmationStatus.CONSUMED

    @pytest.mark.asyncio
    async def test_approval_is_bound_to_call_id(self) -> None:
        manager = ConfirmationManager()
        original_call = make_tool_call()
        request = manager.create_request(original_call)

        manager.resolve(request.request_id, approved=True)

        altered_call = ToolCall(
            tool_name=original_call.tool_name,
            arguments=original_call.arguments,
            call_id=uuid4(),
        )

        assert manager.consume_approval(
            request.request_id,
            altered_call,
        ) is False

        assert manager.consume_approval(
            request.request_id,
            original_call,
        ) is True

    @pytest.mark.asyncio
    async def test_approval_is_bound_to_tool_name(self) -> None:
        manager = ConfirmationManager()
        original_call = make_tool_call(tool_name="send_message")
        request = manager.create_request(original_call)

        manager.resolve(request.request_id, approved=True)

        altered_call = ToolCall(
            tool_name="delete_file",
            arguments=original_call.arguments,
            call_id=original_call.call_id,
        )

        assert manager.consume_approval(
            request.request_id,
            altered_call,
        ) is False

    @pytest.mark.asyncio
    async def test_approval_is_bound_to_arguments(self) -> None:
        manager = ConfirmationManager()
        original_call = make_tool_call(
            arguments={"recipient": "safe@example.com"}
        )
        request = manager.create_request(original_call)

        manager.resolve(request.request_id, approved=True)

        altered_call = ToolCall(
            tool_name=original_call.tool_name,
            arguments={"recipient": "attacker@example.com"},
            call_id=original_call.call_id,
        )

        assert manager.consume_approval(
            request.request_id,
            altered_call,
        ) is False

    @pytest.mark.asyncio
    async def test_rejected_approval_cannot_be_consumed(self) -> None:
        manager = ConfirmationManager()
        call = make_tool_call()
        request = manager.create_request(call)

        manager.resolve(request.request_id, approved=False)

        assert manager.consume_approval(request.request_id, call) is False

    @pytest.mark.asyncio
    async def test_unknown_request_raises_key_error(self) -> None:
        manager = ConfirmationManager()

        with pytest.raises(KeyError):
            manager.get_request(uuid4())

    @pytest.mark.asyncio
    async def test_invalid_timeout_is_rejected(self) -> None:
        manager = ConfirmationManager()

        with pytest.raises(ValueError):
            manager.create_request(
                make_tool_call(),
                timeout_seconds=0,
            )

    @pytest.mark.asyncio
    async def test_invalid_approval_type_is_rejected(self) -> None:
        manager = ConfirmationManager()
        request = manager.create_request(make_tool_call())

        with pytest.raises(TypeError):
            manager.resolve(request.request_id, approved=1)
