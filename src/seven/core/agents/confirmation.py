
"""Domain models for user confirmation workflows."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4


class ConfirmationStatus(StrEnum):
    """Represent the lifecycle of a confirmation request."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfirmationRequest:
    """Represent a request for user approval of a specific tool call."""

    tool_name: str
    call_id: UUID
    arguments: Mapping[str, Any]
    expires_at: datetime
    task_id: UUID | None = None
    request_id: UUID = field(default_factory=uuid4)
    status: ConfirmationStatus = ConfirmationStatus.PENDING
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name.strip():
            raise ValueError("tool_name cannot be empty.")

        if not isinstance(self.call_id, UUID):
            raise TypeError("call_id must be a UUID.")

        if not isinstance(self.request_id, UUID):
            raise TypeError("request_id must be a UUID.")

        if self.task_id is not None and not isinstance(self.task_id, UUID):
            raise TypeError("task_id must be a UUID or None.")

        if not isinstance(self.status, ConfirmationStatus):
            raise TypeError("status must be a ConfirmationStatus instance.")

        if not isinstance(self.arguments, Mapping):
            raise TypeError("arguments must be a mapping.")

        if not all(isinstance(key, str) for key in self.arguments):
            raise TypeError("argument keys must be strings.")

        object.__setattr__(
            self,
            "arguments",
            MappingProxyType(dict(self.arguments)),
        )

        for field_name in ("created_at", "expires_at"):
            value = getattr(self, field_name)

            if not isinstance(value, datetime):
                raise TypeError(f"{field_name} must be a datetime.")

            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field_name} must be timezone-aware.")

        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be later than created_at.")


@dataclass(frozen=True, slots=True, kw_only=True)
class ConfirmationDecision:
    """Represent the user's decision for a confirmation request."""

    request_id: UUID
    status: ConfirmationStatus
    decided_by: str
    decided_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, UUID):
            raise TypeError("request_id must be a UUID.")

        if not isinstance(self.status, ConfirmationStatus):
            raise TypeError("status must be a ConfirmationStatus instance.")

        if self.status not in (
            ConfirmationStatus.APPROVED,
            ConfirmationStatus.REJECTED,
            ConfirmationStatus.EXPIRED,
        ):
            raise ValueError("Invalid confirmation decision status.")

        if not isinstance(self.decided_by, str) or not self.decided_by.strip():
            raise ValueError("decided_by cannot be empty.")

        if not isinstance(self.decided_at, datetime):
            raise TypeError("decided_at must be a datetime.")

        if (
            self.decided_at.tzinfo is None
            or self.decided_at.utcoffset() is None
        ):
            raise ValueError("decided_at must be timezone-aware.")

        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("reason must be a string or None.")

    @property
    def approved(self) -> bool:
        """Return whether the request was approved."""
        return self.status is ConfirmationStatus.APPROVED
