from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4

@dataclass(frozen=True, slots=True, kw_only=True)
class Event:
    """Base event class for Seven.
    every event has an unique ID and UTC Timestamp."""

event_id: UUID = field(default_factory=uuid4)

occured_at: datetime = field(
       default_factory=lambda: datetime.now(timezone.utc)
       )