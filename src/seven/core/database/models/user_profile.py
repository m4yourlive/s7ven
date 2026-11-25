
"""User profile database models."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import JSON, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from seven.core.database.base import Base, TimestampMixin


class UserProfileAttributeRecord(Base, TimestampMixin):
    """Represent a persistent user profile attribute."""

    __tablename__ = "user_profile_attributes"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )

    value: Mapped[str | bool | int | float] = mapped_column(
        JSON,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
    )

    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="explicit",
    )


class UserPreferenceRecord(Base, TimestampMixin):
    """Represent a persistent user preference."""

    __tablename__ = "user_preferences"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    key: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )

    value: Mapped[str | bool | int | float] = mapped_column(
        JSON,
        nullable=False,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=1.0,
    )

    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="explicit",
    )
