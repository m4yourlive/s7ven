
"""Add user profile attribute and preference tables."""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c2a7e91f4b30"
down_revision: Union[str, None] = "f9c7d82df6b4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create user profile tables."""
    op.create_table(
        "user_profile_attributes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_user_profile_attributes"),
        ),
    )

    op.create_index(
        "ix_user_profile_attributes_key",
        "user_profile_attributes",
        ["key"],
        unique=True,
    )

    op.create_table(
        "user_preferences",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint(
            "id",
            name=op.f("pk_user_preferences"),
        ),
    )

    op.create_index(
        "ix_user_preferences_key",
        "user_preferences",
        ["key"],
        unique=True,
    )


def downgrade() -> None:
    """Remove user profile tables."""
    op.drop_index(
        "ix_user_preferences_key",
        table_name="user_preferences",
    )
    op.drop_table("user_preferences")

    op.drop_index(
        "ix_user_profile_attributes_key",
        table_name="user_profile_attributes",
    )
    op.drop_table("user_profile_attributes")
