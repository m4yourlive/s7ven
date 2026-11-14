"""Tests for SQLAlchemy declarative base and shared mixins."""

from datetime import datetime

from sqlalchemy import MetaData, PrimaryKeyConstraint, String
from sqlalchemy.orm import Mapped, mapped_column

from seven.core.database import Base, TimestampMixin


class SampleModel(TimestampMixin, Base):
    """Test model for validating shared database metadata."""

    __tablename__ = "test_database_base_sample"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)


def test_base_uses_sqlalchemy_metadata() -> None:
    assert isinstance(Base.metadata, MetaData)


def test_base_has_naming_convention() -> None:
    convention = Base.metadata.naming_convention

    assert convention["ix"] == "ix_%(column_0_label)s"
    assert convention["uq"] == "uq_%(table_name)s_%(column_0_name)s"
    assert convention["pk"] == "pk_%(table_name)s"


def test_sample_model_uses_shared_metadata() -> None:
    assert SampleModel.metadata is Base.metadata


def test_sample_model_has_expected_columns() -> None:
    columns = SampleModel.__table__.columns

    assert "id" in columns
    assert "name" in columns
    assert "created_at" in columns
    assert "updated_at" in columns


def test_timestamp_columns_are_not_nullable() -> None:
    table = SampleModel.__table__

    assert table.columns.created_at.nullable is False
    assert table.columns.updated_at.nullable is False


def test_timestamp_columns_have_server_defaults() -> None:
    table = SampleModel.__table__

    assert table.columns.created_at.server_default is not None
    assert table.columns.updated_at.server_default is not None


def test_updated_at_has_onupdate_expression() -> None:
    column = SampleModel.__table__.columns.updated_at

    assert column.onupdate is not None


def test_created_at_uses_datetime_type() -> None:
    column = SampleModel.__table__.columns.created_at

    assert column.type.python_type is datetime


def test_primary_key_uses_naming_convention() -> None:
    table = SampleModel.__table__

    primary_key = next(
        constraint
        for constraint in table.constraints
        if isinstance(constraint, PrimaryKeyConstraint)
    )

    assert primary_key.name == "pk_test_database_base_sample"
