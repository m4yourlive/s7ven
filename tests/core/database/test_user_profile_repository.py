
"""Tests for persistent user profile storage."""

import pytest
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
    UserProfileAttribute,
)
from seven.core.database import Base
from seven.core.database.repositories import UserProfileRepository


@pytest.mark.asyncio
async def test_save_and_retrieve_attribute():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            repository = UserProfileRepository(session)

            await repository.save_attribute(
                UserProfileAttribute(
                    key="occupation",
                    value="Software Developer",
                    confidence=0.95,
                )
            )

            result = await repository.get_attribute("occupation")

            assert result is not None
            assert result.value == "Software Developer"
            assert result.confidence == 0.95

    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_save_attribute_updates_existing_record():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            repository = UserProfileRepository(session)

            await repository.save_attribute(
                UserProfileAttribute(
                    key="occupation",
                    value="Student",
                )
            )

            await repository.save_attribute(
                UserProfileAttribute(
                    key="occupation",
                    value="Software Developer",
                    confidence=0.95,
                    source="explicit",
                )
            )

            attributes = await repository.get_all_attributes()

            assert len(attributes) == 1
            assert attributes[0].value == "Software Developer"

    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_save_and_retrieve_preference():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            repository = UserProfileRepository(session)

            await repository.save_preference(
                UserPreference(
                    key="preference_design",
                    value="prefers minimal design",
                )
            )

            result = await repository.get_preference(
                "preference_design"
            )

            assert result is not None
            assert result.value == "prefers minimal design"

    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_save_and_retrieve_complete_profile():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            repository = UserProfileRepository(session)

            profile = UserProfile(
                attributes=(
                    UserProfileAttribute(
                        key="occupation",
                        value="Student",
                    ),
                    UserProfileAttribute(
                        key="location",
                        value="Indonesia",
                    ),
                ),
                preferences=(
                    UserPreference(
                        key="preference_design",
                        value="prefers minimal design",
                    ),
                ),
            )

            saved = await repository.save_profile(profile)
            retrieved = await repository.get_profile()

            assert len(saved.attributes) == 2
            assert len(saved.preferences) == 1

            assert retrieved == saved

    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_attribute():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            repository = UserProfileRepository(session)

            await repository.save_attribute(
                UserProfileAttribute(
                    key="occupation",
                    value="Student",
                )
            )

            deleted = await repository.delete_attribute("occupation")
            result = await repository.get_attribute("occupation")

            assert deleted is True
            assert result is None

    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_delete_missing_preference_returns_false():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            repository = UserProfileRepository(session)

            result = await repository.delete_preference(
                "preference_unknown"
            )

            assert result is False

    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_profile_storage_preserves_scalar_types():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            repository = UserProfileRepository(session)

            await repository.save_profile(
                UserProfile(
                    attributes=(
                        UserProfileAttribute(
                            key="age",
                            value=16,
                        ),
                        UserProfileAttribute(
                            key="is_student",
                            value=True,
                        ),
                        UserProfileAttribute(
                            key="score",
                            value=98.5,
                        ),
                    ),
                )
            )

            profile = await repository.get_profile()

            values = {
                item.key: item.value
                for item in profile.attributes
            }

            assert values["age"] == 16
            assert values["is_student"] is True
            assert values["score"] == 98.5

    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_repository_rejects_invalid_attribute():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    try:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

        session_factory = async_sessionmaker(
            engine,
            expire_on_commit=False,
        )

        async with session_factory() as session:
            repository = UserProfileRepository(session)

            with pytest.raises(TypeError):
                await repository.save_attribute("invalid")

    finally:
        await engine.dispose()
