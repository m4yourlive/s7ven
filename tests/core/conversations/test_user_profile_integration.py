
"""Integration tests for the structured user profile system."""

import pytest_asyncio
import pytest
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
)

from seven.core.conversations.profile_context_builder import (
    ProfileContextBuilder,
)
from seven.core.conversations.profile_memory_context_composer import (
    ProfileMemoryContextComposer,
)
from seven.core.conversations.user_profile_contracts import (
    ProfileUpdate,
)
from seven.core.conversations.user_profile_extractor import (
    UserProfileExtractor,
)
from seven.core.conversations.user_profile_service import (
    UserProfileService,
)
from seven.core.database.base import Base
from seven.core.database.repositories.user_profile_repository import (
    UserProfileRepository,
)
import seven.core.database.models  # noqa: F401


@pytest_asyncio.fixture
async def session():
    """Create an isolated asynchronous database session."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        engine,
        expire_on_commit=False,
    )

    async with session_factory() as database_session:
        yield database_session

    await engine.dispose()


@pytest.fixture
def profile_service(session):
    """Create a profile service backed by a real repository."""
    repository = UserProfileRepository(session)
    extractor = UserProfileExtractor()

    return UserProfileService(
        repository=repository,
        extractor=extractor,
    )


@pytest.mark.asyncio
async def test_extract_persist_and_retrieve_profile(
    profile_service,
    session,
):
    """Verify extracted preferences survive database persistence."""
    extracted = await profile_service.extract_and_save(
        "I prefer concise responses."
    )

    assert extracted.preferences

    await session.commit()

    retrieved = await profile_service.get_profile()

    assert retrieved.preferences == extracted.preferences
    assert any(
        "concise" in str(preference.value).casefold()
        for preference in retrieved.preferences
    )


@pytest.mark.asyncio
async def test_explicit_profile_update_is_persisted(
    profile_service,
    session,
):
    """Verify explicit profile updates are stored and retrievable."""
    await profile_service.apply_profile_update(
        ProfileUpdate(
            key="occupation",
            value="software developer",
            confidence=1.0,
            source="explicit",
        )
    )

    await session.commit()

    profile = await profile_service.get_profile()

    occupation = next(
        (
            attribute
            for attribute in profile.attributes
            if attribute.key == "occupation"
        ),
        None,
    )

    assert occupation is not None
    assert occupation.value == "software developer"
    assert occupation.confidence == 1.0


@pytest.mark.asyncio
async def test_profile_context_can_be_composed_with_memory(
    profile_service,
    session,
):
    """Verify persisted profile data can enrich memory context."""
    await profile_service.apply_profile_update(
        ProfileUpdate(
            key="occupation",
            value="software developer",
        )
    )

    await profile_service.extract_and_save(
        "I prefer concise responses."
    )

    await session.commit()

    profile = await profile_service.get_profile()

    builder = ProfileContextBuilder()
    composer = ProfileMemoryContextComposer(
        profile_builder=builder,
        max_context_chars=2000,
    )

    result = composer.compose(
        "Relevant memory: The user is building S7VEN.",
        profile,
    )

    assert "Relevant memory:" in result
    assert "User profile:" in result
    assert "software developer" in result
    assert "User preferences:" in result
    assert "concise" in result


@pytest.mark.asyncio
async def test_repeated_extraction_updates_existing_preference(
    profile_service,
    session,
):
    """Verify repeated extraction does not create duplicate keys."""
    await profile_service.extract_and_save(
        "I prefer concise responses."
    )

    await session.commit()

    first_profile = await profile_service.get_profile()
    first_count = len(first_profile.preferences)

    await profile_service.extract_and_save(
        "I prefer concise responses."
    )

    await session.commit()

    second_profile = await profile_service.get_profile()

    assert len(second_profile.preferences) == first_count

    preference_keys = [
        preference.key
        for preference in second_profile.preferences
    ]

    assert len(preference_keys) == len(set(preference_keys))
