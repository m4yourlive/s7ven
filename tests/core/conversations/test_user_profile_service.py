
"""Tests for the user profile service."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from seven.core.conversations.user_profile_contracts import (
    PreferenceUpdate,
    ProfileUpdate,
    UserPreference,
    UserProfile,
    UserProfileAttribute,
)
from seven.core.conversations.user_profile_extractor import (
    UserProfileExtractor,
)
from seven.core.conversations.user_profile_service import (
    UserProfileService,
)
from seven.core.database.repositories.user_profile_repository import (
    UserProfileRepository,
)


@pytest.fixture
def repository():
    return MagicMock(spec=UserProfileRepository)


@pytest.fixture
def extractor():
    return MagicMock(spec=UserProfileExtractor)


@pytest.fixture
def service(repository, extractor):
    return UserProfileService(
        repository=repository,
        extractor=extractor,
    )


def test_initialize_with_default_extractor(repository):
    service = UserProfileService(repository)

    assert service.repository is repository
    assert isinstance(service.extractor, UserProfileExtractor)


def test_initialize_rejects_invalid_repository():
    with pytest.raises(TypeError):
        UserProfileService(repository=object())


def test_initialize_rejects_invalid_extractor(repository):
    with pytest.raises(TypeError):
        UserProfileService(
            repository=repository,
            extractor=object(),
        )


def test_extract_from_text(service, extractor):
    expected = UserProfile(
        attributes=(
            UserProfileAttribute(
                key="occupation",
                value="developer",
            ),
        ),
    )
    extractor.extract.return_value = expected

    result = service.extract_from_text(
        "I am a developer."
    )

    assert result == expected
    extractor.extract.assert_called_once_with(
        "I am a developer."
    )


def test_extract_from_text_rejects_non_string(service):
    with pytest.raises(TypeError):
        service.extract_from_text(123)


@pytest.mark.asyncio
async def test_extract_and_save_persists_profile(
    service,
    extractor,
    repository,
):
    extracted = UserProfile(
        attributes=(
            UserProfileAttribute(
                key="occupation",
                value="developer",
            ),
        ),
    )
    extractor.extract.return_value = extracted
    repository.save_profile = AsyncMock(
        return_value=extracted
    )

    result = await service.extract_and_save(
        "I am a developer."
    )

    assert result == extracted
    repository.save_profile.assert_awaited_once_with(
        extracted
    )


@pytest.mark.asyncio
async def test_extract_and_save_skips_empty_profile(
    service,
    extractor,
    repository,
):
    existing = UserProfile(
        preferences=(
            UserPreference(
                key="communication",
                value="concise",
            ),
        ),
    )
    extractor.extract.return_value = UserProfile()
    repository.get_profile = AsyncMock(
        return_value=existing
    )
    repository.save_profile = AsyncMock()

    result = await service.extract_and_save(
        "Hello there."
    )

    assert result == existing
    repository.get_profile.assert_awaited_once()
    repository.save_profile.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_profile_delegates_to_repository(
    service,
    repository,
):
    expected = UserProfile()
    repository.get_profile = AsyncMock(
        return_value=expected
    )

    result = await service.get_profile()

    assert result == expected
    repository.get_profile.assert_awaited_once()


@pytest.mark.asyncio
async def test_apply_profile_update(
    service,
    repository,
):
    update = ProfileUpdate(
        key="occupation",
        value="developer",
    )
    expected = object()
    repository.apply_profile_update = AsyncMock(
        return_value=expected
    )

    result = await service.apply_profile_update(update)

    assert result is expected
    repository.apply_profile_update.assert_awaited_once_with(
        update
    )


@pytest.mark.asyncio
async def test_apply_preference_update(
    service,
    repository,
):
    update = PreferenceUpdate(
        key="communication",
        value="concise",
    )
    expected = object()
    repository.apply_preference_update = AsyncMock(
        return_value=expected
    )

    result = await service.apply_preference_update(update)

    assert result is expected
    repository.apply_preference_update.assert_awaited_once_with(
        update
    )


@pytest.mark.asyncio
async def test_apply_profile_update_rejects_invalid_input(service):
    with pytest.raises(TypeError):
        await service.apply_profile_update("invalid")


@pytest.mark.asyncio
async def test_apply_preference_update_rejects_invalid_input(
    service,
):
    with pytest.raises(TypeError):
        await service.apply_preference_update("invalid")
