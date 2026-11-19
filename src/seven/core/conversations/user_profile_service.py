
"""User profile service."""

from __future__ import annotations

from seven.core.conversations.user_profile_contracts import (
    PreferenceUpdate,
    ProfileUpdate,
    UserProfile,
)
from seven.core.conversations.user_profile_extractor import (
    UserProfileExtractor,
)
from seven.core.database.repositories.user_profile_repository import (
    UserProfileRepository,
)


class UserProfileService:
    """Coordinate profile extraction, persistence, and retrieval."""

    def __init__(
        self,
        repository: UserProfileRepository,
        extractor: UserProfileExtractor | None = None,
    ) -> None:
        """Initialize the user profile service."""
        if not isinstance(repository, UserProfileRepository):
            raise TypeError(
                "Repository must be a UserProfileRepository."
            )

        if extractor is not None and not isinstance(
            extractor,
            UserProfileExtractor,
        ):
            raise TypeError(
                "Extractor must be a UserProfileExtractor."
            )

        self._repository = repository
        self._extractor = extractor or UserProfileExtractor()

    @property
    def repository(self) -> UserProfileRepository:
        """Return the managed profile repository."""
        return self._repository

    @property
    def extractor(self) -> UserProfileExtractor:
        """Return the configured profile extractor."""
        return self._extractor

    def extract_from_text(self, text: str) -> UserProfile:
        """Extract structured profile information from text."""
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        return self._extractor.extract(text)

    async def extract_and_save(self, text: str) -> UserProfile:
        """Extract profile information and persist discovered entries."""
        extracted = self.extract_from_text(text)

        if not extracted.attributes and not extracted.preferences:
            return await self._repository.get_profile()

        return await self._repository.save_profile(extracted)

    async def get_profile(self) -> UserProfile:
        """Retrieve the current persisted user profile."""
        return await self._repository.get_profile()

    async def apply_profile_update(
        self,
        update: ProfileUpdate,
    ):
        """Apply and persist one explicit profile update."""
        if not isinstance(update, ProfileUpdate):
            raise TypeError("Update must be a ProfileUpdate.")

        return await self._repository.apply_profile_update(update)

    async def apply_preference_update(
        self,
        update: PreferenceUpdate,
    ):
        """Apply and persist one explicit preference update."""
        if not isinstance(update, PreferenceUpdate):
            raise TypeError("Update must be a PreferenceUpdate.")

        return await self._repository.apply_preference_update(update)
