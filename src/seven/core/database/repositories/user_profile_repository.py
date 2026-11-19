
"""User profile repository."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from seven.core.conversations.user_profile_contracts import (
    ProfileUpdate,
    PreferenceUpdate,
    UserPreference,
    UserProfile,
    UserProfileAttribute,
)
from seven.core.database.models import (
    UserPreferenceRecord,
    UserProfileAttributeRecord,
)
from seven.core.database.repositories.base_repository import BaseRepository


class UserProfileRepository:
    """Manage persistent user profile attributes and preferences."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize profile repositories."""
        if not isinstance(session, AsyncSession):
            raise TypeError("Session must be an AsyncSession.")

        self._attributes = BaseRepository(
            session,
            UserProfileAttributeRecord,
        )
        self._preferences = BaseRepository(
            session,
            UserPreferenceRecord,
        )
        self._session = session

    @property
    def session(self) -> AsyncSession:
        """Return the managed database session."""
        return self._session

    async def save_attribute(
        self,
        attribute: UserProfileAttribute,
    ) -> UserProfileAttributeRecord:
        """Create or update one profile attribute."""
        if not isinstance(attribute, UserProfileAttribute):
            raise TypeError(
                "Attribute must be a UserProfileAttribute."
            )

        statement = select(UserProfileAttributeRecord).where(
            UserProfileAttributeRecord.key == attribute.key
        )
        result = await self._session.execute(statement)
        existing = result.scalar_one_or_none()

        values = {
            "value": attribute.value,
            "confidence": attribute.confidence,
            "source": attribute.source,
        }

        if existing is not None:
            return await self._attributes.update(existing, values)

        record = UserProfileAttributeRecord(
            key=attribute.key,
            **values,
        )

        return await self._attributes.add(record)

    async def save_preference(
        self,
        preference: UserPreference,
    ) -> UserPreferenceRecord:
        """Create or update one user preference."""
        if not isinstance(preference, UserPreference):
            raise TypeError(
                "Preference must be a UserPreference."
            )

        statement = select(UserPreferenceRecord).where(
            UserPreferenceRecord.key == preference.key
        )
        result = await self._session.execute(statement)
        existing = result.scalar_one_or_none()

        values = {
            "value": preference.value,
            "confidence": preference.confidence,
            "source": preference.source,
        }

        if existing is not None:
            return await self._preferences.update(existing, values)

        record = UserPreferenceRecord(
            key=preference.key,
            **values,
        )

        return await self._preferences.add(record)

    async def get_attribute(
        self,
        key: str,
    ) -> UserProfileAttribute | None:
        """Retrieve one profile attribute by key."""
        self._validate_key(key)

        statement = select(UserProfileAttributeRecord).where(
            UserProfileAttributeRecord.key == key.strip().casefold()
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return self._to_attribute(record)

    async def get_preference(
        self,
        key: str,
    ) -> UserPreference | None:
        """Retrieve one preference by key."""
        self._validate_key(key)

        statement = select(UserPreferenceRecord).where(
            UserPreferenceRecord.key == key.strip().casefold()
        )
        result = await self._session.execute(statement)
        record = result.scalar_one_or_none()

        if record is None:
            return None

        return self._to_preference(record)

    async def get_all_attributes(
        self,
    ) -> tuple[UserProfileAttribute, ...]:
        """Retrieve all profile attributes."""
        statement = select(UserProfileAttributeRecord).order_by(
            UserProfileAttributeRecord.key.asc()
        )
        result = await self._session.execute(statement)

        return tuple(
            self._to_attribute(record)
            for record in result.scalars().all()
        )

    async def get_all_preferences(
        self,
    ) -> tuple[UserPreference, ...]:
        """Retrieve all user preferences."""
        statement = select(UserPreferenceRecord).order_by(
            UserPreferenceRecord.key.asc()
        )
        result = await self._session.execute(statement)

        return tuple(
            self._to_preference(record)
            for record in result.scalars().all()
        )

    async def get_profile(self) -> UserProfile:
        """Retrieve the complete structured user profile."""
        attributes = await self.get_all_attributes()
        preferences = await self.get_all_preferences()

        return UserProfile(
            attributes=attributes,
            preferences=preferences,
        )

    async def save_profile(
        self,
        profile: UserProfile,
    ) -> UserProfile:
        """Create or update every entry in a user profile."""
        if not isinstance(profile, UserProfile):
            raise TypeError("Profile must be a UserProfile.")

        for attribute in profile.attributes:
            await self.save_attribute(attribute)

        for preference in profile.preferences:
            await self.save_preference(preference)

        return await self.get_profile()

    async def apply_profile_update(
        self,
        update: ProfileUpdate,
    ) -> UserProfileAttributeRecord:
        """Apply one structured profile update."""
        if not isinstance(update, ProfileUpdate):
            raise TypeError("Update must be a ProfileUpdate.")

        return await self.save_attribute(update.to_attribute())

    async def apply_preference_update(
        self,
        update: PreferenceUpdate,
    ) -> UserPreferenceRecord:
        """Apply one structured preference update."""
        if not isinstance(update, PreferenceUpdate):
            raise TypeError(
                "Update must be a PreferenceUpdate."
            )

        return await self.save_preference(update.to_preference())

    async def delete_attribute(self, key: str) -> bool:
        """Delete a profile attribute by key."""
        self._validate_key(key)

        record = await self._get_attribute_record(key)

        if record is None:
            return False

        await self._attributes.delete(record)

        return True

    async def delete_preference(self, key: str) -> bool:
        """Delete a preference by key."""
        self._validate_key(key)

        record = await self._get_preference_record(key)

        if record is None:
            return False

        await self._preferences.delete(record)

        return True

    async def delete_profile(self) -> None:
        """Delete all profile attributes and preferences."""
        attributes = await self._session.execute(
            select(UserProfileAttributeRecord)
        )
        preferences = await self._session.execute(
            select(UserPreferenceRecord)
        )

        for record in attributes.scalars().all():
            await self._attributes.delete(record)

        for record in preferences.scalars().all():
            await self._preferences.delete(record)

    async def _get_attribute_record(
        self,
        key: str,
    ) -> UserProfileAttributeRecord | None:
        statement = select(UserProfileAttributeRecord).where(
            UserProfileAttributeRecord.key == key.strip().casefold()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none()

    async def _get_preference_record(
        self,
        key: str,
    ) -> UserPreferenceRecord | None:
        statement = select(UserPreferenceRecord).where(
            UserPreferenceRecord.key == key.strip().casefold()
        )
        result = await self._session.execute(statement)

        return result.scalar_one_or_none()

    @staticmethod
    def _validate_key(key: str) -> None:
        """Validate a profile entry key."""
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")

        if not key.strip():
            raise ValueError("Key cannot be empty.")

    @staticmethod
    def _to_attribute(
        record: UserProfileAttributeRecord,
    ) -> UserProfileAttribute:
        """Convert a database record into a profile contract."""
        return UserProfileAttribute(
            key=record.key,
            value=record.value,
            confidence=record.confidence,
            source=record.source,
            updated_at=record.updated_at,
        )

    @staticmethod
    def _to_preference(
        record: UserPreferenceRecord,
    ) -> UserPreference:
        """Convert a database record into a preference contract."""
        return UserPreference(
            key=record.key,
            value=record.value,
            confidence=record.confidence,
            source=record.source,
            updated_at=record.updated_at,
        )
