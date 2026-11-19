
"""Compose structured profile data with retrieved memory context."""

from __future__ import annotations

from seven.core.conversations.profile_context_builder import (
    ProfileContextBuilder,
)
from seven.core.conversations.user_profile_contracts import (
    UserProfile,
)


class ProfileMemoryContextComposer:
    """Combine retrieved memories and structured profile context."""

    def __init__(
        self,
        profile_builder: ProfileContextBuilder | None = None,
        *,
        max_context_chars: int = 4000,
    ) -> None:
        """Initialize the context composer."""
        if profile_builder is not None and not isinstance(
            profile_builder,
            ProfileContextBuilder,
        ):
            raise TypeError(
                "Profile builder must be a ProfileContextBuilder."
            )

        if isinstance(max_context_chars, bool) or not isinstance(
            max_context_chars,
            int,
        ):
            raise TypeError(
                "Maximum context characters must be an integer."
            )

        if max_context_chars < 1:
            raise ValueError(
                "Maximum context characters must be at least 1."
            )

        self._profile_builder = (
            profile_builder or ProfileContextBuilder()
        )
        self._max_context_chars = max_context_chars

    @property
    def profile_builder(self) -> ProfileContextBuilder:
        """Return the configured profile context builder."""
        return self._profile_builder

    @property
    def max_context_chars(self) -> int:
        """Return the maximum combined context length."""
        return self._max_context_chars

    def compose(
        self,
        memory_context: str,
        profile: UserProfile | None = None,
    ) -> str:
        """Combine memory and profile context within the size limit."""
        if not isinstance(memory_context, str):
            raise TypeError("Memory context must be a string.")

        if profile is not None and not isinstance(profile, UserProfile):
            raise TypeError("Profile must be a UserProfile or None.")

        normalized_memory = memory_context.strip()
        profile_context = (
            self._profile_builder.build(profile)
            if profile is not None
            else ""
        )

        if not normalized_memory:
            return self._truncate(profile_context)

        if not profile_context:
            return self._truncate(normalized_memory)

        separator = "\n\n"
        available_profile_chars = (
            self._max_context_chars
            - len(normalized_memory)
            - len(separator)
        )

        if available_profile_chars <= 0:
            return self._truncate(normalized_memory)

        limited_profile = self._truncate(
            profile_context,
            limit=available_profile_chars,
        )

        if not limited_profile:
            return self._truncate(normalized_memory)

        return self._truncate(
            f"{normalized_memory}{separator}{limited_profile}"
        )

    def _truncate(
        self,
        value: str,
        *,
        limit: int | None = None,
    ) -> str:
        """Truncate text without exceeding the configured limit."""
        if not value:
            return ""

        effective_limit = (
            self._max_context_chars
            if limit is None
            else limit
        )

        if effective_limit <= 0:
            return ""

        if len(value) <= effective_limit:
            return value

        if effective_limit <= 3:
            return value[:effective_limit]

        return (
            value[: effective_limit - 3].rstrip()
            + "..."
        )
