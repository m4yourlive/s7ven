
"""Build compact prompt context from structured user profiles."""

from __future__ import annotations

from seven.core.conversations.user_profile_contracts import (
    UserProfile,
)


class ProfileContextBuilder:
    """Convert structured profile data into prompt-ready context."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.0,
        max_attributes: int = 20,
        max_preferences: int = 20,
        max_context_chars: int = 2000,
    ) -> None:
        """Initialize profile context formatting options."""
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
        ):
            raise TypeError("Minimum confidence must be numeric.")

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                "Minimum confidence must be between 0.0 and 1.0."
            )

        for name, value in (
            ("Maximum attributes", max_attributes),
            ("Maximum preferences", max_preferences),
            ("Maximum context characters", max_context_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

            if value < 1:
                raise ValueError(f"{name} must be at least 1.")

        self._min_confidence = float(min_confidence)
        self._max_attributes = max_attributes
        self._max_preferences = max_preferences
        self._max_context_chars = max_context_chars

    def build(self, profile: UserProfile) -> str:
        """Build a concise context string from a user profile."""
        if not isinstance(profile, UserProfile):
            raise TypeError("Profile must be a UserProfile.")

        sections: list[str] = []

        attributes = [
            item
            for item in profile.attributes
            if item.confidence >= self._min_confidence
        ][: self._max_attributes]

        preferences = [
            item
            for item in profile.preferences
            if item.confidence >= self._min_confidence
        ][: self._max_preferences]

        if attributes:
            lines = ["User profile:"]

            for item in attributes:
                lines.append(
                    f"- {self._format_key(item.key)}: "
                    f"{self._format_value(item.value)}"
                )

            sections.append("\n".join(lines))

        if preferences:
            lines = ["User preferences:"]

            for item in preferences:
                lines.append(
                    f"- {self._format_key(item.key)}: "
                    f"{self._format_value(item.value)}"
                )

            sections.append("\n".join(lines))

        if not sections:
            return ""

        context = "\n".join(sections)

        if len(context) <= self._max_context_chars:
            return context

        return self._truncate(context)

    @staticmethod
    def _format_key(value: str) -> str:
        """Convert a structured key into readable text."""
        return value.replace("_", " ").strip()

    @staticmethod
    def _format_value(value: str | bool | int | float) -> str:
        """Format a scalar profile value consistently."""
        if isinstance(value, bool):
            return "yes" if value else "no"

        return str(value)

    def _truncate(self, context: str) -> str:
        """Truncate context without exceeding the configured limit."""
        if self._max_context_chars <= 3:
            return context[: self._max_context_chars]

        return context[: self._max_context_chars - 3].rstrip() + "..."
