
"""Extract structured user profile attributes and preferences."""

from __future__ import annotations

import re

from seven.core.conversations.user_profile_contracts import (
    UserPreference,
    UserProfile,
    UserProfileAttribute,
)


class UserProfileExtractor:
    """Extract explicit profile attributes and preferences from text."""

    _PROFILE_PATTERNS = (
        (
            "name",
            re.compile(r"\bmy name is\s+(.+)", re.IGNORECASE),
        ),
        (
            "age",
            re.compile(
                r"\bi am\s+(\d{1,3})\s+years old\b",
                re.IGNORECASE,
            ),
        ),
        (
            "location",
            re.compile(r"\bi live in\s+(.+)", re.IGNORECASE),
        ),
        (
            "occupation",
            re.compile(r"\bi work as\s+(.+)", re.IGNORECASE),
        ),
        (
            "field_of_study",
            re.compile(r"\bi study\s+(.+)", re.IGNORECASE),
        ),
        (
            "role",
            re.compile(
                r"\bi am (?:a|an)\s+(.+)",
                re.IGNORECASE,
            ),
        ),
    )

    _PREFERENCE_PATTERN = re.compile(
        r"\bi (prefer|like|dislike|love|hate)\s+(.+)",
        re.IGNORECASE,
    )

    _PREFERENCE_KEY_SIGNALS = {
        "design": (
            "design",
            "interface",
            "ui",
            "ux",
            "visual",
            "layout",
        ),
        "communication": (
            "response",
            "responses",
            "concise",
            "answer",
            "answers",
            "explanation",
            "explanations",
            "communication",
            "conversation",
            "writing",
        ),
        "coding": (
            "code",
            "coding",
            "programming",
            "development",
            "python",
            "javascript",
        ),
        "learning": (
            "learning",
            "study",
            "studying",
            "tutorial",
            "lesson",
        ),
        "workflow": (
            "workflow",
            "process",
            "productivity",
            "organization",
        ),
    }

    _ACTION_VALUES = {
        "prefer": "prefers",
        "like": "likes",
        "dislike": "dislikes",
        "love": "loves",
        "hate": "hates",
    }

    def __init__(
        self,
        confidence_threshold: float = 0.75,
    ) -> None:
        """Initialize the extractor with a confidence threshold."""
        if isinstance(confidence_threshold, bool) or not isinstance(
            confidence_threshold,
            (int, float),
        ):
            raise TypeError(
                "Confidence threshold must be numeric."
            )

        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                "Confidence threshold must be between 0.0 and 1.0."
            )

        self._confidence_threshold = float(confidence_threshold)

    def extract(self, text: str) -> UserProfile:
        """Extract a structured user profile from text."""
        if not isinstance(text, str):
            raise TypeError(
                "Profile extraction input must be a string."
            )

        if not text.strip():
            return UserProfile()

        attributes: dict[str, UserProfileAttribute] = {}
        preferences: dict[str, UserPreference] = {}

        for sentence in self._split_sentences(text):
            normalized = self._normalize_sentence(sentence)

            if not normalized:
                continue

            attribute = self._extract_attribute(normalized)

            if attribute is not None:
                self._store_attribute(attributes, attribute)
                continue

            preference = self._extract_preference(normalized)

            if preference is not None:
                self._store_preference(preferences, preference)

        return UserProfile(
            attributes=tuple(attributes.values()),
            preferences=tuple(preferences.values()),
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into independent extraction candidates."""
        return re.split(
            r"(?<=[.!?])\s+|\n+",
            text.strip(),
        )

    @staticmethod
    def _normalize_sentence(sentence: str) -> str:
        """Normalize whitespace and surrounding punctuation."""
        return re.sub(
            r"\s+",
            " ",
            sentence.strip().strip(" \t\r\n.!"),
        )

    def _extract_attribute(
        self,
        sentence: str,
    ) -> UserProfileAttribute | None:
        """Extract one explicitly stated profile attribute."""
        for key, pattern in self._PROFILE_PATTERNS:
            match = pattern.fullmatch(sentence)

            if match is None:
                continue

            raw_value = match.group(1).strip()

            if not raw_value:
                continue

            if key == "age":
                value: str | int = int(raw_value)
            else:
                value = self._clean_value(raw_value)

            if isinstance(value, str) and not value:
                continue

            attribute = UserProfileAttribute(
                key=key,
                value=value,
                confidence=0.95,
                source="explicit",
            )

            if attribute.confidence < self._confidence_threshold:
                return None

            return attribute

        return None

    def _extract_preference(
        self,
        sentence: str,
    ) -> UserPreference | None:
        """Extract one explicitly stated user preference."""
        match = self._PREFERENCE_PATTERN.fullmatch(sentence)

        if match is None:
            return None

        action = match.group(1).casefold()
        raw_value = self._clean_value(match.group(2))

        if not raw_value:
            return None

        key = self._build_preference_key(raw_value)
        action_value = self._ACTION_VALUES[action]

        preference = UserPreference(
            key=key,
            value=f"{action_value} {raw_value}",
            confidence=0.90,
            source="explicit",
        )

        if preference.confidence < self._confidence_threshold:
            return None

        return preference

    def _build_preference_key(self, value: str) -> str:
        """Create a stable preference key from its subject."""
        normalized = value.casefold()

        for category, signals in self._PREFERENCE_KEY_SIGNALS.items():
            if any(
                self._contains_phrase(normalized, signal)
                for signal in signals
            ):
                return f"preference_{category}"

        words = re.findall(r"[a-z0-9]+", normalized)

        if not words:
            return "preference_general"

        subject = "_".join(words[:3])

        return f"preference_{subject}"

    @staticmethod
    def _contains_phrase(content: str, phrase: str) -> bool:
        """Match complete words and phrases."""
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"

        return re.search(pattern, content) is not None

    @staticmethod
    def _clean_value(value: str) -> str:
        """Remove surrounding punctuation from an extracted value."""
        return value.strip().strip(" \t\r\n.,!?;:")

    @staticmethod
    def _store_attribute(
        attributes: dict[str, UserProfileAttribute],
        candidate: UserProfileAttribute,
    ) -> None:
        """Store the highest-confidence attribute for each key."""
        existing = attributes.get(candidate.key)

        if (
            existing is None
            or candidate.confidence > existing.confidence
        ):
            attributes[candidate.key] = candidate

    @staticmethod
    def _store_preference(
        preferences: dict[str, UserPreference],
        candidate: UserPreference,
    ) -> None:
        """Store the highest-confidence preference for each key."""
        existing = preferences.get(candidate.key)

        if (
            existing is None
            or candidate.confidence > existing.confidence
        ):
            preferences[candidate.key] = candidate
