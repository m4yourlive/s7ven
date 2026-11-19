"""Validation and fallback handling for memory categories."""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from seven.core.conversations.memory_taxonomy import (
    MemoryCategoryTaxonomy,
)
from seven.core.conversations.memory_types import (
    MemoryCategory,
    normalize_memory_category,
)


class MemoryCategoryValidator:
    """Validate category names and resolve unknown categories safely."""

    _ALIASES: Mapping[str, str] = MappingProxyType(
        {
            "preferences": MemoryCategory.PREFERENCE.value,
            "projects": MemoryCategory.PROJECT.value,
            "goals": MemoryCategory.GOAL.value,
        }
    )

    def __init__(self, fallback_category: str = "general") -> None:
        """Initialize the validator with a registered fallback category."""
        self._fallback_category = self.validate(fallback_category)

    @property
    def fallback_category(self) -> str:
        """Return the configured fallback category."""
        return self._fallback_category

    def validate(self, category: str) -> str:
        """Return a canonical category or raise a validation error."""
        normalized = normalize_memory_category(category)
        canonical = self._ALIASES.get(normalized, normalized)

        definition = MemoryCategoryTaxonomy.get_definition(canonical)

        return definition.category

    def resolve(self, category: str) -> str:
        """Resolve a category, falling back when it is unknown."""
        try:
            return self.validate(category)
        except ValueError:
            return self._fallback_category

    def is_valid(self, category: str) -> bool:
        """Return whether a category can be resolved without fallback."""
        try:
            self.validate(category)
            return True
        except (TypeError, ValueError):
            return False

    def validate_many(
        self,
        categories: tuple[str, ...] | list[str],
    ) -> tuple[str, ...]:
        """Validate and normalize a collection of categories."""
        if not isinstance(categories, (tuple, list)):
            raise TypeError("Categories must be a tuple or list.")

        normalized: list[str] = []

        for category in categories:
            canonical = self.validate(category)

            if canonical not in normalized:
                normalized.append(canonical)

        return tuple(normalized)