"""Central taxonomy and metadata for memory categories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from seven.core.conversations.memory_types import (
    MemoryCategory,
    MemoryCategoryFilter,
    normalize_memory_category,
)


@dataclass(frozen=True, slots=True)
class MemoryCategoryDefinition:
    """Describe the metadata and behavior of a memory category."""

    category: str
    label: str
    description: str
    default_importance: float
    classification_priority: int | None
    browser_filter: str

    def __post_init__(self) -> None:
        """Validate category metadata."""
        normalized_category = normalize_memory_category(
            self.category
        )
        object.__setattr__(
            self,
            "category",
            normalized_category,
        )

        for field_name in ("label", "description"):
            value = getattr(self, field_name)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{field_name} cannot be empty."
                )

            object.__setattr__(
                self,
                field_name,
                value.strip(),
            )

        if (
            isinstance(self.default_importance, bool)
            or not isinstance(self.default_importance, (int, float))
        ):
            raise TypeError("Default importance must be numeric.")

        importance = float(self.default_importance)

        if not math.isfinite(importance):
            raise ValueError("Default importance must be finite.")

        if not 0.0 <= importance <= 1.0:
            raise ValueError(
                "Default importance must be between 0.0 and 1.0."
            )

        object.__setattr__(
            self,
            "default_importance",
            importance,
        )

        if self.classification_priority is not None:
            if (
                isinstance(self.classification_priority, bool)
                or not isinstance(self.classification_priority, int)
            ):
                raise TypeError(
                    "Classification priority must be an integer or None."
                )

            if self.classification_priority < 1:
                raise ValueError(
                    "Classification priority must be positive."
                )

        if not isinstance(self.browser_filter, str):
            raise TypeError("Browser filter must be a string.")

        valid_filters = {
            item.value for item in MemoryCategoryFilter
        }

        if self.browser_filter not in valid_filters:
            raise ValueError("Unsupported memory browser filter.")


class MemoryCategoryTaxonomy:
    """Provide centralized definitions and category mappings."""

    _DEFINITIONS: Mapping[str, MemoryCategoryDefinition] = (
        MappingProxyType(
            {
                MemoryCategory.GOAL.value: MemoryCategoryDefinition(
                    category="goal",
                    label="Goals",
                    description="Long-term objectives, targets, and plans.",
                    default_importance=0.85,
                    classification_priority=10,
                    browser_filter="knowledge",
                ),
                MemoryCategory.PREFERENCE.value: MemoryCategoryDefinition(
                    category="preference",
                    label="Preferences",
                    description="Personal choices, likes, and dislikes.",
                    default_importance=0.75,
                    classification_priority=20,
                    browser_filter="preferences",
                ),
                MemoryCategory.KNOWLEDGE.value: MemoryCategoryDefinition(
                    category="knowledge",
                    label="Knowledge",
                    description="Skills, expertise, and learned information.",
                    default_importance=0.75,
                    classification_priority=30,
                    browser_filter="knowledge",
                ),
                MemoryCategory.PROJECT.value: MemoryCategoryDefinition(
                    category="project",
                    label="Projects",
                    description="Projects, products, and ongoing initiatives.",
                    default_importance=0.85,
                    classification_priority=40,
                    browser_filter="projects",
                ),
                MemoryCategory.EDUCATION.value: MemoryCategoryDefinition(
                    category="education",
                    label="Education",
                    description="Studies, learning, and academic background.",
                    default_importance=0.70,
                    classification_priority=50,
                    browser_filter="knowledge",
                ),
                MemoryCategory.WORK.value: MemoryCategoryDefinition(
                    category="work",
                    label="Work",
                    description="Employment, professional work, and career.",
                    default_importance=0.70,
                    classification_priority=60,
                    browser_filter="knowledge",
                ),
                MemoryCategory.PERSONAL.value: MemoryCategoryDefinition(
                    category="personal",
                    label="Personal",
                    description="Personal identity and background information.",
                    default_importance=0.65,
                    classification_priority=70,
                    browser_filter="personal",
                ),
                MemoryCategory.GENERAL.value: MemoryCategoryDefinition(
                    category="general",
                    label="General",
                    description="Useful information without a specific category.",
                    default_importance=0.50,
                    classification_priority=None,
                    browser_filter="knowledge",
                ),
            }
        )
    )

    @classmethod
    def get_definition(
        cls,
        category: str,
    ) -> MemoryCategoryDefinition:
        """Return the definition associated with a category."""
        normalized = normalize_memory_category(category)
        definition = cls._DEFINITIONS.get(normalized)

        if definition is None:
            raise ValueError(
                f"Unknown memory category: {normalized}"
            )

        return definition

    @classmethod
    def get_all_definitions(
        cls,
    ) -> tuple[MemoryCategoryDefinition, ...]:
        """Return all registered category definitions."""
        return tuple(cls._DEFINITIONS.values())

    @classmethod
    def classification_order(cls) -> tuple[str, ...]:
        """Return categories ordered by classification priority."""
        definitions = (
            definition
            for definition in cls._DEFINITIONS.values()
            if definition.classification_priority is not None
        )

        ordered = sorted(
            definitions,
            key=lambda definition: definition.classification_priority,
        )

        return tuple(
            definition.category for definition in ordered
        )

    @classmethod
    def get_importance(cls, category: str) -> float:
        """Return the default importance for a category."""
        return cls.get_definition(category).default_importance

    @classmethod
    def get_browser_filter(cls, category: str) -> str:
        """Map a stored category to a supported memory browser filter."""
        return cls.get_definition(category).browser_filter