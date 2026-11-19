"""Shared types and category definitions for memory management."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType


class MemoryCategory(StrEnum):
    """Canonical categories for long-term memories."""

    GENERAL = "general"
    PERSONAL = "personal"
    PREFERENCE = "preference"
    PROJECT = "project"
    GOAL = "goal"
    WORK = "work"
    EDUCATION = "education"
    KNOWLEDGE = "knowledge"


class MemoryCategoryFilter(StrEnum):
    """Categories exposed by the memory browser interface."""

    ALL = "all"
    PERSONAL = "personal"
    PREFERENCES = "preferences"
    PROJECTS = "projects"
    KNOWLEDGE = "knowledge"


MEMORY_CATEGORY_FILTERS: Mapping[str, tuple[str, ...]] = (
    MappingProxyType(
        {
            MemoryCategoryFilter.ALL.value: (),
            MemoryCategoryFilter.PERSONAL.value: (
                MemoryCategory.PERSONAL.value,
            ),
            MemoryCategoryFilter.PREFERENCES.value: (
                MemoryCategory.PREFERENCE.value,
                "preferences",
            ),
            MemoryCategoryFilter.PROJECTS.value: (
                MemoryCategory.PROJECT.value,
                "projects",
            ),
            MemoryCategoryFilter.KNOWLEDGE.value: (
                MemoryCategory.KNOWLEDGE.value,
            ),
        }
    )
)


def normalize_memory_category(category: str) -> str:
    """Normalize a memory category without changing its meaning."""
    if not isinstance(category, str):
        raise TypeError("Memory category must be a string.")

    normalized = "_".join(category.strip().lower().replace("-", " ").split())

    if not normalized:
        raise ValueError("Memory category cannot be empty.")

    if len(normalized) > 50:
        raise ValueError(
            "Memory category cannot exceed 50 characters."
        )

    return normalized


def matches_memory_category_filter(
    category: str,
    category_filter: str,
) -> bool:
    """Check whether a memory belongs to a browser category filter."""
    normalized_category = normalize_memory_category(category)
    normalized_filter = normalize_memory_category(category_filter)

    if normalized_filter == MemoryCategoryFilter.ALL.value:
        return True

    supported_categories = MEMORY_CATEGORY_FILTERS.get(
        normalized_filter
    )

    if supported_categories is None:
        return normalized_category == normalized_filter

    return normalized_category in supported_categories