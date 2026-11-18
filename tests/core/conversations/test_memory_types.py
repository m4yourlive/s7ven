"""Tests for shared memory types and category definitions."""

import pytest

from seven.core.conversations import (
    MEMORY_CATEGORY_FILTERS,
    MemoryCategory,
    MemoryCategoryFilter,
    matches_memory_category_filter,
    normalize_memory_category,
)


def test_memory_category_values():
    assert MemoryCategory.GENERAL == "general"
    assert MemoryCategory.PERSONAL == "personal"
    assert MemoryCategory.PREFERENCE == "preference"
    assert MemoryCategory.PROJECT == "project"
    assert MemoryCategory.GOAL == "goal"
    assert MemoryCategory.WORK == "work"
    assert MemoryCategory.EDUCATION == "education"
    assert MemoryCategory.KNOWLEDGE == "knowledge"


def test_memory_category_filter_values():
    assert MemoryCategoryFilter.ALL == "all"
    assert MemoryCategoryFilter.PERSONAL == "personal"
    assert MemoryCategoryFilter.PREFERENCES == "preferences"
    assert MemoryCategoryFilter.PROJECTS == "projects"
    assert MemoryCategoryFilter.KNOWLEDGE == "knowledge"


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("Personal", "personal"),
        ("  PERSONAL  ", "personal"),
        ("long-term-goal", "long_term_goal"),
        ("Software Development", "software_development"),
        ("project", "project"),
    ],
)
def test_normalize_memory_category(category, expected):
    assert normalize_memory_category(category) == expected


def test_normalize_memory_category_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_memory_category(None)


@pytest.mark.parametrize(
    "category",
    [
        "",
        "   ",
    ],
)
def test_normalize_memory_category_rejects_empty_values(category):
    with pytest.raises(ValueError):
        normalize_memory_category(category)


def test_normalize_memory_category_rejects_values_over_50_characters():
    with pytest.raises(ValueError):
        normalize_memory_category("a" * 51)


def test_all_filter_matches_every_category():
    assert matches_memory_category_filter(
        "goal",
        MemoryCategoryFilter.ALL,
    )

    assert matches_memory_category_filter(
        "project",
        MemoryCategoryFilter.ALL,
    )


def test_personal_filter_matches_personal_category():
    assert matches_memory_category_filter(
        "personal",
        "personal",
    )


def test_preferences_filter_supports_backend_and_ui_categories():
    assert matches_memory_category_filter(
        "preference",
        "preferences",
    )

    assert matches_memory_category_filter(
        "preferences",
        "preferences",
    )


def test_projects_filter_supports_backend_and_ui_categories():
    assert matches_memory_category_filter(
        "project",
        "projects",
    )

    assert matches_memory_category_filter(
        "projects",
        "projects",
    )


def test_knowledge_filter_matches_knowledge_category():
    assert matches_memory_category_filter(
        "knowledge",
        "knowledge",
    )


def test_unknown_filter_uses_exact_category_matching():
    assert matches_memory_category_filter(
        "education",
        "education",
    )

    assert not matches_memory_category_filter(
        "education",
        "work",
    )


def test_category_filter_mapping_is_read_only():
    with pytest.raises(TypeError):
        MEMORY_CATEGORY_FILTERS["custom"] = ("custom",)