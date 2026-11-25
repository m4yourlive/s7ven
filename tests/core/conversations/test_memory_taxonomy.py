"""Tests for centralized memory category taxonomy."""

import pytest

from seven.core.conversations import (
    MemoryCategoryDefinition,
    MemoryCategoryTaxonomy,
)


def test_taxonomy_contains_all_categories():
    categories = {
        definition.category
        for definition in MemoryCategoryTaxonomy.get_all_definitions()
    }

    assert categories == {
        "general",
        "personal",
        "preference",
        "project",
        "goal",
        "work",
        "education",
        "knowledge",
    }


@pytest.mark.parametrize(
    ("category", "importance"),
    [
        ("goal", 0.85),
        ("project", 0.85),
        ("preference", 0.75),
        ("knowledge", 0.75),
        ("education", 0.70),
        ("work", 0.70),
        ("personal", 0.65),
        ("general", 0.50),
    ],
)
def test_taxonomy_returns_default_importance(category, importance):
    assert MemoryCategoryTaxonomy.get_importance(category) == importance


@pytest.mark.parametrize(
    ("category", "expected_filter"),
    [
        ("personal", "personal"),
        ("preference", "preferences"),
        ("project", "projects"),
        ("goal", "knowledge"),
        ("work", "knowledge"),
        ("education", "knowledge"),
        ("knowledge", "knowledge"),
        ("general", "knowledge"),
    ],
)
def test_taxonomy_maps_browser_filters(category, expected_filter):
    assert (
        MemoryCategoryTaxonomy.get_browser_filter(category)
        == expected_filter
    )


def test_taxonomy_returns_display_metadata():
    definition = MemoryCategoryTaxonomy.get_definition("preference")

    assert definition.category == "preference"
    assert definition.label == "Preferences"
    assert definition.description
    assert definition.default_importance == 0.75
    assert definition.browser_filter == "preferences"


def test_taxonomy_normalizes_category_names():
    definition = MemoryCategoryTaxonomy.get_definition("GOAL")

    assert definition.category == "goal"


def test_taxonomy_rejects_unknown_category():
    with pytest.raises(ValueError):
        MemoryCategoryTaxonomy.get_definition("unknown")


def test_taxonomy_rejects_non_string_category():
    with pytest.raises(TypeError):
        MemoryCategoryTaxonomy.get_definition(None)


def test_taxonomy_returns_immutable_definitions():
    definitions = MemoryCategoryTaxonomy.get_all_definitions()

    assert isinstance(definitions, tuple)


def test_taxonomy_classification_order():
    assert MemoryCategoryTaxonomy.classification_order() == (
        "goal",
        "preference",
        "knowledge",
        "project",
        "education",
        "work",
        "personal",
    )


def test_taxonomy_general_category_has_no_classification_priority():
    definition = MemoryCategoryTaxonomy.get_definition("general")

    assert definition.classification_priority is None


def test_category_definition_rejects_invalid_importance():
    with pytest.raises(ValueError):
        MemoryCategoryDefinition(
            category="goal",
            label="Goals",
            description="Long-term objectives.",
            default_importance=1.5,
            classification_priority=10,
            browser_filter="knowledge",
        )


def test_category_definition_rejects_invalid_priority():
    with pytest.raises(ValueError):
        MemoryCategoryDefinition(
            category="goal",
            label="Goals",
            description="Long-term objectives.",
            default_importance=0.8,
            classification_priority=0,
            browser_filter="knowledge",
        )


def test_category_definition_rejects_invalid_browser_filter():
    with pytest.raises(ValueError):
        MemoryCategoryDefinition(
            category="goal",
            label="Goals",
            description="Long-term objectives.",
            default_importance=0.8,
            classification_priority=10,
            browser_filter="unknown",
        )


def test_category_definition_is_immutable():
    definition = MemoryCategoryTaxonomy.get_definition("goal")

    with pytest.raises(AttributeError):
        definition.label = "Updated"