"""Tests for memory category validation and fallback handling."""

import pytest

from seven.core.conversations import MemoryCategoryValidator


@pytest.fixture
def validator():
    return MemoryCategoryValidator()


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("goal", "goal"),
        ("GOAL", "goal"),
        ("personal", "personal"),
        ("preference", "preference"),
        ("project", "project"),
        ("knowledge", "knowledge"),
        ("education", "education"),
        ("work", "work"),
        ("general", "general"),
    ],
)
def test_validate_registered_categories(validator, category, expected):
    assert validator.validate(category) == expected


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("preferences", "preference"),
        ("projects", "project"),
        ("goals", "goal"),
    ],
)
def test_validate_category_aliases(validator, alias, expected):
    assert validator.validate(alias) == expected


def test_validate_unknown_category_raises_error(validator):
    with pytest.raises(ValueError):
        validator.validate("unknown")


def test_validate_rejects_non_string_category(validator):
    with pytest.raises(TypeError):
        validator.validate(None)


def test_validate_rejects_empty_category(validator):
    with pytest.raises(ValueError):
        validator.validate("   ")


def test_resolve_unknown_category_uses_general_fallback(validator):
    assert validator.resolve("unknown") == "general"


def test_resolve_known_category_preserves_category(validator):
    assert validator.resolve("project") == "project"


def test_resolve_alias_returns_canonical_category(validator):
    assert validator.resolve("preferences") == "preference"


def test_fallback_category_can_be_configured():
    custom_validator = MemoryCategoryValidator(
        fallback_category="knowledge"
    )

    assert custom_validator.fallback_category == "knowledge"
    assert custom_validator.resolve("unknown") == "knowledge"


def test_constructor_rejects_invalid_fallback():
    with pytest.raises(ValueError):
        MemoryCategoryValidator(fallback_category="unknown")


@pytest.mark.parametrize(
    ("category", "expected"),
    [
        ("goal", True),
        ("preferences", True),
        ("projects", True),
        ("unknown", False),
        ("", False),
    ],
)
def test_is_valid(category, expected):
    validator = MemoryCategoryValidator()

    assert validator.is_valid(category) is expected


def test_is_valid_rejects_non_string_without_raising(validator):
    assert validator.is_valid(None) is False


def test_validate_many_normalizes_and_removes_duplicates(validator):
    result = validator.validate_many(
        [
            "goal",
            "GOAL",
            "preferences",
            "preference",
            "projects",
        ]
    )

    assert result == (
        "goal",
        "preference",
        "project",
    )


def test_validate_many_rejects_unknown_category(validator):
    with pytest.raises(ValueError):
        validator.validate_many(["goal", "unknown"])


def test_validate_many_rejects_non_collection(validator):
    with pytest.raises(TypeError):
        validator.validate_many("goal")


def test_fallback_category_is_immutable(validator):
    assert validator.fallback_category == "general"