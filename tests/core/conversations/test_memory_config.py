"""Tests for memory architecture configuration."""

from dataclasses import FrozenInstanceError

import pytest

from seven.core.conversations import MemoryArchitectureConfig


def test_memory_config_defaults():
    config = MemoryArchitectureConfig()

    assert config.retrieval_limit == 5
    assert config.retrieval_candidate_multiplier == 5
    assert config.retrieval_min_candidates == 20
    assert config.retrieval_min_importance == 0.0
    assert config.relevance_weight == 0.7
    assert config.importance_weight == 0.3
    assert config.max_context_memories == 5


def test_memory_config_calculates_default_candidate_limit():
    config = MemoryArchitectureConfig()

    assert config.candidate_limit() == 25


def test_memory_config_respects_minimum_candidate_limit():
    config = MemoryArchitectureConfig()

    assert config.candidate_limit(1) == 20
    assert config.candidate_limit(4) == 20


def test_memory_config_scales_candidate_limit():
    config = MemoryArchitectureConfig()

    assert config.candidate_limit(10) == 50
    assert config.candidate_limit(25) == 125


def test_memory_config_uses_custom_values():
    config = MemoryArchitectureConfig(
        retrieval_limit=8,
        retrieval_candidate_multiplier=4,
        retrieval_min_candidates=30,
        retrieval_min_importance=0.2,
        relevance_weight=0.6,
        importance_weight=0.4,
        max_context_memories=7,
    )

    assert config.retrieval_limit == 8
    assert config.candidate_limit() == 32
    assert config.retrieval_min_importance == 0.2
    assert config.relevance_weight == 0.6
    assert config.importance_weight == 0.4
    assert config.max_context_memories == 7


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_limit",
        "retrieval_candidate_multiplier",
        "retrieval_min_candidates",
        "max_context_memories",
    ],
)
def test_memory_config_rejects_non_integer_fields(field):
    with pytest.raises(TypeError):
        MemoryArchitectureConfig(**{field: 2.5})


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_limit",
        "retrieval_candidate_multiplier",
        "retrieval_min_candidates",
        "max_context_memories",
    ],
)
def test_memory_config_rejects_boolean_integer_fields(field):
    with pytest.raises(TypeError):
        MemoryArchitectureConfig(**{field: True})


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_limit",
        "retrieval_candidate_multiplier",
        "retrieval_min_candidates",
        "max_context_memories",
    ],
)
def test_memory_config_rejects_non_positive_integer_fields(field):
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(**{field: 0})


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_min_importance",
        "relevance_weight",
        "importance_weight",
    ],
)
@pytest.mark.parametrize(
    "value",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_memory_config_rejects_invalid_unit_interval_values(
    field,
    value,
):
    values = {
        "relevance_weight": 0.7,
        "importance_weight": 0.3,
    }
    values[field] = value

    with pytest.raises(ValueError):
        MemoryArchitectureConfig(**values)


@pytest.mark.parametrize(
    "field",
    [
        "retrieval_min_importance",
        "relevance_weight",
        "importance_weight",
    ],
)
def test_memory_config_rejects_non_numeric_values(field):
    with pytest.raises(TypeError):
        MemoryArchitectureConfig(**{field: "0.5"})


@pytest.mark.parametrize(
    ("relevance_weight", "importance_weight"),
    [
        (0.8, 0.3),
        (0.0, 0.0),
        (0.6, 0.5),
        (0.2, 0.2),
    ],
)
def test_memory_config_rejects_invalid_weight_combinations(
    relevance_weight,
    importance_weight,
):
    with pytest.raises(ValueError):
        MemoryArchitectureConfig(
            relevance_weight=relevance_weight,
            importance_weight=importance_weight,
        )


@pytest.mark.parametrize(
    ("relevance_weight", "importance_weight"),
    [
        (0.7, 0.3),
        (0.5, 0.5),
        (0.6, 0.4),
        (0.0, 1.0),
        (1.0, 0.0),
    ],
)
def test_memory_config_accepts_valid_weight_combinations(
    relevance_weight,
    importance_weight,
):
    config = MemoryArchitectureConfig(
        relevance_weight=relevance_weight,
        importance_weight=importance_weight,
    )

    assert config.relevance_weight == relevance_weight
    assert config.importance_weight == importance_weight


def test_candidate_limit_rejects_non_positive_values():
    config = MemoryArchitectureConfig()

    with pytest.raises(ValueError):
        config.candidate_limit(0)

    with pytest.raises(ValueError):
        config.candidate_limit(-1)


@pytest.mark.parametrize(
    "limit",
    [True, 2.5, "5"],
)
def test_candidate_limit_rejects_invalid_types(limit):
    config = MemoryArchitectureConfig()

    with pytest.raises(TypeError):
        config.candidate_limit(limit)


def test_memory_config_is_immutable():
    config = MemoryArchitectureConfig()

    with pytest.raises(FrozenInstanceError):
        config.retrieval_limit = 10