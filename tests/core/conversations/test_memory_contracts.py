"""Tests for memory operation data contracts."""

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from seven.core.conversations import (
    MemorySearchMatch,
    MemorySearchRequest,
)


def test_memory_search_request_defaults():
    request = MemorySearchRequest(query="  software development  ")

    assert request.query == "software development"
    assert request.limit == 5
    assert request.min_importance == 0.0
    assert request.categories == ()
    assert request.conversation_id is None
    assert request.include_global is True


def test_memory_search_request_normalizes_categories():
    request = MemorySearchRequest(
        query="development",
        categories=("Project", "preference", "project"),
    )

    assert request.categories == ("project", "preference")


def test_memory_search_request_accepts_category_list():
    request = MemorySearchRequest(
        query="development",
        categories=["project", "goal"],
    )

    assert request.categories == ("project", "goal")


def test_memory_search_request_normalizes_conversation_id():
    request = MemorySearchRequest(
        query="development",
        conversation_id=" conversation-123 ",
    )

    assert request.conversation_id == "conversation-123"


@pytest.mark.parametrize("query", ["", "   "])
def test_memory_search_request_rejects_empty_query(query):
    with pytest.raises(ValueError):
        MemorySearchRequest(query=query)


def test_memory_search_request_rejects_non_string_query():
    with pytest.raises(TypeError):
        MemorySearchRequest(query=123)


@pytest.mark.parametrize("limit", [0, -1])
def test_memory_search_request_rejects_non_positive_limit(limit):
    with pytest.raises(ValueError):
        MemorySearchRequest(query="development", limit=limit)


@pytest.mark.parametrize("limit", [True, 2.5, "5"])
def test_memory_search_request_rejects_invalid_limit_type(limit):
    with pytest.raises(TypeError):
        MemorySearchRequest(query="development", limit=limit)


@pytest.mark.parametrize(
    "importance",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_memory_search_request_rejects_invalid_importance(importance):
    with pytest.raises(ValueError):
        MemorySearchRequest(
            query="development",
            min_importance=importance,
        )


@pytest.mark.parametrize("importance", [True, "0.5", None])
def test_memory_search_request_rejects_non_numeric_importance(importance):
    with pytest.raises(TypeError):
        MemorySearchRequest(
            query="development",
            min_importance=importance,
        )


def test_memory_search_request_rejects_invalid_categories():
    with pytest.raises(TypeError):
        MemorySearchRequest(
            query="development",
            categories="project",
        )


def test_memory_search_request_rejects_empty_conversation_id():
    with pytest.raises(ValueError):
        MemorySearchRequest(
            query="development",
            conversation_id="   ",
        )


def test_memory_search_request_rejects_invalid_global_flag():
    with pytest.raises(TypeError):
        MemorySearchRequest(
            query="development",
            include_global=1,
        )


def test_memory_search_request_is_immutable():
    request = MemorySearchRequest(query="development")

    with pytest.raises(FrozenInstanceError):
        request.limit = 10


def test_memory_search_match_accepts_valid_data():
    created_at = datetime.now(timezone.utc)

    match = MemorySearchMatch(
        memory_id="memory-123",
        content="The user is learning software development.",
        category="Knowledge",
        importance=0.8,
        relevance_score=0.9,
        combined_score=0.87,
        conversation_id="conversation-123",
        created_at=created_at,
    )

    assert match.memory_id == "memory-123"
    assert match.category == "knowledge"
    assert match.importance == 0.8
    assert match.relevance_score == 0.9
    assert match.combined_score == 0.87
    assert match.created_at == created_at


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("importance", -0.1),
        ("importance", 1.1),
        ("relevance_score", -0.1),
        ("relevance_score", 1.1),
        ("combined_score", -0.1),
        ("combined_score", 1.1),
        ("combined_score", float("nan")),
    ],
)
def test_memory_search_match_rejects_invalid_scores(field, value):
    data = {
        "memory_id": "memory-123",
        "content": "A memory entry.",
        "category": "general",
        "importance": 0.5,
        "relevance_score": 0.5,
        "combined_score": 0.5,
    }

    data[field] = value

    with pytest.raises(ValueError):
        MemorySearchMatch(**data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("importance", True),
        ("relevance_score", "0.5"),
        ("combined_score", None),
    ],
)
def test_memory_search_match_rejects_non_numeric_scores(field, value):
    data = {
        "memory_id": "memory-123",
        "content": "A memory entry.",
        "category": "general",
        "importance": 0.5,
        "relevance_score": 0.5,
        "combined_score": 0.5,
    }

    data[field] = value

    with pytest.raises(TypeError):
        MemorySearchMatch(**data)


def test_memory_search_match_rejects_empty_memory_id():
    with pytest.raises(ValueError):
        MemorySearchMatch(
            memory_id=" ",
            content="A memory entry.",
            category="general",
            importance=0.5,
            relevance_score=0.5,
            combined_score=0.5,
        )


def test_memory_search_match_rejects_empty_content():
    with pytest.raises(ValueError):
        MemorySearchMatch(
            memory_id="memory-123",
            content=" ",
            category="general",
            importance=0.5,
            relevance_score=0.5,
            combined_score=0.5,
        )


def test_memory_search_match_rejects_invalid_created_at():
    with pytest.raises(TypeError):
        MemorySearchMatch(
            memory_id="memory-123",
            content="A memory entry.",
            category="general",
            importance=0.5,
            relevance_score=0.5,
            combined_score=0.5,
            created_at="2026-01-01",
        )


def test_memory_search_match_is_immutable():
    match = MemorySearchMatch(
        memory_id="memory-123",
        content="A memory entry.",
        category="general",
        importance=0.5,
        relevance_score=0.5,
        combined_score=0.5,
    )

    with pytest.raises(FrozenInstanceError):
        match.content = "Updated content"