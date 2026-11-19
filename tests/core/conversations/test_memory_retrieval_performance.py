
"""Performance and validation tests for memory retrieval."""

from types import SimpleNamespace

import pytest

from seven.core.conversations.memory_config import (
    MemoryArchitectureConfig,
)
from seven.core.conversations.memory_contracts import (
    MemorySearchMatch,
)
from seven.core.conversations.memory_retriever import MemoryRetriever
from seven.core.conversations.memory_scorer import MemoryScorer


class FakeMemoryService:
    """Provide deterministic memory search results."""

    def __init__(self, memories: list[SimpleNamespace]) -> None:
        self.memories = memories
        self.search_limits: list[int] = []

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[SimpleNamespace]:
        self.search_limits.append(limit)
        return self.memories[:limit]


def make_memory(
    memory_id: str,
    content: str,
    *,
    importance: float = 0.5,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=memory_id,
        content=content,
        category="general",
        importance=importance,
        conversation_id=None,
        created_at=None,
    )


def make_retriever(
    service: FakeMemoryService,
    config: MemoryArchitectureConfig,
    scorer: MemoryScorer,
) -> MemoryRetriever:
    """Create a retriever without invoking its service type guard."""
    retriever = object.__new__(MemoryRetriever)
    retriever._memory_service = service
    retriever._max_results = 5
    retriever._min_importance = config.retrieval_min_importance
    retriever._config = config
    retriever._scorer = scorer
    return retriever


@pytest.mark.asyncio
async def test_adaptive_retrieval_scores_only_new_candidates() -> None:
    memories = [
        make_memory(
            f"memory-{index}",
            f"Unrelated subject number {index}.",
        )
        for index in range(20)
    ]
    relevant_memory = make_memory(
        "memory-relevant",
        "Bitcoin investment strategy and market analysis.",
    )
    memories.append(relevant_memory)

    service = FakeMemoryService(memories)
    config = MemoryArchitectureConfig(
        adaptive_retrieval_enabled=True,
        adaptive_retrieval_max_candidates=40,
        adaptive_retrieval_expansion_factor=2,
    )
    scorer = MemoryScorer(config)
    retriever = make_retriever(service, config, scorer)

    scored_batch_sizes: list[int] = []
    original_score_many = scorer.score_many

    def track_score_many(
        query: str,
        candidates: list[SimpleNamespace],
        context=None,
    ) -> list[MemorySearchMatch]:
        scored_batch_sizes.append(len(candidates))
        return original_score_many(
            query,
            candidates,
            context=context,
        )

    scorer.score_many = track_score_many

    result = await retriever._retrieve_adaptively(
        "bitcoin",
        result_limit=1,
        context=None,
    )

    assert result == [relevant_memory]
    assert service.search_limits == [20, 40]
    assert scored_batch_sizes == [20, 1]


@pytest.mark.asyncio
async def test_adaptive_retrieval_does_not_score_duplicate_ids_twice() -> None:
    memories = [
        make_memory(
            f"memory-{index}",
            f"Unrelated subject number {index}.",
        )
        for index in range(20)
    ]

    memories.append(
        make_memory(
            "memory-19",
            "Bitcoin investment strategy.",
        )
    )

    service = FakeMemoryService(memories)
    config = MemoryArchitectureConfig(
        adaptive_retrieval_enabled=True,
        adaptive_retrieval_max_candidates=40,
        adaptive_retrieval_expansion_factor=2,
    )
    scorer = MemoryScorer(config)
    retriever = make_retriever(service, config, scorer)

    scored_batch_sizes: list[int] = []
    original_score_many = scorer.score_many

    def track_score_many(
        query: str,
        candidates: list[SimpleNamespace],
        context=None,
    ) -> list[MemorySearchMatch]:
        scored_batch_sizes.append(len(candidates))
        return original_score_many(
            query,
            candidates,
            context=context,
        )

    scorer.score_many = track_score_many

    await retriever._retrieve_adaptively(
        "bitcoin",
        result_limit=2,
        context=None,
    )

    assert scored_batch_sizes == [20]


def test_rank_matches_preserves_descending_score_order() -> None:
    scorer = MemoryScorer()

    matches = [
        MemorySearchMatch(
            memory_id="low",
            content="Low relevance memory.",
            category="general",
            importance=0.2,
            relevance_score=0.2,
            combined_score=0.2,
        ),
        MemorySearchMatch(
            memory_id="high",
            content="High relevance memory.",
            category="general",
            importance=0.8,
            relevance_score=0.9,
            combined_score=0.87,
        ),
        MemorySearchMatch(
            memory_id="medium",
            content="Medium relevance memory.",
            category="general",
            importance=0.5,
            relevance_score=0.5,
            combined_score=0.5,
        ),
    ]

    ranked = scorer.rank_matches(matches)

    assert [match.memory_id for match in ranked] == [
        "high",
        "medium",
        "low",
    ]


def test_rank_matches_respects_top_k_limit() -> None:
    scorer = MemoryScorer()

    matches = [
        MemorySearchMatch(
            memory_id=f"memory-{index}",
            content=f"Memory content {index}.",
            category="general",
            importance=index / 10,
            relevance_score=index / 10,
            combined_score=index / 10,
        )
        for index in range(10)
    ]

    ranked = scorer.rank_matches(matches, limit=3)

    assert len(ranked) == 3
    assert [match.memory_id for match in ranked] == [
        "memory-9",
        "memory-8",
        "memory-7",
    ]


def test_rank_matches_preserves_input_order_for_equal_scores() -> None:
    scorer = MemoryScorer()

    matches = [
        MemorySearchMatch(
            memory_id=memory_id,
            content="Equivalent memory content.",
            category="general",
            importance=0.5,
            relevance_score=0.5,
            combined_score=0.5,
        )
        for memory_id in ("first", "second", "third")
    ]

    ranked = scorer.rank_matches(matches)

    assert [match.memory_id for match in ranked] == [
        "first",
        "second",
        "third",
    ]


def test_rank_matches_rejects_invalid_items() -> None:
    scorer = MemoryScorer()

    with pytest.raises(TypeError):
        scorer.rank_matches([object()])


@pytest.mark.parametrize(
    ("limit", "exception"),
    [
        (0, ValueError),
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ],
)
def test_rank_matches_validates_limit(
    limit: int,
    exception: type[Exception],
) -> None:
    scorer = MemoryScorer()

    with pytest.raises(exception):
        scorer.rank_matches([], limit=limit)


def test_rank_matches_returns_empty_list_for_empty_input() -> None:
    scorer = MemoryScorer()

    assert scorer.rank_matches([]) == []
