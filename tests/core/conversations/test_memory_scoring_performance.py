
"""Performance regression tests for memory scoring."""

from types import SimpleNamespace

from seven.core.conversations.memory_scorer import MemoryScorer


def create_memory(
    memory_id: str,
    content: str,
    importance: float = 0.5,
) -> SimpleNamespace:
    """Create a lightweight memory record."""
    return SimpleNamespace(
        id=memory_id,
        content=content,
        category="general",
        importance=importance,
        conversation_id=None,
        created_at=None,
    )


def test_score_many_prepares_query_only_once(monkeypatch):
    """Batch scoring should prepare the query only once."""
    scorer = MemoryScorer()

    memories = [
        create_memory("1", "Python programming"),
        create_memory("2", "Python application development"),
        create_memory("3", "Java programming"),
    ]

    original_prepare_query = scorer._prepare_query
    call_count = 0

    def tracked_prepare_query(query):
        nonlocal call_count
        call_count += 1
        return original_prepare_query(query)

    monkeypatch.setattr(
        scorer,
        "_prepare_query",
        tracked_prepare_query,
    )

    results = scorer.score_many("Python programming", memories)

    assert len(results) == 3
    assert call_count == 1


def test_rank_prepares_query_only_once(monkeypatch):
    """Ranking should reuse prepared query data across memories."""
    scorer = MemoryScorer()

    memories = [
        create_memory("1", "Python programming"),
        create_memory("2", "Python application development"),
        create_memory("3", "Java programming"),
    ]

    original_prepare_query = scorer._prepare_query
    call_count = 0

    def tracked_prepare_query(query):
        nonlocal call_count
        call_count += 1
        return original_prepare_query(query)

    monkeypatch.setattr(
        scorer,
        "_prepare_query",
        tracked_prepare_query,
    )

    results = scorer.rank("Python programming", memories)

    assert len(results) == 3
    assert call_count == 1


def test_score_many_matches_individual_scoring_results():
    """Batch scoring should preserve individual scoring results."""
    scorer = MemoryScorer()

    memories = [
        create_memory("1", "Python programming", 0.8),
        create_memory("2", "Python application development", 0.6),
        create_memory("3", "Java programming", 0.4),
    ]

    batch_results = scorer.score_many(
        "Python programming",
        memories,
    )

    individual_results = [
        scorer.score("Python programming", memory)
        for memory in memories
    ]

    assert batch_results == individual_results


def test_score_many_preserves_input_order():
    """Batch scoring should preserve the input order."""
    scorer = MemoryScorer()

    memories = [
        create_memory("first", "Python programming"),
        create_memory("second", "Java programming"),
        create_memory("third", "Python development"),
    ]

    results = scorer.score_many("Python", memories)

    assert [result.memory_id for result in results] == [
        "first",
        "second",
        "third",
    ]


def test_rank_sorts_batch_results_by_score():
    """Ranking should sort batch results by combined score."""
    scorer = MemoryScorer()

    memories = [
        create_memory("low", "Java development", 0.1),
        create_memory("high", "Python programming", 0.9),
        create_memory("medium", "Python development", 0.5),
    ]

    results = scorer.rank("Python programming", memories)

    assert len(results) == 3

    assert [
        result.combined_score for result in results
    ] == sorted(
        [result.combined_score for result in results],
        reverse=True,
    )


def test_score_many_supports_generators():
    """Batch scoring should support one-pass iterables."""
    scorer = MemoryScorer()

    memories = (
        create_memory(str(index), f"Python project {index}")
        for index in range(3)
    )

    results = scorer.score_many("Python project", memories)

    assert len(results) == 3


def test_score_many_returns_empty_list_for_empty_memories():
    """Batch scoring should handle empty memory collections."""
    scorer = MemoryScorer()

    results = scorer.score_many("Python programming", [])

    assert results == []


def test_rank_with_limit_returns_only_top_k_results():
    """Limited ranking should return only the highest-scoring memories."""
    scorer = MemoryScorer()

    memories = [
        create_memory("low", "Java development", 0.1),
        create_memory("high", "Python programming", 0.9),
        create_memory("medium", "Python development", 0.5),
        create_memory("highest", "Python programming", 1.0),
    ]

    results = scorer.rank(
        "Python programming",
        memories,
        limit=2,
    )

    assert len(results) == 2
    assert results[0].memory_id == "highest"
    assert results[1].memory_id == "high"


def test_rank_with_limit_matches_full_ranking():
    """Limited ranking should match the prefix of full ranking."""
    scorer = MemoryScorer()

    memories = [
        create_memory("1", "Python programming", 0.8),
        create_memory("2", "Java programming", 0.4),
        create_memory("3", "Python development", 0.6),
        create_memory("4", "Python programming", 0.9),
    ]

    full_results = scorer.rank(
        "Python programming",
        memories,
    )

    limited_results = scorer.rank(
        "Python programming",
        memories,
        limit=2,
    )

    assert limited_results == full_results[:2]


def test_rank_with_limit_preserves_tie_order():
    """Limited ranking should preserve input order for tied scores."""
    scorer = MemoryScorer()

    memories = [
        create_memory("first", "Python programming", 0.5),
        create_memory("second", "Python programming", 0.5),
        create_memory("third", "Python programming", 0.5),
    ]

    results = scorer.rank(
        "Python programming",
        memories,
        limit=2,
    )

    assert [result.memory_id for result in results] == [
        "first",
        "second",
    ]


def test_prepare_text_cache_is_shared_across_scorer_instances():
    """Scorer instances should reuse the shared text preparation cache."""
    first_scorer = MemoryScorer()
    second_scorer = MemoryScorer()

    first_scorer._prepare_text.cache_clear()

    first_result = first_scorer._prepare_text("Python programming")
    second_result = second_scorer._prepare_text("Python programming")

    assert first_result == second_result
    assert first_scorer._prepare_text.cache_info().hits == 1



def test_score_many_reuses_cached_memory_content():
    """Repeated memory content should reuse cached preprocessing."""
    scorer = MemoryScorer()
    scorer._prepare_text.cache_clear()

    memories = [
        create_memory("1", "Python programming"),
        create_memory("2", "Python programming"),
        create_memory("3", "Python programming"),
    ]

    results = scorer.score_many("Python programming", memories)

    cache_info = scorer._prepare_text.cache_info()

    assert len(results) == 3
    assert cache_info.misses == 1
    assert cache_info.hits == 3


def test_cache_clear_resets_cache_statistics():
    """Clearing the cache should reset stored entries and statistics."""
    scorer = MemoryScorer()

    scorer._prepare_text("Python programming")
    scorer._prepare_text("Python programming")

    scorer._prepare_text.cache_clear()

    cache_info = scorer._prepare_text.cache_info()

    assert cache_info.hits == 0
    assert cache_info.misses == 0
    assert cache_info.currsize == 0
    assert cache_info.maxsize == 4096
