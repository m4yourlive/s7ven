
"""Relevance scoring and ranking for long-term memories."""

from __future__ import annotations
from functools import lru_cache
import heapq
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from seven.core.conversations.memory_config import (
    MemoryArchitectureConfig,
)
from seven.core.conversations.memory_contracts import (
    MemoryScoreBreakdown,
    MemoryScoringContext,
    MemorySearchMatch,
)
from seven.core.conversations.memory_normalizer import (
    MemoryCandidateNormalizer,
)


class MemoryRecord(Protocol):
    """Describe the required fields for memory scoring."""

    id: str
    content: str
    category: str
    importance: float


@dataclass(frozen=True, slots=True)
class MemoryRelevance:
    """Represent lexical and contextual relevance components."""

    token_coverage: float
    token_precision: float
    jaccard_similarity: float
    phrase_match: float
    relevance_score: float
    lexical_relevance_score: float
    category_match: float = 0.0
    conversation_match: float = 0.0
    recency_score: float = 0.0


class MemoryScorer:
    """Calculate lexical, contextual, and combined memory scores."""

    _TOKEN_PATTERN = re.compile(r"\b\w+\b")

    def __init__(
        self,
        config: MemoryArchitectureConfig | None = None,
    ) -> None:
        """Initialize the scorer with architecture configuration."""
        if config is not None and not isinstance(
            config,
            MemoryArchitectureConfig,
        ):
            raise TypeError(
                "config must be a MemoryArchitectureConfig."
            )

        self._config = config or MemoryArchitectureConfig()

    @classmethod
    def tokenize(cls, text: str) -> frozenset[str]:
        """Return unique normalized tokens from text."""
        return cls._prepare_text(text)[1]

    @classmethod
    def _tokenize_normalized(
        cls,
        normalized_text: str,
    ) -> frozenset[str]:
        """Tokenize text that has already been normalized."""
        return frozenset(
            token
            for token in cls._TOKEN_PATTERN.findall(normalized_text)
            if len(token) > 1
        )

    @classmethod
    @lru_cache(maxsize=4096)
    def _prepare_text(
        cls,
        text: str,
    ) -> tuple[str, frozenset[str]]:
        """Normalize and tokenize text using a bounded cache."""
        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        normalized = MemoryCandidateNormalizer.normalize_text(text)

        tokens = frozenset(
            token
            for token in cls._TOKEN_PATTERN.findall(normalized)
            if len(token) > 1
        )

        return normalized, tokens

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Validate and normalize a search query."""
        if not isinstance(query, str):
            raise TypeError("Search query must be a string.")

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        normalized = MemoryCandidateNormalizer.normalize_text(query)

        if not normalized:
            raise ValueError("Search query cannot be empty.")

        return normalized

    def _prepare_query(
        self,
        query: str,
    ) -> tuple[str, frozenset[str]]:
        """Prepare reusable query data for scoring operations."""
        if not isinstance(query, str):
            raise TypeError("Search query must be a string.")

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        normalized, tokens = self._prepare_text(query)

        if not normalized:
            raise ValueError("Search query cannot be empty.")

        return normalized, tokens

    def calculate_relevance(
        self,
        query: str,
        content: str,
    ) -> MemoryRelevance:
        """Calculate lexical relevance between query and memory content."""
        if not isinstance(query, str):
            raise TypeError("Search query must be a string.")

        if not isinstance(content, str):
            raise TypeError("Memory content must be a string.")

        normalized_query, query_tokens = self._prepare_query(query)

        return self._calculate_relevance_prepared(
            normalized_query,
            query_tokens,
            content,
        )


    def _calculate_relevance_prepared(
        self,
        normalized_query: str,
        query_tokens: frozenset[str],
        content: str,
    ) -> MemoryRelevance:
        """Calculate lexical relevance using prepared query data."""
        if not isinstance(content, str):
            raise TypeError("Memory content must be a string.")

        normalized_content, content_tokens = self._prepare_text(content)

        if not normalized_content:
            raise ValueError("Memory content cannot be empty.")

        if not query_tokens or not content_tokens:
            token_coverage = 0.0
            token_precision = 0.0
            jaccard_similarity = 0.0
        else:
            common_count = len(query_tokens.intersection(content_tokens))
            union_count = len(query_tokens.union(content_tokens))

            token_coverage = common_count / len(query_tokens)
            token_precision = common_count / len(content_tokens)
            jaccard_similarity = common_count / union_count

        phrase_match = float(
            f" {normalized_query} "
            in f" {normalized_content} "
        )

        lexical_score = (
            token_coverage * self._config.lexical_coverage_weight
            + token_precision * self._config.lexical_precision_weight
            + jaccard_similarity * self._config.lexical_jaccard_weight
            + phrase_match * self._config.lexical_phrase_weight
        )

        normalized_score = self._clamp(lexical_score)

        return MemoryRelevance(
            token_coverage=self._clamp(token_coverage),
            token_precision=self._clamp(token_precision),
            jaccard_similarity=self._clamp(jaccard_similarity),
            phrase_match=phrase_match,
            relevance_score=normalized_score,
            lexical_relevance_score=normalized_score,
        )


    def explain(
        self,
        query: str,
        memory: MemoryRecord,
        context: MemoryScoringContext | None = None,
    ) -> MemoryScoreBreakdown:
        """Return a detailed explanation of a memory's ranking score."""
        normalized_query, query_tokens = self._prepare_query(query)

        self._validate_context(context)
        self._validate_memory_record(memory)

        return self._explain_prepared(
            normalized_query,
            query_tokens,
            memory,
            context,
        )

    def _explain_prepared(
        self,
        normalized_query: str,
        query_tokens: frozenset[str],
        memory: MemoryRecord,
        context: MemoryScoringContext | None,
    ) -> MemoryScoreBreakdown:
        """Build score diagnostics using prepared query data."""
        relevance = self._calculate_relevance_prepared(
            normalized_query,
            query_tokens,
            memory.content,
        )

        category_score = 0.0
        conversation_score = 0.0
        recency_score = 0.0

        memory_conversation_id = getattr(
            memory,
            "conversation_id",
            None,
        )
        memory_created_at = getattr(
            memory,
            "created_at",
            None,
        )

        contextual_score = relevance.lexical_relevance_score

        if context is not None:
            contextual_score = self._calculate_contextual_relevance(
                relevance=relevance,
                memory=memory,
                context=context,
            )

            if context.preferred_categories:
                category_score = float(
                    memory.category.casefold()
                    in context.preferred_categories
                )

            if context.conversation_id is not None:
                conversation_score = float(
                    memory_conversation_id == context.conversation_id
                )

            if memory_created_at is not None:
                recency_score = self._calculate_recency(
                    memory_created_at,
                    context,
                )

        importance_score = float(memory.importance)

        combined_score = (
            contextual_score * self._config.relevance_weight
            + importance_score * self._config.importance_weight
        )

        return MemoryScoreBreakdown(
            lexical_score=relevance.lexical_relevance_score,
            category_score=category_score,
            conversation_score=conversation_score,
            recency_score=recency_score,
            contextual_score=self._clamp(contextual_score),
            importance_score=importance_score,
            combined_score=self._clamp(combined_score),
        )

    def score(
        self,
        query: str,
        memory: MemoryRecord,
        context: MemoryScoringContext | None = None,
    ) -> MemorySearchMatch:
        """Calculate a combined score and return a structured match."""
        normalized_query, query_tokens = self._prepare_query(query)

        self._validate_context(context)
        self._validate_memory_record(memory)

        return self._score_prepared(
            normalized_query,
            query_tokens,
            memory,
            context,
        )

    def score_many(
        self,
        query: str,
        memories: Iterable[MemoryRecord],
        context: MemoryScoringContext | None = None,
    ) -> list[MemorySearchMatch]:
        """Score multiple memories while reusing prepared query data."""
        normalized_query, query_tokens = self._prepare_query(query)
        self._validate_context(context)

        matches: list[MemorySearchMatch] = []

        for memory in memories:
            self._validate_memory_record(memory)

            matches.append(
                self._score_prepared(
                    normalized_query,
                    query_tokens,
                    memory,
                    context,
                )
            )

        return matches

    def _score_prepared(
        self,
        normalized_query: str,
        query_tokens: frozenset[str],
        memory: MemoryRecord,
        context: MemoryScoringContext | None,
    ) -> MemorySearchMatch:
        """Score one memory using already prepared query data."""
        breakdown = self._explain_prepared(
            normalized_query,
            query_tokens,
            memory,
            context,
        )

        return MemorySearchMatch(
            memory_id=memory.id,
            content=memory.content,
            category=memory.category,
            importance=breakdown.importance_score,
            relevance_score=breakdown.contextual_score,
            combined_score=breakdown.combined_score,
            conversation_id=getattr(
                memory,
                "conversation_id",
                None,
            ),
            created_at=getattr(
                memory,
                "created_at",
                None,
            ),
        )

    def rank(
        self,
        query: str,
        memories: Iterable[MemoryRecord],
        limit: int | None = None,
        context: MemoryScoringContext | None = None,
    ) -> list[MemorySearchMatch]:
        """Score and rank memories by relevance and importance."""
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int):
                raise TypeError("Ranking limit must be an integer.")

            if limit < 1:
                raise ValueError("Ranking limit must be positive.")

        matches = self.score_many(
            query,
            memories,
            context=context,
        )

        ranking_key = lambda match: (
            match.combined_score,
            match.relevance_score,
            match.importance,
            match.created_at.isoformat()
            if match.created_at is not None
            else "",
        )

        if limit is None:
            matches.sort(
                key=ranking_key,
                reverse=True,
            )
            return matches

        return heapq.nlargest(
            limit,
            matches,
            key=ranking_key,
        )

    def _calculate_contextual_relevance(
        self,
        relevance: MemoryRelevance,
        memory: MemoryRecord,
        context: MemoryScoringContext,
    ) -> float:
        """Combine available relevance and contextual signals."""
        weighted_score = (
            relevance.lexical_relevance_score
            * self._config.context_lexical_weight
        )
        total_weight = self._config.context_lexical_weight

        if context.preferred_categories:
            category_match = float(
                memory.category.casefold()
                in context.preferred_categories
            )

            weighted_score += (
                category_match * self._config.context_category_weight
            )
            total_weight += self._config.context_category_weight

        if context.conversation_id is not None:
            conversation_id = getattr(
                memory,
                "conversation_id",
                None,
            )

            conversation_match = float(
                conversation_id == context.conversation_id
            )

            weighted_score += (
                conversation_match
                * self._config.context_conversation_weight
            )
            total_weight += self._config.context_conversation_weight

        created_at = getattr(memory, "created_at", None)

        if created_at is not None:
            recency_score = self._calculate_recency(
                created_at,
                context,
            )

            weighted_score += (
                recency_score * self._config.context_recency_weight
            )
            total_weight += self._config.context_recency_weight

        if total_weight <= 0.0:
            return 0.0

        return self._clamp(weighted_score / total_weight)

    def _calculate_recency(
        self,
        created_at: datetime,
        context: MemoryScoringContext,
    ) -> float:
        """Calculate recency using exponential half-life decay."""
        if not isinstance(created_at, datetime):
            raise TypeError(
                "Memory creation time must be a datetime."
            )

        reference_time = (
            context.reference_time or datetime.now(timezone.utc)
        )

        created_utc = self._as_utc(created_at)
        reference_utc = self._as_utc(reference_time)

        age_seconds = max(
            0.0,
            (reference_utc - created_utc).total_seconds(),
        )

        age_days = age_seconds / 86400.0

        half_life = (
            context.recency_half_life_days
            if context.recency_half_life_days is not None
            else self._config.recency_half_life_days
        )

        decay = math.exp(
            -math.log(2.0) * age_days / half_life
        )

        return self._clamp(decay)

    @staticmethod
    def _validate_context(
        context: MemoryScoringContext | None,
    ) -> None:
        """Validate optional scoring context."""
        if context is not None and not isinstance(
            context,
            MemoryScoringContext,
        ):
            raise TypeError(
                "context must be a MemoryScoringContext."
            )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        """Normalize naive and timezone-aware datetimes to UTC."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)

        return value.astimezone(timezone.utc)

    @staticmethod
    def _validate_memory_record(memory: MemoryRecord) -> None:
        """Validate required memory fields and optional metadata."""
        required_fields = (
            "id",
            "content",
            "category",
            "importance",
        )

        for attribute in required_fields:
            if not hasattr(memory, attribute):
                raise TypeError(
                    f"Memory record is missing '{attribute}'."
                )

        if not isinstance(memory.id, str) or not memory.id.strip():
            raise ValueError("Memory ID cannot be empty.")

        if (
            not isinstance(memory.content, str)
            or not memory.content.strip()
        ):
            raise ValueError("Memory content cannot be empty.")

        if (
            not isinstance(memory.category, str)
            or not memory.category.strip()
        ):
            raise ValueError("Memory category cannot be empty.")

        conversation_id = getattr(
            memory,
            "conversation_id",
            None,
        )

        if conversation_id is not None and not isinstance(
            conversation_id,
            str,
        ):
            raise TypeError(
                "Memory conversation ID must be a string or None."
            )

        created_at = getattr(memory, "created_at", None)

        if created_at is not None and not isinstance(
            created_at,
            datetime,
        ):
            raise TypeError(
                "Memory creation time must be a datetime or None."
            )

        if (
            isinstance(memory.importance, bool)
            or not isinstance(memory.importance, (int, float))
        ):
            raise TypeError("Memory importance must be numeric.")

        importance = float(memory.importance)

        if not math.isfinite(importance):
            raise ValueError("Memory importance must be finite.")

        if not 0.0 <= importance <= 1.0:
            raise ValueError(
                "Memory importance must be between 0.0 and 1.0."
            )

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp a score to the normalized zero-to-one range."""
        return max(0.0, min(float(value), 1.0))
