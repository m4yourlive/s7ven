"""Memory retrieval and relevance ranking for S7VEN."""

from __future__ import annotations

import math
from collections.abc import Sequence

from seven.core.conversations.memory_config import (
    MemoryArchitectureConfig,
)
from seven.core.conversations.memory_contracts import (
    MemoryScoringContext,
)
from seven.core.conversations.memory_scorer import MemoryScorer
from seven.core.conversations.memory_service import MemoryService
from seven.core.database.models import Memory


class MemoryRetriever:
    """Retrieve and rank relevant long-term memories."""

    def __init__(
        self,
        memory_service: MemoryService,
        *,
        max_results: int = 5,
        min_importance: float | None = None,
        scorer: MemoryScorer | None = None,
        config: MemoryArchitectureConfig | None = None,
    ) -> None:
        if not isinstance(memory_service, MemoryService):
            raise TypeError(
                "memory_service must be a MemoryService instance."
            )

        if (
            isinstance(max_results, bool)
            or not isinstance(max_results, int)
            or max_results < 1
        ):
            raise ValueError(
                "max_results must be a positive integer."
            )

        if config is not None and not isinstance(
            config,
            MemoryArchitectureConfig,
        ):
            raise TypeError(
                "config must be a MemoryArchitectureConfig."
            )

        if scorer is not None and not isinstance(
            scorer,
            MemoryScorer,
        ):
            raise TypeError(
                "scorer must be a MemoryScorer instance."
            )

        if scorer is not None and config is not None:
            raise ValueError(
                "Provide either scorer or config, not both."
            )

        selected_config = config or MemoryArchitectureConfig()

        selected_importance = (
            selected_config.retrieval_min_importance
            if min_importance is None
            else min_importance
        )

        if (
            isinstance(selected_importance, bool)
            or not isinstance(selected_importance, (int, float))
        ):
            raise ValueError(
                "min_importance must be between 0.0 and 1.0."
            )

        selected_importance = float(selected_importance)

        if (
            not math.isfinite(selected_importance)
            or not 0.0 <= selected_importance <= 1.0
        ):
            raise ValueError(
                "min_importance must be between 0.0 and 1.0."
            )

        self._memory_service = memory_service
        self._max_results = max_results
        self._min_importance = selected_importance
        self._config = selected_config
        self._scorer = scorer or MemoryScorer(selected_config)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Normalize text into searchable tokens."""
        return set(MemoryScorer.tokenize(text))

    @classmethod
    def _relevance_score(
        cls,
        query_tokens: set[str],
        content: str,
    ) -> float:
        """Calculate keyword overlap between query and memory content."""
        if not query_tokens:
            return 0.0

        memory_tokens = cls._tokenize(content)

        if not memory_tokens:
            return 0.0

        return len(query_tokens & memory_tokens) / len(query_tokens)

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        context: MemoryScoringContext | None = None,
    ) -> list[Memory]:
        """Retrieve memories using calibrated relevance thresholds."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        result_limit = (
            self._max_results
            if limit is None
            else limit
        )

        if (
            isinstance(result_limit, bool)
            or not isinstance(result_limit, int)
            or result_limit < 1
        ):
            raise ValueError(
                "limit must be a positive integer."
            )

        if context is not None and not isinstance(
            context,
            MemoryScoringContext,
        ):
            raise TypeError(
                "context must be a MemoryScoringContext."
            )

        if not self._tokenize(query):
            return []

        candidates = await self._memory_service.search(
            query,
            limit=self._config.candidate_limit(result_limit),
        )

        eligible_candidates = [
            memory
            for memory in candidates
            if memory.importance >= self._min_importance
        ]

        if not eligible_candidates:
            return []

        ranked_matches = self._scorer.rank(
            query,
            eligible_candidates,
            context=context,
        )

        memories_by_id = {
            str(memory.id): memory
            for memory in eligible_candidates
        }

        accepted_matches = [
            match
            for match in ranked_matches
            if (
                match.memory_id in memories_by_id
                and match.relevance_score > 0.0
                and (
                    match.relevance_score
                    >= self._config.retrieval_min_relevance
                )
                and (
                    match.combined_score
                    >= self._config.retrieval_min_combined_score
                )
            )
        ]

        return [
            memories_by_id[match.memory_id]
            for match in accepted_matches[:result_limit]
        ]

    async def retrieve_by_tokens(
        self,
        tokens: Sequence[str],
        *,
        limit: int | None = None,
        context: MemoryScoringContext | None = None,
    ) -> list[Memory]:
        """Retrieve memories using a sequence of keywords."""
        if isinstance(tokens, str) or not isinstance(tokens, Sequence):
            raise TypeError(
                "Tokens must be a sequence of strings."
            )

        if not tokens:
            raise ValueError("Tokens cannot be empty.")

        normalized_tokens = [
            token.strip()
            for token in tokens
            if isinstance(token, str) and token.strip()
        ]

        if not normalized_tokens:
            raise ValueError(
                "At least one valid token is required."
            )

        return await self.retrieve(
            " ".join(normalized_tokens),
            limit=limit,
            context=context,
        )