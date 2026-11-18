
"""Memory retrieval and relevance ranking for S7VEN."""

from __future__ import annotations

import math
from collections.abc import Sequence

from seven.core.conversations.memory_config import (
    MemoryArchitectureConfig,
)
from seven.core.conversations.memory_contracts import (
    MemoryScoringContext,
    MemorySearchMatch,
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

    def _select_accepted_memories(
        self,
        ranked_matches: Sequence[MemorySearchMatch],
        memories_by_id: dict[str, Memory],
        *,
        result_limit: int,
    ) -> list[Memory]:
        """Select ranked memories that satisfy configured thresholds."""
        accepted_memories: list[Memory] = []

        for match in ranked_matches:
            memory = memories_by_id.get(match.memory_id)

            if memory is None:
                continue

            if match.relevance_score <= 0.0:
                continue

            if (
                match.relevance_score
                < self._config.retrieval_min_relevance
            ):
                continue

            if (
                match.combined_score
                < self._config.retrieval_min_combined_score
            ):
                continue

            accepted_memories.append(memory)

            if len(accepted_memories) >= result_limit:
                break

        return accepted_memories

    def _get_accepted_memories(
        self,
        query: str,
        candidates: list[Memory],
        *,
        result_limit: int,
        context: MemoryScoringContext | None,
    ) -> list[Memory]:
        """Score candidates and return accepted memories."""
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

        return self._select_accepted_memories(
            ranked_matches,
            memories_by_id,
            result_limit=result_limit,
        )

    async def _retrieve_adaptively(
        self,
        query: str,
        *,
        result_limit: int,
        context: MemoryScoringContext | None,
    ) -> list[Memory]:
        """Expand retrieval while scoring each unique candidate only once."""
        initial_limit = self._config.candidate_limit(result_limit)

        if self._config.adaptive_retrieval_enabled:
            current_limit = min(
                initial_limit,
                self._config.adaptive_retrieval_max_candidates,
            )
        else:
            current_limit = initial_limit

        memories_by_id: dict[str, Memory] = {}
        matches_by_id: dict[str, MemorySearchMatch] = {}

        while True:
            candidates = await self._memory_service.search(
                query,
                limit=current_limit,
            )

            new_eligible_candidates: list[Memory] = []

            for memory in candidates:
                memory_id = str(memory.id)

                if memory_id in memories_by_id:
                    continue

                memories_by_id[memory_id] = memory

                if memory.importance >= self._min_importance:
                    new_eligible_candidates.append(memory)

            if new_eligible_candidates:
                new_matches = self._scorer.score_many(
                    query,
                    new_eligible_candidates,
                    context=context,
                )

                for match in new_matches:
                    matches_by_id[match.memory_id] = match

            ranked_matches = self._scorer.rank_matches(
                matches_by_id.values(),
            )

            accepted_memories = self._select_accepted_memories(
                ranked_matches,
                memories_by_id,
                result_limit=result_limit,
            )

            if len(accepted_memories) >= result_limit:
                return accepted_memories

            if not self._config.adaptive_retrieval_enabled:
                return accepted_memories

            if len(candidates) < current_limit:
                return accepted_memories

            maximum = self._config.adaptive_retrieval_max_candidates

            if current_limit >= maximum:
                return accepted_memories

            next_limit = min(
                current_limit
                * self._config.adaptive_retrieval_expansion_factor,
                maximum,
            )

            if next_limit <= current_limit:
                return accepted_memories

            current_limit = next_limit

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
        context: MemoryScoringContext | None = None,
    ) -> list[Memory]:
        """Retrieve memories using adaptive relevance ranking."""
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

        return await self._retrieve_adaptively(
            query,
            result_limit=result_limit,
            context=context,
        )

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
