
"""Memory retrieval and relevance ranking for S7VEN."""

import re
from collections.abc import Sequence

from seven.core.database.models import Memory
from seven.core.conversations.memory_service import MemoryService


class MemoryRetriever:
    """Retrieve and rank relevant long-term memories."""

    def __init__(
        self,
        memory_service: MemoryService,
        *,
        max_results: int = 5,
        min_importance: float = 0.0,
    ) -> None:
        if not isinstance(memory_service, MemoryService):
            raise TypeError(
                "memory_service must be a MemoryService instance."
            )

        if (
            not isinstance(max_results, int)
            or isinstance(max_results, bool)
            or max_results < 1
        ):
            raise ValueError("max_results must be a positive integer.")

        if (
            not isinstance(min_importance, (int, float))
            or isinstance(min_importance, bool)
            or not 0.0 <= min_importance <= 1.0
        ):
            raise ValueError(
                "min_importance must be between 0.0 and 1.0."
            )

        self._memory_service = memory_service
        self._max_results = max_results
        self._min_importance = float(min_importance)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Normalize text into searchable tokens."""
        return {
            token.lower()
            for token in re.findall(r"\b\w+\b", text)
            if len(token) > 1
        }

    @classmethod
    def _relevance_score(
        cls,
        query_tokens: set[str],
        content: str,
    ) -> float:
        """Calculate keyword overlap between query and memory."""
        memory_tokens = cls._tokenize(content)

        if not query_tokens or not memory_tokens:
            return 0.0

        overlap = query_tokens & memory_tokens

        return len(overlap) / len(query_tokens)

    async def retrieve(
        self,
        query: str,
        *,
        limit: int | None = None,
    ) -> list[Memory]:
        """Retrieve memories relevant to a query."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query cannot be empty.")

        result_limit = self._max_results if limit is None else limit

        if (
            not isinstance(result_limit, int)
            or isinstance(result_limit, bool)
            or result_limit < 1
        ):
            raise ValueError("limit must be a positive integer.")

        query_tokens = self._tokenize(query)

        if not query_tokens:
            return []

        candidates = await self._memory_service.search(
            query,
            limit=max(result_limit * 5, 20),
        )

        ranked: list[tuple[float, float, Memory]] = []

        for memory in candidates:
            if memory.importance < self._min_importance:
                continue

            relevance = self._relevance_score(
                query_tokens,
                memory.content,
            )

            if relevance <= 0:
                continue

            combined_score = (
                relevance * 0.7
                + memory.importance * 0.3
            )

            ranked.append(
                (combined_score, relevance, memory)
            )

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2].created_at,
            ),
            reverse=True,
        )

        return [
            memory
            for _, _, memory in ranked[:result_limit]
        ]

    async def retrieve_by_tokens(
        self,
        tokens: Sequence[str],
        *,
        limit: int | None = None,
    ) -> list[Memory]:
        """Retrieve memories using a sequence of keywords."""
        if not tokens:
            raise ValueError("Tokens cannot be empty.")

        normalized_tokens = [
            token.strip()
            for token in tokens
            if isinstance(token, str) and token.strip()
        ]

        if not normalized_tokens:
            raise ValueError("At least one valid token is required.")

        return await self.retrieve(
            " ".join(normalized_tokens),
            limit=limit,
        )
