
"""Build response style instructions from personalization results."""

from __future__ import annotations

from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationDirective,
    PersonalizationResult,
)


class ResponseStylePersonalizer:
    """Convert personalization directives into response style context."""

    _STYLE_CATEGORIES = frozenset(
        {
            PersonalizationCategory.COMMUNICATION_STYLE,
            PersonalizationCategory.RESPONSE_LENGTH,
            PersonalizationCategory.FORMATTING,
            PersonalizationCategory.TECHNICAL_DEPTH,
            PersonalizationCategory.LEARNING_STYLE,
        }
    )

    def __init__(
        self,
        *,
        max_context_chars: int = 1500,
        max_directives: int = 5,
    ) -> None:
        """Initialize response style personalization options."""
        for name, value in (
            ("Maximum context characters", max_context_chars),
            ("Maximum directives", max_directives),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")

            if value < 1:
                raise ValueError(f"{name} must be at least 1.")

        self._max_context_chars = max_context_chars
        self._max_directives = max_directives

    @property
    def max_context_chars(self) -> int:
        """Return the maximum response style context length."""
        return self._max_context_chars

    @property
    def max_directives(self) -> int:
        """Return the maximum number of selected directives."""
        return self._max_directives

    def personalize(
        self,
        result: PersonalizationResult,
    ) -> PersonalizationResult:
        """Build a response style result from personalization directives."""
        if not isinstance(result, PersonalizationResult):
            raise TypeError(
                "Result must be a PersonalizationResult."
            )

        candidates = [
            directive
            for directive in result.directives
            if directive.category in self._STYLE_CATEGORIES
        ]

        selected = self._select_directives(candidates)

        if not selected:
            return PersonalizationResult()

        return PersonalizationResult(
            directives=selected,
            context=self._build_context(selected),
            applied=True,
        )

    def _select_directives(
        self,
        candidates: list[PersonalizationDirective],
    ) -> tuple[PersonalizationDirective, ...]:
        """Select the strongest directive for each style category."""
        selected: dict[
            PersonalizationCategory,
            PersonalizationDirective,
        ] = {}

        for candidate in candidates:
            current = selected.get(candidate.category)

            if current is None:
                selected[candidate.category] = candidate
                continue

            if self._is_stronger(candidate, current):
                selected[candidate.category] = candidate

        ordered = sorted(
            selected.values(),
            key=lambda item: (
                item.priority,
                -item.confidence,
            ),
        )

        return tuple(ordered[: self._max_directives])

    @staticmethod
    def _is_stronger(
        candidate: PersonalizationDirective,
        current: PersonalizationDirective,
    ) -> bool:
        """Compare directives using confidence and priority."""
        if candidate.confidence != current.confidence:
            return candidate.confidence > current.confidence

        return candidate.priority < current.priority

    def _build_context(
        self,
        directives: tuple[PersonalizationDirective, ...],
    ) -> str:
        """Build bounded prompt-ready response style instructions."""
        lines = ["Response style instructions:"]

        for directive in directives:
            lines.append(f"- {directive.instruction}")

        context = "\n".join(lines)

        if len(context) <= self._max_context_chars:
            return context

        if self._max_context_chars <= 3:
            return context[: self._max_context_chars]

        return (
            context[: self._max_context_chars - 3].rstrip()
            + "..."
        )
