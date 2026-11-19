
"""Apply personalization directives based on conversation context."""

from __future__ import annotations

import re

from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationDirective,
    PersonalizationRequest,
    PersonalizationResult,
)


class ContextAwarePersonalizer:
    """Filter personalization directives according to request context."""

    _GLOBAL_CATEGORIES = frozenset(
        {
            PersonalizationCategory.COMMUNICATION_STYLE,
            PersonalizationCategory.RESPONSE_LENGTH,
        }
    )

    _TECHNICAL_TERMS = frozenset(
        {
            "algorithm",
            "api",
            "architecture",
            "automation",
            "backend",
            "code",
            "coding",
            "compiler",
            "computer",
            "css",
            "database",
            "debug",
            "developer",
            "development",
            "docker",
            "framework",
            "function",
            "frontend",
            "html",
            "implementation",
            "javascript",
            "library",
            "machine learning",
            "model",
            "program",
            "programming",
            "python",
            "query",
            "react",
            "script",
            "server",
            "software",
            "sql",
            "system",
            "technical",
            "typescript",
        }
    )

    _LEARNING_TERMS = frozenset(
        {
            "beginner",
            "course",
            "educate",
            "exercise",
            "explain",
            "guide",
            "how do i",
            "how to",
            "learn",
            "learning",
            "lesson",
            "practice",
            "study",
            "teach",
            "theory",
            "tutorial",
            "understand",
            "walk me through",
        }
    )

    _FORMATTING_TERMS = frozenset(
        {
            "advantages",
            "checklist",
            "compare",
            "comparison",
            "differences",
            "difference",
            "disadvantages",
            "features",
            "list",
            "options",
            "pros and cons",
            "requirements",
            "step by step",
            "steps",
            "table",
        }
    )

    _DOMAIN_STOP_WORDS = frozenset(
        {
            "a",
            "an",
            "and",
            "answering",
            "as",
            "context",
            "domain",
            "for",
            "in",
            "is",
            "it",
            "of",
            "on",
            "related",
            "relevant",
            "the",
            "to",
            "use",
            "when",
            "with",
            "questions",
        }
    )

    def personalize(
        self,
        request: PersonalizationRequest,
        result: PersonalizationResult,
    ) -> PersonalizationResult:
        """Select directives that are relevant to the current context."""
        if not isinstance(request, PersonalizationRequest):
            raise TypeError(
                "Request must be a PersonalizationRequest."
            )

        if not isinstance(result, PersonalizationResult):
            raise TypeError(
                "Result must be a PersonalizationResult."
            )

        if not result.directives:
            return PersonalizationResult()

        context = self._build_context_source(request)

        selected = tuple(
            directive
            for directive in result.directives
            if self._is_relevant(directive, context)
        )

        if not selected:
            return PersonalizationResult()

        return PersonalizationResult(
            directives=selected,
            context=self._build_context(selected),
            applied=True,
        )

    @staticmethod
    def _build_context_source(
        request: PersonalizationRequest,
    ) -> str:
        """Combine available conversation context for relevance checks."""
        return " ".join(
            (
                request.user_message,
                request.conversation_context,
                request.memory_context,
            )
        ).casefold()

    def _is_relevant(
        self,
        directive: PersonalizationDirective,
        context: str,
    ) -> bool:
        """Determine whether a directive applies to the current context."""
        category = directive.category

        if category in self._GLOBAL_CATEGORIES:
            return True

        if category is PersonalizationCategory.TECHNICAL_DEPTH:
            return self._matches_terms(
                context,
                self._TECHNICAL_TERMS,
            )

        if category is PersonalizationCategory.LEARNING_STYLE:
            return self._matches_terms(
                context,
                self._LEARNING_TERMS,
            )

        if category is PersonalizationCategory.FORMATTING:
            return self._matches_terms(
                context,
                self._FORMATTING_TERMS,
            )

        if category is PersonalizationCategory.DOMAIN_CONTEXT:
            return self._matches_domain(directive, context)

        return False

    @classmethod
    def _matches_terms(
        cls,
        context: str,
        terms: frozenset[str],
    ) -> bool:
        """Check whether context contains one of the configured terms."""
        normalized_context = cls._normalize_text(context)
        context_tokens = set(cls._tokenize(context))

        for term in terms:
            normalized_term = cls._normalize_text(term)

            if " " in normalized_term:
                if normalized_term in normalized_context:
                    return True
            elif normalized_term in context_tokens:
                return True

        return False

    @classmethod
    def _matches_domain(
        cls,
        directive: PersonalizationDirective,
        context: str,
    ) -> bool:
        """Check whether a domain directive matches conversation topics."""
        directive_tokens = set(
            cls._tokenize(directive.instruction)
        )

        domain_tokens = (
            directive_tokens - cls._DOMAIN_STOP_WORDS
        )

        if not domain_tokens:
            return False

        context_tokens = set(cls._tokenize(context))

        return bool(domain_tokens & context_tokens)

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize text for phrase matching."""
        return " ".join(
            ContextAwarePersonalizer._tokenize(value)
        )

    @staticmethod
    def _tokenize(value: str) -> tuple[str, ...]:
        """Extract normalized alphanumeric tokens."""
        return tuple(re.findall(r"[a-z0-9]+", value.casefold()))

    @staticmethod
    def _build_context(
        directives: tuple[PersonalizationDirective, ...],
    ) -> str:
        """Build prompt-ready context from selected directives."""
        if not directives:
            return ""

        lines = ["Context-aware personalization instructions:"]

        for directive in directives:
            lines.append(f"- {directive.instruction}")

        return "\n".join(lines)
