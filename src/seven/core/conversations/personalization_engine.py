
"""Rule-based personalization engine."""

from __future__ import annotations

from seven.core.conversations.personalization_contracts import (
    PersonalizationCategory,
    PersonalizationDirective,
    PersonalizationRequest,
    PersonalizationResult,
)


class PersonalizationEngine:
    """Convert recognized user preferences into actionable directives."""

    _RULES: dict[
        str,
        tuple[PersonalizationCategory, dict[str, str]],
    ] = {
        "communication": (
            PersonalizationCategory.COMMUNICATION_STYLE,
            {
                "casual": "Use a casual and natural communication style.",
                "friendly": "Use a friendly and approachable communication style.",
                "formal": "Use a formal and respectful communication style.",
                "professional": "Use a professional communication style.",
                "direct": "Communicate directly and avoid unnecessary wording.",
                "concise": "Keep communication concise and focused.",
                "conversational": "Use a natural conversational communication style.",
            },
        ),
        "communication_style": (
            PersonalizationCategory.COMMUNICATION_STYLE,
            {
                "casual": "Use a casual and natural communication style.",
                "friendly": "Use a friendly and approachable communication style.",
                "formal": "Use a formal and respectful communication style.",
                "professional": "Use a professional communication style.",
                "direct": "Communicate directly and avoid unnecessary wording.",
                "concise": "Keep communication concise and focused.",
                "conversational": "Use a natural conversational communication style.",
            },
        ),
        "response_style": (
            PersonalizationCategory.COMMUNICATION_STYLE,
            {
                "casual": "Use a casual and natural communication style.",
                "friendly": "Use a friendly and approachable communication style.",
                "formal": "Use a formal and respectful communication style.",
                "professional": "Use a professional communication style.",
                "direct": "Communicate directly and avoid unnecessary wording.",
                "concise": "Keep communication concise and focused.",
                "conversational": "Use a natural conversational communication style.",
            },
        ),
        "tone": (
            PersonalizationCategory.COMMUNICATION_STYLE,
            {
                "casual": "Use a casual and natural communication style.",
                "friendly": "Use a friendly and approachable communication style.",
                "formal": "Use a formal and respectful communication style.",
                "professional": "Use a professional communication style.",
                "direct": "Communicate directly and avoid unnecessary wording.",
            },
        ),
        "response_length": (
            PersonalizationCategory.RESPONSE_LENGTH,
            {
                "short": "Keep responses short and focused.",
                "brief": "Keep responses brief and focused.",
                "concise": "Keep responses concise and avoid unnecessary detail.",
                "detailed": "Provide detailed explanations with relevant context.",
                "long": "Provide thorough explanations when appropriate.",
                "thorough": "Provide thorough explanations with sufficient detail.",
            },
        ),
        "answer_length": (
            PersonalizationCategory.RESPONSE_LENGTH,
            {
                "short": "Keep responses short and focused.",
                "brief": "Keep responses brief and focused.",
                "concise": "Keep responses concise and avoid unnecessary detail.",
                "detailed": "Provide detailed explanations with relevant context.",
                "long": "Provide thorough explanations when appropriate.",
                "thorough": "Provide thorough explanations with sufficient detail.",
            },
        ),
        "verbosity": (
            PersonalizationCategory.RESPONSE_LENGTH,
            {
                "low": "Keep responses concise and focused.",
                "minimal": "Keep responses concise and focused.",
                "medium": "Use a balanced level of detail.",
                "high": "Provide detailed explanations with relevant context.",
            },
        ),
        "formatting": (
            PersonalizationCategory.FORMATTING,
            {
                "bullet_points": "Prefer bullet points when presenting multiple items.",
                "bullets": "Prefer bullet points when presenting multiple items.",
                "numbered_list": "Use numbered lists for ordered instructions.",
                "step_by_step": "Explain procedures in clear sequential steps.",
                "tables": "Use tables when comparing structured information.",
                "paragraphs": "Prefer clear, organized paragraphs.",
            },
        ),
        "response_format": (
            PersonalizationCategory.FORMATTING,
            {
                "bullet_points": "Prefer bullet points when presenting multiple items.",
                "bullets": "Prefer bullet points when presenting multiple items.",
                "numbered_list": "Use numbered lists for ordered instructions.",
                "step_by_step": "Explain procedures in clear sequential steps.",
                "tables": "Use tables when comparing structured information.",
                "paragraphs": "Prefer clear, organized paragraphs.",
            },
        ),
        "technical_depth": (
            PersonalizationCategory.TECHNICAL_DEPTH,
            {
                "beginner": "Explain technical concepts from the fundamentals and define unfamiliar terms.",
                "basic": "Use accessible explanations and introduce technical concepts gradually.",
                "intermediate": "Assume foundational knowledge and explain important technical details.",
                "advanced": "Use advanced technical explanations and include implementation details when relevant.",
                "expert": "Use precise technical language and focus on advanced details.",
            },
        ),
        "expertise_level": (
            PersonalizationCategory.TECHNICAL_DEPTH,
            {
                "beginner": "Explain technical concepts from the fundamentals and define unfamiliar terms.",
                "basic": "Use accessible explanations and introduce technical concepts gradually.",
                "intermediate": "Assume foundational knowledge and explain important technical details.",
                "advanced": "Use advanced technical explanations and include implementation details when relevant.",
                "expert": "Use precise technical language and focus on advanced details.",
            },
        ),
        "learning_style": (
            PersonalizationCategory.LEARNING_STYLE,
            {
                "practical": "Favor practical explanations and actionable exercises.",
                "hands_on": "Include hands-on exercises and practical applications when useful.",
                "examples": "Use concrete examples to explain unfamiliar concepts.",
                "visual": "Use visual descriptions, diagrams, or structured layouts when useful.",
                "theoretical": "Explain underlying concepts and theory before practical applications.",
            },
        ),
        "learning_preference": (
            PersonalizationCategory.LEARNING_STYLE,
            {
                "practical": "Favor practical explanations and actionable exercises.",
                "hands_on": "Include hands-on exercises and practical applications when useful.",
                "examples": "Use concrete examples to explain unfamiliar concepts.",
                "visual": "Use visual descriptions, diagrams, or structured layouts when useful.",
                "theoretical": "Explain underlying concepts and theory before practical applications.",
            },
        ),
    }

    _DOMAIN_KEYS = {
        "domain",
        "domain_context",
        "interests",
        "preferred_topics",
        "topics",
    }

    _CATEGORY_PRIORITY = {
        PersonalizationCategory.COMMUNICATION_STYLE: 10,
        PersonalizationCategory.RESPONSE_LENGTH: 20,
        PersonalizationCategory.FORMATTING: 30,
        PersonalizationCategory.TECHNICAL_DEPTH: 40,
        PersonalizationCategory.LEARNING_STYLE: 50,
        PersonalizationCategory.DOMAIN_CONTEXT: 60,
    }

    def personalize(
        self,
        request: PersonalizationRequest,
    ) -> PersonalizationResult:
        """Generate personalization directives from a structured profile."""
        if not isinstance(request, PersonalizationRequest):
            raise TypeError(
                "Request must be a PersonalizationRequest."
            )

        if request.profile is None:
            return PersonalizationResult()

        candidates: list[PersonalizationDirective] = []

        for preference in request.profile.preferences:
            if preference.confidence < request.min_confidence:
                continue

            directive = self._build_directive(
                key=preference.key,
                value=preference.value,
                confidence=preference.confidence,
            )

            if directive is not None:
                candidates.append(directive)

        selected = self._select_directives(
            candidates,
            max_directives=request.max_directives,
        )

        if not selected:
            return PersonalizationResult()

        return PersonalizationResult(
            directives=selected,
            context=self._build_context(selected),
            applied=True,
        )

    def _build_directive(
        self,
        *,
        key: str,
        value: str | bool | int | float,
        confidence: float,
    ) -> PersonalizationDirective | None:
        """Resolve a profile preference into a directive."""
        normalized_key = key.strip().casefold()
        normalized_value = self._normalize_value(value)

        if normalized_key in self._DOMAIN_KEYS:
            instruction = self._build_domain_instruction(value)

            if instruction is None:
                return None

            return PersonalizationDirective(
                category=PersonalizationCategory.DOMAIN_CONTEXT,
                instruction=instruction,
                confidence=confidence,
                priority=self._CATEGORY_PRIORITY[
                    PersonalizationCategory.DOMAIN_CONTEXT
                ],
                source="profile",
                key=normalized_key,
            )

        rule = self._RULES.get(normalized_key)

        if rule is None:
            return None

        category, value_rules = rule
        instruction = self._resolve_value(
            normalized_value,
            value_rules,
        )

        if instruction is None:
            return None

        return PersonalizationDirective(
            category=category,
            instruction=instruction,
            confidence=confidence,
            priority=self._CATEGORY_PRIORITY[category],
            source="profile",
            key=normalized_key,
        )

    @staticmethod
    def _normalize_value(value: str | bool | int | float) -> str:
        """Normalize preference values for rule matching."""
        if isinstance(value, bool):
            return "true" if value else "false"

        return (
            str(value)
            .strip()
            .casefold()
            .replace("-", "_")
            .replace(" ", "_")
        )

    @staticmethod
    def _resolve_value(
        value: str,
        rules: dict[str, str],
    ) -> str | None:
        """Resolve exact or phrase-based preference values."""
        if value in rules:
            return rules[value]

        tokens = value.split("_")

        for rule_key, instruction in rules.items():
            rule_tokens = rule_key.split("_")

            if len(rule_tokens) == 1 and rule_key in tokens:
                return instruction

            if len(rule_tokens) > 1 and rule_key in value:
                return instruction

        return None

    @staticmethod
    def _build_domain_instruction(
        value: str | bool | int | float,
    ) -> str | None:
        """Build a domain-context instruction from a profile value."""
        if isinstance(value, bool):
            return None

        normalized = str(value).strip()

        if not normalized:
            return None

        return (
            f"Use {normalized} as relevant domain context "
            "when answering related questions."
        )

    def _select_directives(
        self,
        candidates: list[PersonalizationDirective],
        *,
        max_directives: int,
    ) -> tuple[PersonalizationDirective, ...]:
        """Select the strongest directive for each category."""
        selected: dict[
            PersonalizationCategory,
            PersonalizationDirective,
        ] = {}

        for candidate in candidates:
            current = selected.get(candidate.category)

            if current is None:
                selected[candidate.category] = candidate
                continue

            if candidate.confidence > current.confidence:
                selected[candidate.category] = candidate

        ordered = sorted(
            selected.values(),
            key=lambda item: (
                item.priority,
                -item.confidence,
                item.key or "",
            ),
        )

        return tuple(ordered[:max_directives])

    @staticmethod
    def _build_context(
        directives: tuple[PersonalizationDirective, ...],
    ) -> str:
        """Format selected directives into prompt-ready context."""
        if not directives:
            return ""

        lines = ["Personalization instructions:"]

        for directive in directives:
            lines.append(f"- {directive.instruction}")

        return "\n".join(lines)
