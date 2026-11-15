"""Central manager for S7VEN prompt templates."""

from seven.core.ai.prompts.models import PromptTemplate


class PromptManager:
    """Register, retrieve, update, and remove prompt templates."""

    def __init__(
        self,
        templates: tuple[PromptTemplate, ...] = (),
    ) -> None:
        self._templates: dict[str, PromptTemplate] = {}

        for template in templates:
            self.register(template)

    @property
    def templates(self) -> tuple[PromptTemplate, ...]:
        """Return all registered prompt templates."""
        return tuple(self._templates.values())

    def register(self, template: PromptTemplate) -> None:
        """Register a new prompt template."""
        if not isinstance(template, PromptTemplate):
            raise TypeError(
                "template must be a PromptTemplate instance."
            )

        key = template.name.strip()

        if key in self._templates:
            raise ValueError(
                f"Prompt template '{key}' is already registered."
            )

        self._templates[key] = template

    def get(self, name: str) -> PromptTemplate:
        """Retrieve a prompt template by name."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Prompt name cannot be empty.")

        try:
            return self._templates[name.strip()]
        except KeyError as exc:
            raise KeyError(
                f"Prompt template '{name}' was not found."
            ) from exc

    def render(
        self,
        name: str,
        **variables: str,
    ) -> str:
        """Render a registered prompt template."""
        return self.get(name).render(**variables)

    def update(self, template: PromptTemplate) -> None:
        """Replace an existing prompt template."""
        if not isinstance(template, PromptTemplate):
            raise TypeError(
                "template must be a PromptTemplate instance."
            )

        key = template.name.strip()

        if key not in self._templates:
            raise KeyError(
                f"Prompt template '{key}' was not found."
            )

        self._templates[key] = template

    def remove(self, name: str) -> None:
        """Remove a registered prompt template."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Prompt name cannot be empty.")

        try:
            del self._templates[name.strip()]
        except KeyError as exc:
            raise KeyError(
                f"Prompt template '{name}' was not found."
            ) from exc

    def exists(self, name: str) -> bool:
        """Check whether a prompt template exists."""
        return isinstance(name, str) and name.strip() in self._templates
