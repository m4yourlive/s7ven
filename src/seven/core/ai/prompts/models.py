"""Data models for S7VEN prompt management."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """Represent a reusable AI prompt template."""

    name: str
    content: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Prompt name cannot be empty.")

        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("Prompt content cannot be empty.")

        if not isinstance(self.description, str):
            raise TypeError("Prompt description must be a string.")

    def render(self, **variables: str) -> str:
        """Render the prompt using the supplied variables."""
        try:
            return self.content.format(**variables)
        except KeyError as exc:
            raise ValueError(
                f"Missing prompt variable: {exc.args[0]}"
            ) from exc
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid prompt template.") from exc
