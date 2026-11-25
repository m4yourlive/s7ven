
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Application:
    """Represent an installed desktop application."""

    name: str
    executable: Path
    arguments: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        normalized_name = self.name.strip()

        if not normalized_name:
            raise ValueError("Application name cannot be empty.")

        if not isinstance(self.executable, Path):
            raise TypeError("Application executable must be a Path.")

        if not self.executable.name:
            raise ValueError("Application executable cannot be empty.")

        if not all(
            isinstance(argument, str)
            for argument in self.arguments
        ):
            raise TypeError("Application arguments must contain strings.")

        if self.description is not None and not isinstance(
            self.description,
            str,
        ):
            raise TypeError("Application description must be a string.")

        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(
            self,
            "executable",
            self.executable.expanduser(),
        )
        object.__setattr__(
            self,
            "arguments",
            tuple(self.arguments),
        )
