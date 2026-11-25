
"""Configuration models for the autonomous reasoning loop."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReasoningConfig:
    """Configure execution limits for the reasoning loop."""

    max_iterations: int = 10
    max_tool_calls: int = 25
    max_consecutive_failures: int = 3
    timeout_seconds: float = 120.0
    max_tool_calls_per_iteration: int = 5

    def __post_init__(self) -> None:
        integer_fields = (
            "max_iterations",
            "max_tool_calls",
            "max_consecutive_failures",
            "max_tool_calls_per_iteration",
        )

        for field_name in integer_fields:
            value = getattr(self, field_name)

            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{field_name} must be an integer.")

            if value < 1:
                raise ValueError(
                    f"{field_name} must be greater than zero."
                )

        if (
            not isinstance(self.timeout_seconds, (int, float))
            or isinstance(self.timeout_seconds, bool)
        ):
            raise TypeError("timeout_seconds must be a number.")

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )
