"""JSON Schema subset validator for tool arguments."""

from collections.abc import Mapping
from typing import Any

from seven.core.agents.tools.exceptions import ToolArgumentValidationError


class ToolArgumentValidator:
    """Validate tool arguments against their parameter schema."""

    def validate(
        self,
        arguments: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> None:
        """Raise an error when arguments violate the supplied schema."""
        if not isinstance(arguments, Mapping):
            raise ToolArgumentValidationError(
                "Tool arguments must be a mapping."
            )

        if not isinstance(schema, Mapping):
            raise ToolArgumentValidationError(
                "Tool parameter schema must be a mapping."
            )

        if not schema:
            return

        self._validate_value(
            value=arguments,
            schema=schema,
            path="arguments",
        )

    def _validate_value(
        self,
        value: Any,
        schema: Mapping[str, Any],
        path: str,
    ) -> None:
        expected_type = schema.get("type")

        if expected_type is not None:
            if not self._matches_type(value, expected_type):
                raise ToolArgumentValidationError(
                    f"{path} must have type '{expected_type}'."
                )

        if "enum" in schema and value not in schema["enum"]:
            raise ToolArgumentValidationError(
                f"{path} must be one of the allowed values."
            )

        if isinstance(value, Mapping):
            self._validate_object(value, schema, path)

        elif isinstance(value, str):
            self._validate_string(value, schema, path)

        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self._validate_number(value, schema, path)

        elif isinstance(value, list):
            self._validate_array(value, schema, path)

    @staticmethod
    def _matches_type(value: Any, expected_type: str) -> bool:
        """Check a value against a supported JSON Schema type."""
        type_checks = {
            "object": lambda item: isinstance(item, Mapping),
            "string": lambda item: isinstance(item, str),
            "integer": lambda item: isinstance(item, int)
            and not isinstance(item, bool),
            "number": lambda item: isinstance(item, (int, float))
            and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "array": lambda item: isinstance(item, list),
            "null": lambda item: item is None,
        }

        if not isinstance(expected_type, str):
            return False

        checker = type_checks.get(expected_type)
        if checker is None:
            raise ToolArgumentValidationError(
                f"Unsupported schema type '{expected_type}'."
            )

        return checker(value)

    def _validate_object(
        self,
        value: Mapping[str, Any],
        schema: Mapping[str, Any],
        path: str,
    ) -> None:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        additional_properties = schema.get("additionalProperties", True)

        if not isinstance(properties, Mapping):
            raise ToolArgumentValidationError(
                f"{path} schema properties must be a mapping."
            )

        if not isinstance(required, list) or not all(
            isinstance(item, str) for item in required
        ):
            raise ToolArgumentValidationError(
                f"{path} schema required must be a list of strings."
            )

        for key in required:
            if key not in value:
                raise ToolArgumentValidationError(
                    f"{path}.{key} is required."
                )

        for key, item in value.items():
            property_schema = properties.get(key)

            if property_schema is None:
                if additional_properties is False:
                    raise ToolArgumentValidationError(
                        f"{path}.{key} is not an allowed property."
                    )
                continue

            if not isinstance(property_schema, Mapping):
                raise ToolArgumentValidationError(
                    f"Schema for {path}.{key} must be a mapping."
                )

            self._validate_value(
                value=item,
                schema=property_schema,
                path=f"{path}.{key}",
            )

    @staticmethod
    def _validate_string(
        value: str,
        schema: Mapping[str, Any],
        path: str,
    ) -> None:
        min_length = schema.get("minLength")
        max_length = schema.get("maxLength")

        if min_length is not None and len(value) < min_length:
            raise ToolArgumentValidationError(
                f"{path} must contain at least {min_length} characters."
            )

        if max_length is not None and len(value) > max_length:
            raise ToolArgumentValidationError(
                f"{path} must contain at most {max_length} characters."
            )

    @staticmethod
    def _validate_number(
        value: int | float,
        schema: Mapping[str, Any],
        path: str,
    ) -> None:
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")

        if minimum is not None and value < minimum:
            raise ToolArgumentValidationError(
                f"{path} must be greater than or equal to {minimum}."
            )

        if maximum is not None and value > maximum:
            raise ToolArgumentValidationError(
                f"{path} must be less than or equal to {maximum}."
            )

    def _validate_array(
        self,
        value: list[Any],
        schema: Mapping[str, Any],
        path: str,
    ) -> None:
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        item_schema = schema.get("items")

        if min_items is not None and len(value) < min_items:
            raise ToolArgumentValidationError(
                f"{path} must contain at least {min_items} items."
            )

        if max_items is not None and len(value) > max_items:
            raise ToolArgumentValidationError(
                f"{path} must contain at most {max_items} items."
            )

        if item_schema is not None:
            if not isinstance(item_schema, Mapping):
                raise ToolArgumentValidationError(
                    f"{path} schema items must be a mapping."
                )

            for index, item in enumerate(value):
                self._validate_value(
                    value=item,
                    schema=item_schema,
                    path=f"{path}[{index}]",
                )