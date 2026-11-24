"""Tests for tool argument validation."""

import pytest

from seven.core.agents.tools import (
    ToolArgumentValidationError,
    ToolArgumentValidator,
)


@pytest.fixture
def validator():
    return ToolArgumentValidator()


def test_accepts_valid_object(validator):
    validator.validate(
        arguments={"query": "Bitcoin"},
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    )


def test_accepts_empty_schema(validator):
    validator.validate(
        arguments={"anything": 123},
        schema={},
    )


def test_rejects_non_mapping_arguments(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate([], {"type": "object"})


def test_rejects_non_mapping_schema(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate({}, [])


def test_rejects_missing_required_property(validator):
    with pytest.raises(ToolArgumentValidationError, match="is required"):
        validator.validate(
            arguments={},
            schema={
                "type": "object",
                "required": ["query"],
            },
        )


def test_rejects_unknown_property_when_disabled(validator):
    with pytest.raises(ToolArgumentValidationError, match="not an allowed"):
        validator.validate(
            arguments={"extra": "value"},
            schema={
                "type": "object",
                "additionalProperties": False,
            },
        )


def test_rejects_wrong_string_type(validator):
    with pytest.raises(ToolArgumentValidationError, match="type 'string'"):
        validator.validate(
            arguments={"query": 123},
            schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
            },
        )


def test_rejects_wrong_boolean_type(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"enabled": 1},
            schema={
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean"},
                },
            },
        )


def test_rejects_wrong_integer_type(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"count": True},
            schema={
                "type": "object",
                "properties": {
                    "count": {"type": "integer"},
                },
            },
        )


def test_rejects_string_below_minimum_length(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"query": "a"},
            schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 3,
                    },
                },
            },
        )


def test_rejects_string_above_maximum_length(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"query": "abcdef"},
            schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "maxLength": 3,
                    },
                },
            },
        )


def test_rejects_number_below_minimum(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"count": 1},
            schema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "number",
                        "minimum": 2,
                    },
                },
            },
        )


def test_rejects_number_above_maximum(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"count": 10},
            schema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "number",
                        "maximum": 5,
                    },
                },
            },
        )


def test_rejects_value_outside_enum(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"mode": "unknown"},
            schema={
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "safe"],
                    },
                },
            },
        )


def test_validates_nested_object(validator):
    validator.validate(
        arguments={
            "location": {
                "city": "Jakarta",
            },
        },
        schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        },
    )


def test_rejects_invalid_nested_object(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={
                "location": {
                    "city": 123,
                },
            },
            schema={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                    },
                },
            },
        )


def test_validates_array_items(validator):
    validator.validate(
        arguments={"symbols": ["BTC", "ETH"]},
        schema={
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    )


def test_rejects_invalid_array_item(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"symbols": ["BTC", 123]},
            schema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        )


def test_rejects_array_below_minimum_items(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"symbols": []},
            schema={
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "minItems": 1,
                    },
                },
            },
        )


def test_rejects_unsupported_schema_type(validator):
    with pytest.raises(ToolArgumentValidationError):
        validator.validate(
            arguments={"value": "test"},
            schema={
                "type": "object",
                "properties": {
                    "value": {"type": "unsupported"},
                },
            },
        )