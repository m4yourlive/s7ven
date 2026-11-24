
"""Tests for the generation request factory."""

from unittest.mock import Mock

import pytest

from seven.core.ai.models import AIMessage, GenerationRequest, MessageRole
from seven.core.ai.prompts.prompt_manager import PromptManager
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.config.config_schema import AIConfig


@pytest.fixture
def prompt_manager() -> Mock:
    manager = Mock(spec=PromptManager)
    manager.render.return_value = "You are S7VEN."
    return manager


@pytest.fixture
def messages() -> tuple[AIMessage, ...]:
    return (
        AIMessage(
            role=MessageRole.USER,
            content="Hello, S7VEN.",
        ),
    )


@pytest.fixture
def api_config() -> AIConfig:
    return AIConfig(
        mode="api",
        provider="openai",
        model="gpt-4o-mini",
        local_model="llama3.2",
    )


@pytest.fixture
def local_config() -> AIConfig:
    return AIConfig(
        mode="local",
        provider="openai",
        model="gpt-4o-mini",
        local_model="llama3.2",
    )


def test_init_rejects_invalid_config(prompt_manager: Mock) -> None:
    with pytest.raises(TypeError, match="config must be an AIConfig"):
        GenerationRequestFactory(
            config=Mock(),
            prompt_manager=prompt_manager,
        )


def test_init_rejects_invalid_prompt_manager(api_config: AIConfig) -> None:
    with pytest.raises(
        TypeError,
        match="prompt_manager must be a PromptManager",
    ):
        GenerationRequestFactory(
            config=api_config,
            prompt_manager=Mock(),
        )


def test_build_uses_api_model(
    api_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    request = factory.build(messages)

    assert isinstance(request, GenerationRequest)
    assert request.model == "gpt-4o-mini"


def test_build_uses_local_model(
    local_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(local_config, prompt_manager)

    request = factory.build(messages)

    assert request.model == "llama3.2"


def test_build_explicit_model_overrides_configured_model(
    local_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(local_config, prompt_manager)

    request = factory.build(messages, model="qwen2.5")

    assert request.model == "qwen2.5"


def test_build_includes_system_message(
    api_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    request = factory.build(messages)

    assert request.messages[0] == AIMessage(
        role=MessageRole.SYSTEM,
        content="You are S7VEN.",
    )
    assert request.messages[1:] == messages
    prompt_manager.render.assert_called_once_with("seven_system")


def test_build_uses_configured_generation_parameters(
    api_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    request = factory.build(messages)

    assert request.temperature == api_config.temperature
    assert request.max_tokens == api_config.max_tokens
    assert request.top_p == api_config.top_p
    assert request.frequency_penalty == api_config.frequency_penalty
    assert request.presence_penalty == api_config.presence_penalty


def test_build_supports_generation_parameter_overrides(
    api_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    request = factory.build(
        messages,
        temperature=0.3,
        max_tokens=256,
        metadata={"source": "test"},
    )

    assert request.temperature == 0.3
    assert request.max_tokens == 256
    assert request.metadata == {"source": "test"}


def test_build_defaults_metadata_to_empty_dict(
    api_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    request = factory.build(messages)

    assert request.metadata == {}


def test_build_preserves_explicit_empty_metadata(
    api_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    request = factory.build(messages, metadata={})

    assert request.metadata == {}


def test_config_property_returns_active_configuration(
    api_config: AIConfig,
    prompt_manager: Mock,
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    assert factory.config is api_config


def test_update_config_replaces_active_configuration(
    api_config: AIConfig,
    local_config: AIConfig,
    prompt_manager: Mock,
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    factory.update_config(local_config)

    assert factory.config is local_config


def test_update_config_rejects_invalid_configuration(
    api_config: AIConfig,
    prompt_manager: Mock,
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    with pytest.raises(TypeError, match="config must be an AIConfig"):
        factory.update_config(Mock())

    assert factory.config is api_config


def test_update_config_switches_api_model_to_local_model(
    api_config: AIConfig,
    local_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    api_request = factory.build(messages)

    factory.update_config(local_config)

    local_request = factory.build(messages)

    assert api_request.model == "gpt-4o-mini"
    assert local_request.model == "llama3.2"


def test_update_config_switches_local_model_to_api_model(
    api_config: AIConfig,
    local_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(local_config, prompt_manager)

    local_request = factory.build(messages)

    factory.update_config(api_config)

    api_request = factory.build(messages)

    assert local_request.model == "llama3.2"
    assert api_request.model == "gpt-4o-mini"


def test_update_config_applies_new_generation_parameters(
    api_config: AIConfig,
    prompt_manager: Mock,
    messages: tuple[AIMessage, ...],
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    updated_config = api_config.model_copy(
        update={
            "temperature": 0.25,
            "max_tokens": 512,
            "top_p": 0.8,
            "frequency_penalty": 0.4,
            "presence_penalty": 0.2,
        }
    )

    factory.update_config(updated_config)

    request = factory.build(messages)

    assert request.temperature == 0.25
    assert request.max_tokens == 512
    assert request.top_p == 0.8
    assert request.frequency_penalty == 0.4
    assert request.presence_penalty == 0.2


def test_update_config_preserves_factory_instance(
    api_config: AIConfig,
    local_config: AIConfig,
    prompt_manager: Mock,
) -> None:
    factory = GenerationRequestFactory(api_config, prompt_manager)

    original_id = id(factory)

    factory.update_config(local_config)

    assert id(factory) == original_id
