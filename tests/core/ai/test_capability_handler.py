
import pytest

from seven.core.ai.exceptions import AIProviderCapabilityError
from seven.core.ai.models import (
    AIMessage,
    GenerationRequest,
    MessageRole,
    ProviderCapabilities,
)
from seven.core.ai.providers.capability_handler import CapabilityHandler


@pytest.fixture
def generation_request() -> GenerationRequest:
    return GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Hello.",
            ),
        ),
        model="test-model",
        temperature=0.8,
        max_tokens=128,
        top_p=0.9,
        frequency_penalty=0.2,
        presence_penalty=0.1,
    )


def test_builds_all_supported_options(generation_request):
    options = CapabilityHandler.build_generation_options(
        generation_request,
        ProviderCapabilities(),
    )

    assert options == {
        "temperature": 0.8,
        "max_tokens": 128,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.1,
    }


def test_filters_unsupported_options(generation_request):
    capabilities = ProviderCapabilities(
        supports_temperature=False,
        supports_max_tokens=False,
        supports_top_p=False,
        supports_frequency_penalty=False,
        supports_presence_penalty=False,
    )

    options = CapabilityHandler.build_generation_options(
        generation_request,
        capabilities,
    )

    assert options == {}


def test_omits_unset_max_tokens():
    generation_request = GenerationRequest(
        messages=(
            AIMessage(
                role=MessageRole.USER,
                content="Hello.",
            ),
        ),
        model="test-model",
    )

    options = CapabilityHandler.build_generation_options(
        generation_request,
        ProviderCapabilities(),
        include_unset_max_tokens=False,
    )

    assert "max_tokens" not in options


def test_supports_provider_specific_max_tokens_key(generation_request):
    options = CapabilityHandler.build_generation_options(
        generation_request,
        ProviderCapabilities(),
        max_tokens_key="num_predict",
    )

    assert options["num_predict"] == 128
    assert "max_tokens" not in options


def test_rejects_unsupported_system_messages():
    messages = (
        AIMessage(
            role=MessageRole.SYSTEM,
            content="System instruction.",
        ),
    )

    capabilities = ProviderCapabilities(
        supports_system_messages=False,
    )

    with pytest.raises(AIProviderCapabilityError):
        CapabilityHandler.validate_system_messages(
            messages,
            capabilities,
            "Test Provider",
        )


def test_accepts_supported_system_messages():
    messages = (
        AIMessage(
            role=MessageRole.SYSTEM,
            content="System instruction.",
        ),
    )

    CapabilityHandler.validate_system_messages(
        messages,
        ProviderCapabilities(),
        "Test Provider",
    )


def test_rejects_unsupported_streaming():
    capabilities = ProviderCapabilities(
        supports_streaming=False,
    )

    with pytest.raises(AIProviderCapabilityError):
        CapabilityHandler.validate_streaming(
            capabilities,
            "Test Provider",
        )


def test_accepts_supported_streaming():
    CapabilityHandler.validate_streaming(
        ProviderCapabilities(),
        "Test Provider",
    )
