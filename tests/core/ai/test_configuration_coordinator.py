
"""Tests for transactional AI configuration coordination."""

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from seven.core.ai.configuration_coordinator import (
    AIConfigurationCoordinator,
)
from seven.core.ai.engine import AIEngine
from seven.core.ai.exceptions import AIConfigurationError
from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.models import GenerationRequest, GenerationResponse
from seven.core.ai.providers.local_ai_provider import LocalAIProvider
from seven.core.ai.providers.openai_provider import OpenAIProvider
from seven.core.ai.request_factory import GenerationRequestFactory
from seven.core.ai.prompts import DEFAULT_PROMPTS, PromptManager
from seven.core.ai.runtimes import LocalRuntimeFactory
from seven.core.config import AppConfig, ConfigurationManager
from seven.core.config.config_schema import AIConfig


class FakeLocalRuntime(LocalModelRuntime):
    """Provide a lightweight local runtime for coordinator tests."""

    @property
    def runtime_name(self) -> str:
        return "test-runtime"

    async def generate(
        self,
        generation_request: GenerationRequest,
    ) -> GenerationResponse:
        raise NotImplementedError

    async def stream(
        self,
        generation_request: GenerationRequest,
    ) -> AsyncIterator[str]:
        if False:
            yield ""

    async def close(self) -> None:
        return None


@pytest.fixture
def configuration_environment(tmp_path: Path):
    """Create an isolated configuration environment."""

    config_path = tmp_path / "app.yaml"

    config_path.write_text(
        """
ai:
  mode: api
  provider: openai
  model: gpt-4o-mini
  api_key: test-api-key
""",
        encoding="utf-8",
    )

    config_manager = ConfigurationManager(
        config_path=config_path,
        schema=AppConfig,
    )

    config_manager.load()

    runtime_factory = LocalRuntimeFactory()
    runtime_factory.register(
        "test-runtime",
        lambda config: FakeLocalRuntime(),
    )

    return {
        "config_path": config_path,
        "config_manager": config_manager,
        "runtime_factory": runtime_factory,
    }


@pytest.fixture
def coordinator_components(configuration_environment):
    """Build the coordinator and its dependencies."""

    config_manager = configuration_environment["config_manager"]
    runtime_factory = configuration_environment["runtime_factory"]

    initial_provider = OpenAIProvider(
        api_key="test-api-key",
        model="gpt-4o-mini",
    )

    engine = AIEngine(initial_provider)

    request_factory = GenerationRequestFactory(
        config=config_manager.config.ai,
        prompt_manager=PromptManager(DEFAULT_PROMPTS),
    )

    coordinator = AIConfigurationCoordinator(
        config_manager=config_manager,
        engine=engine,
        request_factory=request_factory,
        local_runtime_factory=runtime_factory,
    )

    return {
        **configuration_environment,
        "engine": engine,
        "initial_provider": initial_provider,
        "request_factory": request_factory,
        "coordinator": coordinator,
    }


def write_configuration(
    config_path: Path,
    content: str,
) -> None:
    """Replace the application configuration file."""

    config_path.write_text(
        content,
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_reload_switches_from_api_to_local(
    coordinator_components,
):
    components = coordinator_components

    write_configuration(
        components["config_path"],
        """
ai:
  mode: local
  provider: openai
  model: gpt-4o-mini
  local_runtime: test-runtime
  local_model: test-model
""",
    )

    provider = await components["coordinator"].reload()

    assert isinstance(provider, LocalAIProvider)
    assert components["engine"].provider is provider

    assert components["config_manager"].config.ai.mode == "local"
    assert components["request_factory"].config.mode == "local"


@pytest.mark.asyncio
async def test_reload_switches_from_local_to_api(
    coordinator_components,
):
    components = coordinator_components

    write_configuration(
        components["config_path"],
        """
ai:
  mode: local
  provider: openai
  local_runtime: test-runtime
  local_model: test-model
""",
    )

    local_provider = await components["coordinator"].reload()

    assert isinstance(local_provider, LocalAIProvider)

    write_configuration(
        components["config_path"],
        """
ai:
  mode: api
  provider: openai
  model: gpt-4o-mini
  api_key: updated-api-key
""",
    )

    api_provider = await components["coordinator"].reload()

    assert isinstance(api_provider, OpenAIProvider)
    assert api_provider is components["engine"].provider
    assert api_provider is not local_provider

    assert components["config_manager"].config.ai.mode == "api"
    assert components["request_factory"].config.mode == "api"


@pytest.mark.asyncio
async def test_invalid_configuration_preserves_active_provider(
    coordinator_components,
):
    components = coordinator_components

    original_provider = components["engine"].provider
    original_config = components["config_manager"].config.ai
    original_request_config = components["request_factory"].config

    write_configuration(
        components["config_path"],
        """
ai:
  mode: api
  provider: openai
  model: gpt-4o-mini
""",
    )

    with pytest.raises(AIConfigurationError):
        await components["coordinator"].reload()

    assert components["engine"].provider is original_provider
    assert components["config_manager"].config.ai == original_config
    assert components["request_factory"].config is original_request_config


@pytest.mark.asyncio
async def test_failed_candidate_commit_preserves_active_state(
    coordinator_components,
    monkeypatch,
):
    components = coordinator_components

    original_provider = components["engine"].provider
    original_config = components["config_manager"].config.ai
    original_request_config = components["request_factory"].config

    write_configuration(
        components["config_path"],
        """
ai:
  mode: local
  provider: openai
  local_runtime: test-runtime
  local_model: test-model
""",
    )

    def fail_commit(candidate):
        raise RuntimeError("Simulated configuration commit failure")

    monkeypatch.setattr(
        components["config_manager"],
        "commit_candidate",
        fail_commit,
    )

    with pytest.raises(RuntimeError, match="commit failure"):
        await components["coordinator"].reload()

    assert components["engine"].provider is original_provider
    assert components["config_manager"].config.ai == original_config
    assert components["request_factory"].config is original_request_config


@pytest.mark.asyncio
async def test_reload_rejects_closed_engine(
    coordinator_components,
):
    components = coordinator_components

    await components["engine"].close()

    with pytest.raises(AIConfigurationError, match="engine is closed"):
        await components["coordinator"].reload()


def test_coordinator_exposes_dependencies(
    coordinator_components,
):
    components = coordinator_components
    coordinator = components["coordinator"]

    assert coordinator.config_manager is components["config_manager"]
    assert coordinator.engine is components["engine"]
    assert coordinator.request_factory is components["request_factory"]
    assert coordinator.active_provider is components["initial_provider"]
