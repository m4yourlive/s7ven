
"""Tests for the local runtime registry."""

from collections.abc import AsyncIterator

import pytest

from seven.core.ai.interfaces.local_runtime import LocalModelRuntime
from seven.core.ai.models import (
    GenerationRequest,
    GenerationResponse,
)
from seven.core.ai.runtimes import LocalRuntimeRegistry


class FakeLocalRuntime(LocalModelRuntime):
    """Test implementation of a local runtime."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.closed = False

    @property
    def runtime_name(self) -> str:
        return self._name

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResponse:
        return GenerationResponse(
            content="Test response",
            model="test-model",
            provider=self._name,
        )

    async def stream(
        self,
        request: GenerationRequest,
    ) -> AsyncIterator[str]:
        yield "Test response"

    async def close(self) -> None:
        self.closed = True

@pytest.fixture
def registry() -> LocalRuntimeRegistry:
    return LocalRuntimeRegistry()

def test_registry_starts_empty(
    registry: LocalRuntimeRegistry,
) -> None:
    assert len(registry) == 0
    assert registry.list_runtimes() == []
    assert registry.is_closed is False

def test_register_runtime(
    registry: LocalRuntimeRegistry,
) -> None:
    runtime = FakeLocalRuntime("ollama")

    registry.register(runtime)

    assert len(registry) == 1
    assert registry.get("ollama") is runtime

def test_register_normalizes_runtime_name(
    registry: LocalRuntimeRegistry,
) -> None:
    runtime = FakeLocalRuntime("  Ollama ")

    registry.register(runtime)

    assert registry.has_runtime("OLLAMA")
    assert registry.get("ollama") is runtime

def test_register_rejects_duplicate_runtime_name(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.register(FakeLocalRuntime("ollama"))

    with pytest.raises(ValueError, match="already registered"):
        registry.register(FakeLocalRuntime("OLLAMA"))

def test_register_rejects_invalid_runtime(
    registry: LocalRuntimeRegistry,
) -> None:
    with pytest.raises(TypeError, match="LocalModelRuntime"):
        registry.register(object())  # type: ignore[arg-type]

def test_get_rejects_unknown_runtime(
    registry: LocalRuntimeRegistry,
) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get("unknown")

def test_unregister_runtime(
    registry: LocalRuntimeRegistry,
) -> None:
    runtime = FakeLocalRuntime("ollama")
    registry.register(runtime)

    removed = registry.unregister("ollama")

    assert removed is runtime
    assert len(registry) == 0
    assert not registry.has_runtime("ollama")

def test_unregister_does_not_close_runtime(
    registry: LocalRuntimeRegistry,
) -> None:
    runtime = FakeLocalRuntime("ollama")
    registry.register(runtime)

    registry.unregister("ollama")

    assert runtime.closed is False

def test_unregister_rejects_unknown_runtime(
    registry: LocalRuntimeRegistry,
) -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.unregister("unknown")

def test_has_runtime(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.register(FakeLocalRuntime("ollama"))

    assert registry.has_runtime("ollama") is True
    assert registry.has_runtime("llamacpp") is False

def test_list_runtimes(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.register(FakeLocalRuntime("ollama"))
    registry.register(FakeLocalRuntime("llamacpp"))

    assert registry.list_runtimes() == ["ollama", "llamacpp"]

def test_contains_runtime(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.register(FakeLocalRuntime("ollama"))

    assert "ollama" in registry
    assert "unknown" not in registry

def test_close_clears_registry(
    registry: LocalRuntimeRegistry,
) -> None:
    runtime = FakeLocalRuntime("ollama")
    registry.register(runtime)

    registry.close()

    assert registry.is_closed is True
    assert len(registry) == 0
    assert runtime.closed is False

def test_close_is_idempotent(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.register(FakeLocalRuntime("ollama"))

    registry.close()
    registry.close()

    assert registry.is_closed is True

def test_closed_registry_rejects_registration(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.close()

    with pytest.raises(RuntimeError, match="closed"):
        registry.register(FakeLocalRuntime("ollama"))

def test_closed_registry_rejects_lookup(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.close()

    with pytest.raises(RuntimeError, match="closed"):
        registry.get("ollama")

def test_closed_registry_rejects_unregistration(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.close()

    with pytest.raises(RuntimeError, match="closed"):
        registry.unregister("ollama")

def test_closed_registry_rejects_runtime_checks(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.close()

    with pytest.raises(RuntimeError, match="closed"):
        registry.has_runtime("ollama")

def test_closed_registry_rejects_listing(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.close()

    with pytest.raises(RuntimeError, match="closed"):
        registry.list_runtimes()

def test_closed_registry_contains_nothing(
    registry: LocalRuntimeRegistry,
) -> None:
    registry.register(FakeLocalRuntime("ollama"))
    registry.close()

    assert "ollama" not in registry