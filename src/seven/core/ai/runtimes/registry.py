"""Registry for local AI model runtimes."""

from seven.core.ai.interfaces.local_runtime import LocalModelRuntime


class LocalRuntimeRegistry:
    """Manage the registration of local model runtimes."""

    def __init__(self) -> None:
        self._runtimes: dict[str, LocalModelRuntime] = {}
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Local runtime registry is closed")

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError("Runtime name must be a string")

        normalized = name.strip().lower()

        if not normalized:
            raise ValueError("Runtime name must not be empty")

        return normalized

    def register(self, runtime: LocalModelRuntime) -> None:
        self._ensure_open()

        if not isinstance(runtime, LocalModelRuntime):
            raise TypeError("runtime must implement LocalModelRuntime")

        name = self._normalize_name(runtime.runtime_name)

        if name in self._runtimes:
            raise ValueError(f"Runtime {name!r} is already registered")

        self._runtimes[name] = runtime

    def get(self, name: str) -> LocalModelRuntime:
        self._ensure_open()
        normalized = self._normalize_name(name)

        try:
            return self._runtimes[normalized]
        except KeyError as exc:
            raise KeyError(
                f"Runtime {normalized!r} is not registered"
            ) from exc

    def unregister(self, name: str) -> LocalModelRuntime:
        self._ensure_open()
        normalized = self._normalize_name(name)

        try:
            return self._runtimes.pop(normalized)
        except KeyError as exc:
            raise KeyError(
                f"Runtime {normalized!r} is not registered"
            ) from exc

    def has_runtime(self, name: str) -> bool:
        self._ensure_open()
        return self._normalize_name(name) in self._runtimes

    def list_runtimes(self) -> list[str]:
        self._ensure_open()
        return list(self._runtimes)

    def close(self) -> None:
        if self._closed:
            return

        self._runtimes.clear()
        self._closed = True

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str) or self._closed:
            return False

        return name.strip().lower() in self._runtimes

    def __len__(self) -> int:
        return len(self._runtimes)
