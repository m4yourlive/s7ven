from dataclasses import dataclass

from seven.core.events import Event


@dataclass(frozen=True, slots=True)
class RuntimeStarting(Event):
    pass

@dataclass(frozen=True, slots=True)
class RuntimeStarted(Event):
    pass

@dataclass(frozen=True, slots=True)
class RuntimeStopping(Event):
    pass

@dataclass(frozen=True, slots=True)
class RuntimeStopped(Event):
    pass

@dataclass(frozen=True, slots=True)
class RuntimeFailed(Event):
    reason: str