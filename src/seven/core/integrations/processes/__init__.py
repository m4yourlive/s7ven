
"""Operating system process integration."""

from seven.core.integrations.processes.discovery import (
    PsutilProcessDiscovery,
)
from seven.core.integrations.processes.inspection import (
    PsutilProcessInspector,
)
from seven.core.integrations.processes.interfaces import (
    ProcessDiscovery,
    ProcessInspector,
    ProcessTerminator,
)
from seven.core.integrations.processes.models import ProcessInfo
from seven.core.integrations.processes.policy import ProcessAccessPolicy
from seven.core.integrations.processes.service import ProcessService
from seven.core.integrations.processes.termination import (
    PsutilProcessTerminator,
)

__all__ = [
    "ProcessAccessPolicy",
    "ProcessDiscovery",
    "ProcessInfo",
    "ProcessInspector",
    "ProcessService",
    "ProcessTerminator",
    "PsutilProcessDiscovery",
    "PsutilProcessInspector",
    "PsutilProcessTerminator",
]
