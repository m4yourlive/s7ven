
import platform
from dataclasses import dataclass
from enum import Enum

class OperatingSystem(str, Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class PlatformInfo:
    operating_system: OperatingSystem
    system: str
    release: str
    version: str
    machine: str

    @property
    def is_windows(self) -> bool:
        return self.operating_system == OperatingSystem.WINDOWS

    @property
    def is_linux(self) -> bool:
        return self.operating_system == OperatingSystem.LINUX

    @property
    def is_macos(self) -> bool:
        return self.operating_system == OperatingSystem.MACOS

def detect_platform() -> PlatformInfo:
    system = platform.system()

    operating_systems = {
        "Windows": OperatingSystem.WINDOWS,
        "Linux": OperatingSystem.LINUX,
        "Darwin": OperatingSystem.MACOS,
    }

    return PlatformInfo(
        operating_system=operating_systems.get(
            system,
            OperatingSystem.UNKNOWN,
        ),
        system=system,
        release=platform.release(),
        version=platform.version(),
        machine=platform.machine(),
    )