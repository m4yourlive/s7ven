
import os
import platform
import socket
import sys
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemInformation:
    hostname: str
    operating_system: str
    operating_system_release: str
    architecture: str
    processor: str
    python_version: str
    python_implementation: str


@dataclass(frozen=True, slots=True)
class EnvironmentInformation:
    working_directory: str
    home_directory: str
    temporary_directory: str
    executable_path: str
    environment: str | None


def detect_system_information() -> SystemInformation:
    return SystemInformation(
        hostname=socket.gethostname(),
        operating_system=platform.system(),
        operating_system_release=platform.release(),
        architecture=platform.machine(),
        processor=platform.processor(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
    )


def detect_environment_information() -> EnvironmentInformation:
    return EnvironmentInformation(
        working_directory=os.getcwd(),
        home_directory=os.path.expanduser("~"),
        temporary_directory=os.path.abspath(
            os.getenv("TEMP") or os.getenv("TMP") or "/tmp"
        ),
        executable_path=sys.executable,
        environment=os.getenv("SEVEN_ENVIRONMENT"),
    )
