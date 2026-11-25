
import os
import sys

from seven.core.integrations.system import (
    EnvironmentInformation,
    SystemInformation,
    detect_environment_information,
    detect_system_information,
)


def test_detect_system_information_returns_valid_result() -> None:
    result = detect_system_information()

    assert isinstance(result, SystemInformation)
    assert result.hostname
    assert result.operating_system
    assert result.operating_system_release
    assert result.architecture
    assert result.python_version
    assert result.python_implementation


def test_detect_environment_information_returns_valid_result() -> None:
    result = detect_environment_information()

    assert isinstance(result, EnvironmentInformation)
    assert result.working_directory == os.getcwd()
    assert result.home_directory
    assert result.temporary_directory
    assert result.executable_path == sys.executable


def test_environment_information_is_immutable() -> None:
    result = detect_environment_information()

    try:
        result.working_directory = "modified"
    except AttributeError:
        pass
    else:
        raise AssertionError("EnvironmentInformation must be immutable.")


def test_system_information_is_immutable() -> None:
    result = detect_system_information()

    try:
        result.hostname = "modified"
    except AttributeError:
        pass
    else:
        raise AssertionError("SystemInformation must be immutable.")
