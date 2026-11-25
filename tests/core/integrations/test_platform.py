
from seven.core.integrations.system import (
    OperatingSystem,
    PlatformInfo,
    detect_platform,
)


def test_detect_platform_returns_platform_info() -> None:
    result = detect_platform()

    assert isinstance(result, PlatformInfo)
    assert isinstance(result.operating_system, OperatingSystem)
    assert result.system
    assert result.machine


def test_detect_platform_identifies_current_operating_system() -> None:
    result = detect_platform()

    assert result.operating_system != OperatingSystem.UNKNOWN


def test_platform_info_windows_property() -> None:
    info = PlatformInfo(
        operating_system=OperatingSystem.WINDOWS,
        system="Windows",
        release="11",
        version="10.0.26100",
        machine="AMD64",
    )

    assert info.is_windows is True
    assert info.is_linux is False
    assert info.is_macos is False


def test_platform_info_linux_property() -> None:
    info = PlatformInfo(
        operating_system=OperatingSystem.LINUX,
        system="Linux",
        release="6.8.0",
        version="generic",
        machine="x86_64",
    )

    assert info.is_linux is True
    assert info.is_windows is False
    assert info.is_macos is False


def test_platform_info_macos_property() -> None:
    info = PlatformInfo(
        operating_system=OperatingSystem.MACOS,
        system="Darwin",
        release="24.0.0",
        version="Darwin Kernel Version",
        machine="arm64",
    )

    assert info.is_macos is True
    assert info.is_windows is False
    assert info.is_linux is False


def test_platform_info_is_immutable() -> None:
    info = detect_platform()

    try:
        info.machine = "modified"
    except AttributeError:
        pass
    else:
        raise AssertionError("PlatformInfo must be immutable.")
