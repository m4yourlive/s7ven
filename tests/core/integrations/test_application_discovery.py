
import os
from pathlib import Path

from seven.core.integrations.applications import (
    Application,
    DirectoryApplicationDiscovery,
)


def test_discovery_finds_executable_files(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("placeholder", encoding="utf-8")

    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path,),
        extensions=(".exe",),
    )

    applications = discovery.discover()

    assert len(applications) == 1
    assert applications[0] == Application(
        name="example",
        executable=executable.resolve(),
    )


def test_discovery_ignores_non_matching_extensions(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.txt").write_text(
        "placeholder",
        encoding="utf-8",
    )

    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path,),
        extensions=(".exe",),
    )

    assert discovery.discover() == ()


def test_discovery_ignores_missing_directories(
    tmp_path: Path,
) -> None:
    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path / "missing",),
        extensions=(".exe",),
    )

    assert discovery.discover() == ()


def test_discovery_ignores_directories(
    tmp_path: Path,
) -> None:
    (tmp_path / "folder.exe").mkdir()

    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path,),
        extensions=(".exe",),
    )

    assert discovery.discover() == ()


def test_discovery_returns_sorted_results(
    tmp_path: Path,
) -> None:
    for name in ("zulu.exe", "alpha.exe", "bravo.exe"):
        (tmp_path / name).write_text("", encoding="utf-8")

    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path,),
        extensions=(".exe",),
    )

    names = [
        application.name
        for application in discovery.discover()
    ]

    assert names == ["alpha", "bravo", "zulu"]


def test_discovery_deduplicates_executables(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path, tmp_path),
        extensions=(".exe",),
    )

    assert len(discovery.discover()) == 1


def test_discovery_ignores_symlinks(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    link = tmp_path / "linked.exe"

    try:
        link.symlink_to(executable)
    except (OSError, NotImplementedError):
        return

    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path,),
        extensions=(".exe",),
    )

    assert [
        application.name
        for application in discovery.discover()
    ] == ["example"]


def test_discovery_supports_extension_without_dot(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path,),
        extensions=("EXE",),
    )

    assert len(discovery.discover()) == 1


def test_discovery_returns_application_instances(
    tmp_path: Path,
) -> None:
    (tmp_path / "example.exe").write_text("", encoding="utf-8")

    discovery = DirectoryApplicationDiscovery(
        directories=(tmp_path,),
        extensions=(".exe",),
    )

    assert all(
        isinstance(application, Application)
        for application in discovery.discover()
    )
