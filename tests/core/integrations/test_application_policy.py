
from pathlib import Path

import pytest

from seven.core.integrations.applications import (
    Application,
    ApplicationAccessPolicy,
)


def test_policy_authorizes_allowed_executable(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    policy = ApplicationAccessPolicy(
        allowed_executables=(executable,),
    )

    result = policy.authorize(
        Application(
            name="Example",
            executable=executable,
        )
    )

    assert result == executable.resolve()


def test_policy_rejects_unlisted_executable(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    policy = ApplicationAccessPolicy()

    with pytest.raises(PermissionError):
        policy.authorize(
            Application(
                name="Example",
                executable=executable,
            )
        )


def test_policy_rejects_missing_executable(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "missing.exe"

    policy = ApplicationAccessPolicy(
        allowed_executables=(executable,),
    )

    with pytest.raises(FileNotFoundError):
        policy.authorize(
            Application(
                name="Missing",
                executable=executable,
            )
        )


def test_policy_rejects_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "example.exe"
    directory.mkdir()

    policy = ApplicationAccessPolicy(
        allowed_executables=(directory,),
    )

    with pytest.raises(PermissionError):
        policy.authorize(
            Application(
                name="Invalid",
                executable=directory,
            )
        )


def test_policy_normalizes_and_deduplicates_paths(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"

    policy = ApplicationAccessPolicy(
        allowed_executables=(
            executable,
            executable.parent / "." / executable.name,
        ),
    )

    assert policy.allowed_executables == (
        executable.resolve(),
    )


def test_policy_rejects_non_path_entries(
    tmp_path: Path,
) -> None:
    with pytest.raises(TypeError):
        ApplicationAccessPolicy(
            allowed_executables=("example.exe",),  # type: ignore[arg-type]
        )


def test_policy_is_immutable(
    tmp_path: Path,
) -> None:
    policy = ApplicationAccessPolicy()

    with pytest.raises(AttributeError):
        policy.allowed_executables = (tmp_path,)  # type: ignore[misc]
