
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from seven.core.integrations.applications import (
    Application,
    SubprocessApplicationLauncher,
)


def test_launcher_starts_application_and_returns_pid(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    process = MagicMock()
    process.pid = 1234

    with patch(
        "seven.core.integrations.applications.launcher.subprocess.Popen",
        return_value=process,
    ) as popen:
        launcher = SubprocessApplicationLauncher()
        application = Application(
            name="Example",
            executable=executable,
        )

        result = launcher.launch(application)

    assert result == 1234

    popen.assert_called_once_with(
        [str(executable.resolve())],
        cwd=None,
        shell=False,
        close_fds=True,
    )


def test_launcher_passes_arguments_without_shell(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    process = MagicMock()
    process.pid = 5678

    with patch(
        "seven.core.integrations.applications.launcher.subprocess.Popen",
        return_value=process,
    ) as popen:
        launcher = SubprocessApplicationLauncher()
        application = Application(
            name="Example",
            executable=executable,
            arguments=("--profile", "My Profile"),
        )

        launcher.launch(application)

    command = popen.call_args.args[0]

    assert command == [
        str(executable.resolve()),
        "--profile",
        "My Profile",
    ]
    assert popen.call_args.kwargs["shell"] is False


def test_launcher_uses_configured_working_directory(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    working_directory = tmp_path / "workspace"
    working_directory.mkdir()

    process = MagicMock()
    process.pid = 1234

    with patch(
        "seven.core.integrations.applications.launcher.subprocess.Popen",
        return_value=process,
    ) as popen:
        launcher = SubprocessApplicationLauncher(
            working_directory=working_directory,
        )

        launcher.launch(
            Application(
                name="Example",
                executable=executable,
            )
        )

    assert popen.call_args.kwargs["cwd"] == str(
        working_directory.resolve()
    )


def test_launcher_rejects_missing_executable(
    tmp_path: Path,
) -> None:
    launcher = SubprocessApplicationLauncher()

    application = Application(
        name="Missing",
        executable=tmp_path / "missing.exe",
    )

    with pytest.raises(FileNotFoundError):
        launcher.launch(application)


def test_launcher_rejects_directory_as_executable(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "application.exe"
    directory.mkdir()

    launcher = SubprocessApplicationLauncher()

    application = Application(
        name="Invalid",
        executable=directory,
    )

    with pytest.raises(ValueError):
        launcher.launch(application)


def test_launcher_rejects_invalid_application_type() -> None:
    launcher = SubprocessApplicationLauncher()

    with pytest.raises(TypeError):
        launcher.launch("example.exe")  # type: ignore[arg-type]


def test_launcher_propagates_process_start_error(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "example.exe"
    executable.write_text("", encoding="utf-8")

    with patch(
        "seven.core.integrations.applications.launcher.subprocess.Popen",
        side_effect=OSError("Process creation failed."),
    ):
        launcher = SubprocessApplicationLauncher()

        with pytest.raises(OSError):
            launcher.launch(
                Application(
                    name="Example",
                    executable=executable,
                )
            )
