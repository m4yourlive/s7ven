
from pathlib import Path

import pytest

from seven.core.integrations.applications import Application


def test_application_stores_basic_information() -> None:
    application = Application(
        name="Visual Studio Code",
        executable=Path("C:/Program Files/VS Code/code.exe"),
        arguments=("--disable-extensions",),
        description="Code editor",
    )

    assert application.name == "Visual Studio Code"
    assert application.executable == Path(
        "C:/Program Files/VS Code/code.exe"
    )
    assert application.arguments == ("--disable-extensions",)
    assert application.description == "Code editor"


def test_application_normalizes_name() -> None:
    application = Application(
        name="  Visual Studio Code  ",
        executable=Path("code.exe"),
    )

    assert application.name == "Visual Studio Code"


def test_application_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        Application(
            name=" ",
            executable=Path("code.exe"),
        )


def test_application_rejects_non_path_executable() -> None:
    with pytest.raises(TypeError):
        Application(
            name="Example",
            executable="example.exe",  # type: ignore[arg-type]
        )


def test_application_rejects_non_string_arguments() -> None:
    with pytest.raises(TypeError):
        Application(
            name="Example",
            executable=Path("example.exe"),
            arguments=(123,),  # type: ignore[arg-type]
        )


def test_application_is_immutable() -> None:
    application = Application(
        name="Example",
        executable=Path("example.exe"),
    )

    with pytest.raises(AttributeError):
        application.name = "Changed"  # type: ignore[misc]