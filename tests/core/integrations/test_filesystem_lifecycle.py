
from pathlib import Path

import pytest

from seven.core.integrations.filesystem import (
    FileSystemAccessPolicy,
    FileSystemService,
)


@pytest.fixture
def filesystem(tmp_path: Path) -> FileSystemService:
    return FileSystemService(
        policy=FileSystemAccessPolicy(
            allowed_roots=(tmp_path,),
            allow_read=True,
            allow_write=True,
            allow_delete=True,
        )
    )


def test_service_has_correct_name(filesystem: FileSystemService) -> None:
    assert filesystem.name == "filesystem"


def test_service_starts_stopped(filesystem: FileSystemService) -> None:
    assert filesystem.is_running is False


def test_start_activates_service(filesystem: FileSystemService) -> None:
    filesystem.start()

    assert filesystem.is_running is True


def test_stop_deactivates_service(filesystem: FileSystemService) -> None:
    filesystem.start()
    filesystem.stop()

    assert filesystem.is_running is False


def test_start_is_idempotent(filesystem: FileSystemService) -> None:
    filesystem.start()
    filesystem.start()

    assert filesystem.is_running is True


def test_stop_is_idempotent(filesystem: FileSystemService) -> None:
    filesystem.stop()

    assert filesystem.is_running is False


def test_operations_require_running_service(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("content", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not running"):
        filesystem.read_text(target)


def test_operations_work_after_start(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("content", encoding="utf-8")

    filesystem.start()

    result = filesystem.read_text(target)

    assert result.content == "content"


def test_operations_fail_after_stop(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("content", encoding="utf-8")

    filesystem.start()
    filesystem.stop()

    with pytest.raises(RuntimeError, match="not running"):
        filesystem.read_text(target)
