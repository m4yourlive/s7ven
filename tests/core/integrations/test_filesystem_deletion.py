
from pathlib import Path

import pytest

from seven.core.integrations.filesystem import (
    FileSystemAccessPolicy,
    FileSystemService,
)


@pytest.fixture
def filesystem(tmp_path: Path) -> FileSystemService:
    policy = FileSystemAccessPolicy(
        allowed_roots=(tmp_path,),
        allow_read=True,
        allow_write=True,
        allow_delete=True,
    )
    service = FileSystemService(policy=policy)
    service.start()
    return service


def test_delete_file_removes_file(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.txt"
    target.write_text("content", encoding="utf-8")

    result = filesystem.delete(target)

    assert result == target
    assert not target.exists()


def test_delete_file_rejects_missing_file(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        filesystem.delete(tmp_path / "missing.txt")


def test_delete_file_rejects_directory(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "folder"
    directory.mkdir()

    with pytest.raises(IsADirectoryError):
        filesystem.delete(directory)


def test_delete_directory_removes_empty_directory(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()

    result = filesystem.delete_directory(directory)

    assert result == directory
    assert not directory.exists()


def test_delete_directory_rejects_non_empty_directory(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "non-empty"
    directory.mkdir()
    (directory / "sample.txt").write_text("content", encoding="utf-8")

    with pytest.raises(OSError):
        filesystem.delete_directory(directory)


def test_delete_rejects_when_permission_disabled(
    tmp_path: Path,
) -> None:
    policy = FileSystemAccessPolicy(
        allowed_roots=(tmp_path,),
        allow_delete=False,
    )
    service = FileSystemService(policy=policy)
    service.start()

    target = tmp_path / "sample.txt"
    target.write_text("content", encoding="utf-8")

    with pytest.raises(PermissionError):
        service.delete(target)

    assert target.exists()


def test_delete_rejects_allowed_root(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    with pytest.raises(PermissionError):
        filesystem.delete_directory(tmp_path)


def test_delete_rejects_path_outside_allowed_root(
    filesystem: FileSystemService,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("content", encoding="utf-8")

    try:
        with pytest.raises(PermissionError):
            filesystem.delete(outside)
    finally:
        outside.unlink(missing_ok=True)
