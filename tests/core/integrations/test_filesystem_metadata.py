
from pathlib import Path

import pytest

from seven.core.integrations.filesystem import (
    DirectoryEntry,
    FileMetadata,
    FileSystemAccessPolicy,
    FileSystemService,
    list_directory,
    read_file_metadata,
)


def test_read_file_metadata(tmp_path: Path) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Seven", encoding="utf-8")

    result = read_file_metadata(target)

    assert isinstance(result, FileMetadata)
    assert result.name == "document.txt"
    assert result.extension == ".txt"
    assert result.size_bytes == 5
    assert result.is_file is True
    assert result.is_directory is False


def test_read_directory_metadata(tmp_path: Path) -> None:
    result = read_file_metadata(tmp_path)

    assert result.is_directory is True
    assert result.is_file is False


def test_read_metadata_rejects_missing_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        read_file_metadata(tmp_path / "missing.txt")


def test_list_directory_returns_sorted_entries(
    tmp_path: Path,
) -> None:
    (tmp_path / "zeta.txt").write_text("Z", encoding="utf-8")
    (tmp_path / "alpha.txt").write_text("A", encoding="utf-8")
    (tmp_path / "folder").mkdir()

    result = list_directory(tmp_path)

    assert isinstance(result, tuple)
    assert all(isinstance(item, DirectoryEntry) for item in result)
    assert [item.name for item in result] == [
        "alpha.txt",
        "folder",
        "zeta.txt",
    ]


def test_list_directory_rejects_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Seven", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        list_directory(target)


def test_filesystem_service_get_metadata(
    tmp_path: Path,
) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Seven", encoding="utf-8")

    policy = FileSystemAccessPolicy(
        allowed_roots=(tmp_path,),
        allow_read=True,
    )
    service = FileSystemService(policy)
    service.start()

    result = service.get_metadata(target)

    assert result.name == "document.txt"
    assert result.size_bytes == 5


def test_filesystem_service_enforces_size_limit(
    tmp_path: Path,
) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Seven", encoding="utf-8")

    policy = FileSystemAccessPolicy(
        allowed_roots=(tmp_path,),
        max_file_size_bytes=2,
    )
    service = FileSystemService(policy)
    service.start()

    with pytest.raises(PermissionError):
        service.get_metadata(target)


def test_filesystem_service_enforces_allowed_roots(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    outside = tmp_path / "outside.txt"
    outside.write_text("Seven", encoding="utf-8")

    service = FileSystemService(
        FileSystemAccessPolicy(allowed_roots=(allowed,))
    )
    service.start()

    with pytest.raises(PermissionError):
        service.get_metadata(outside)


def test_filesystem_service_lists_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "document.txt").write_text(
        "Seven",
        encoding="utf-8",
    )

    service = FileSystemService(
        FileSystemAccessPolicy(allowed_roots=(tmp_path,))
    )
    service.start()

    result = service.list_directory(tmp_path)

    assert len(result) == 1
    assert result[0].name == "document.txt"
