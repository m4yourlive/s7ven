
from pathlib import Path

import pytest

from seven.core.integrations.filesystem import (
    FileSystemAccessPolicy,
    FileSystemService,
    FileWriteResult,
    create_directory,
    write_binary_file,
    write_text_file,
)


def create_service(root: Path) -> FileSystemService:
    service = FileSystemService(
        FileSystemAccessPolicy(
            allowed_roots=(root,),
            allow_read=True,
            allow_write=True,
            max_file_size_bytes=1024,
        )
    )
    service.start()
    return service


def test_write_text_file(tmp_path: Path) -> None:
    target = tmp_path / "document.txt"

    result = write_text_file(target, "Seven")

    assert isinstance(result, FileWriteResult)
    assert result.bytes_written == 5
    assert result.created is True
    assert target.read_text(encoding="utf-8") == "Seven"


def test_write_binary_file(tmp_path: Path) -> None:
    target = tmp_path / "document.bin"

    result = write_binary_file(target, b"\x00\x01")

    assert result.bytes_written == 2
    assert target.read_bytes() == b"\x00\x01"


def test_write_rejects_existing_file_by_default(
    tmp_path: Path,
) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_text_file(target, "Replacement")

    assert target.read_text(encoding="utf-8") == "Original"


def test_write_supports_explicit_overwrite(
    tmp_path: Path,
) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Original", encoding="utf-8")

    result = write_text_file(target, "Replacement", overwrite=True)

    assert result.created is False
    assert target.read_text(encoding="utf-8") == "Replacement"


def test_write_rejects_missing_parent_by_default(
    tmp_path: Path,
) -> None:
    target = tmp_path / "missing" / "document.txt"

    with pytest.raises(FileNotFoundError):
        write_text_file(target, "Seven")


def test_write_can_create_parent_directories(
    tmp_path: Path,
) -> None:
    target = tmp_path / "nested" / "document.txt"

    write_text_file(target, "Seven", create_parents=True)

    assert target.read_text(encoding="utf-8") == "Seven"


def test_create_directory(tmp_path: Path) -> None:
    target = tmp_path / "new_folder"

    result = create_directory(target)

    assert result == target
    assert target.is_dir()


def test_service_write_text(tmp_path: Path) -> None:
    service = create_service(tmp_path)
    target = tmp_path / "document.txt"

    result = service.write_text(target, "Seven")

    assert result.created is True
    assert target.read_text(encoding="utf-8") == "Seven"


def test_service_rejects_write_when_disabled(
    tmp_path: Path,
) -> None:
    service = FileSystemService(
        FileSystemAccessPolicy(
            allowed_roots=(tmp_path,),
            allow_write=False,
        )
    )
    service.start()

    with pytest.raises(PermissionError):
        service.write_text(tmp_path / "document.txt", "Seven")


def test_service_rejects_oversized_content(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)

    with pytest.raises(PermissionError):
        service.write_text(
            tmp_path / "document.txt",
            "X" * 1025,
        )


def test_service_rejects_write_outside_allowed_root(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    service = create_service(allowed)

    with pytest.raises(PermissionError):
        service.write_text(tmp_path / "outside.txt", "Seven")


def test_service_creates_directory(
    tmp_path: Path,
) -> None:
    service = create_service(tmp_path)
    target = tmp_path / "new_folder"

    result = service.create_directory(target)

    assert result == target
    assert target.is_dir()
