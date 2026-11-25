
from pathlib import Path

import pytest

from seven.core.integrations.filesystem import (
    BinaryFileContent,
    FileContent,
    FileSystemAccessPolicy,
    FileSystemService,
    read_binary_file,
    read_text_file,
)


def test_read_text_file(tmp_path: Path) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Seven", encoding="utf-8")

    result = read_text_file(target)

    assert isinstance(result, FileContent)
    assert result.content == "Seven"
    assert result.encoding == "utf-8"
    assert result.size_bytes == 5


def test_read_binary_file(tmp_path: Path) -> None:
    target = tmp_path / "image.bin"
    target.write_bytes(b"\x00\x01\x02")

    result = read_binary_file(target)

    assert isinstance(result, BinaryFileContent)
    assert result.content == b"\x00\x01\x02"
    assert result.size_bytes == 3


def test_read_text_file_rejects_missing_path(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        read_text_file(tmp_path / "missing.txt")


def test_read_binary_file_rejects_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(IsADirectoryError):
        read_binary_file(tmp_path)


def test_filesystem_service_reads_text(
    tmp_path: Path,
) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Seven", encoding="utf-8")

    service = FileSystemService(
        FileSystemAccessPolicy(allowed_roots=(tmp_path,))
    )
    service.start()

    result = service.read_text(target)

    assert result.content == "Seven"


def test_filesystem_service_reads_binary(
    tmp_path: Path,
) -> None:
    target = tmp_path / "image.bin"
    target.write_bytes(b"\x00\x01\x02")

    service = FileSystemService(
        FileSystemAccessPolicy(allowed_roots=(tmp_path,))
    )
    service.start()

    result = service.read_binary(target)

    assert result.content == b"\x00\x01\x02"


def test_filesystem_service_rejects_oversized_text(
    tmp_path: Path,
) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Seven", encoding="utf-8")

    service = FileSystemService(
        FileSystemAccessPolicy(
            allowed_roots=(tmp_path,),
            max_file_size_bytes=2,
        )
    )
    service.start()

    with pytest.raises(PermissionError):
        service.read_text(target)


def test_filesystem_service_rejects_outside_path(
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
        service.read_text(outside)


def test_read_text_file_supports_custom_encoding(
    tmp_path: Path,
) -> None:
    target = tmp_path / "document.txt"
    target.write_text("Seven", encoding="utf-16")

    result = read_text_file(target, encoding="utf-16")

    assert result.content == "Seven"
    assert result.encoding == "utf-16"
