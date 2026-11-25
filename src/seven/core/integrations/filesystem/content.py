
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileContent:
    path: Path
    content: str
    encoding: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class BinaryFileContent:
    path: Path
    content: bytes
    size_bytes: int


def read_text_file(
    path: Path,
    encoding: str = "utf-8",
) -> FileContent:
    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise IsADirectoryError(path)

    size_bytes = path.stat().st_size
    content = path.read_text(encoding=encoding)

    return FileContent(
        path=path,
        content=content,
        encoding=encoding,
        size_bytes=size_bytes,
    )


def read_binary_file(path: Path) -> BinaryFileContent:
    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise IsADirectoryError(path)

    content = path.read_bytes()

    return BinaryFileContent(
        path=path,
        content=content,
        size_bytes=len(content),
    )
