
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileWriteResult:
    path: Path
    bytes_written: int
    created: bool


def write_text_file(
    path: Path,
    content: str,
    encoding: str = "utf-8",
    overwrite: bool = False,
    create_parents: bool = False,
) -> FileWriteResult:
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(path)

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    existed = path.exists()
    data = content.encode(encoding)

    with path.open("wb") as file:
        file.write(data)

    return FileWriteResult(
        path=path,
        bytes_written=len(data),
        created=not existed,
    )


def write_binary_file(
    path: Path,
    content: bytes,
    overwrite: bool = False,
    create_parents: bool = False,
) -> FileWriteResult:
    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(path)

    if create_parents:
        path.parent.mkdir(parents=True, exist_ok=True)

    existed = path.exists()

    with path.open("wb") as file:
        file.write(content)

    return FileWriteResult(
        path=path,
        bytes_written=len(content),
        created=not existed,
    )


def create_directory(
    path: Path,
    parents: bool = False,
    exist_ok: bool = False,
) -> Path:
    path = Path(path)
    path.mkdir(parents=parents, exist_ok=exist_ok)
    return path


def delete_file(path: Path) -> Path:
    path = Path(path)

    if path.is_symlink():
        raise PermissionError("Symbolic link deletion is not allowed.")

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file():
        raise IsADirectoryError(path)

    path.unlink()
    return path


def delete_empty_directory(path: Path) -> Path:
    path = Path(path)

    if path.is_symlink():
        raise PermissionError("Symbolic link deletion is not allowed.")

    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_dir():
        raise NotADirectoryError(path)

    path.rmdir()
    return path
