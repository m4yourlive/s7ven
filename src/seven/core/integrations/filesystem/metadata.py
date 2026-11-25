
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FileMetadata:
    path: Path
    name: str
    extension: str
    size_bytes: int
    is_file: bool
    is_directory: bool
    is_symlink: bool
    created_at: float
    modified_at: float


@dataclass(frozen=True, slots=True)
class DirectoryEntry:
    path: Path
    name: str
    is_file: bool
    is_directory: bool
    is_symlink: bool
    size_bytes: int


def read_file_metadata(path: Path) -> FileMetadata:
    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_file() and not path.is_dir():
        raise OSError(f"Unsupported filesystem entry: {path}")

    stat = path.stat()

    return FileMetadata(
        path=path,
        name=path.name,
        extension=path.suffix,
        size_bytes=stat.st_size,
        is_file=path.is_file(),
        is_directory=path.is_dir(),
        is_symlink=path.is_symlink(),
        created_at=stat.st_ctime,
        modified_at=stat.st_mtime,
    )


def list_directory(path: Path) -> tuple[DirectoryEntry, ...]:
    if not path.exists():
        raise FileNotFoundError(path)

    if not path.is_dir():
        raise NotADirectoryError(path)

    entries: list[DirectoryEntry] = []

    for entry in sorted(path.iterdir(), key=lambda item: item.name.casefold()):
        if entry.is_symlink():
            size_bytes = 0
            is_file = False
            is_directory = False
        else:
            is_file = entry.is_file()
            is_directory = entry.is_dir()
            size_bytes = entry.stat().st_size if is_file else 0

        entries.append(
            DirectoryEntry(
                path=entry,
                name=entry.name,
                is_file=is_file,
                is_directory=is_directory,
                is_symlink=entry.is_symlink(),
                size_bytes=size_bytes,
            )
        )

    return tuple(entries)
