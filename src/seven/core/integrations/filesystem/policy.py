
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FileOperation(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class FileSystemAccessPolicy:
    allowed_roots: tuple[Path, ...]
    allow_read: bool = True
    allow_write: bool = False
    allow_delete: bool = False
    max_file_size_bytes: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        normalized_roots = tuple(
            Path(root).expanduser().resolve()
            for root in self.allowed_roots
        )

        object.__setattr__(self, "allowed_roots", normalized_roots)

        if self.max_file_size_bytes <= 0:
            raise ValueError("Maximum file size must be positive.")

    def authorize(
        self,
        path: str | Path,
        operation: FileOperation,
    ) -> Path:
        permissions = {
            FileOperation.READ: self.allow_read,
            FileOperation.WRITE: self.allow_write,
            FileOperation.DELETE: self.allow_delete,
        }

        if not permissions[operation]:
            raise PermissionError(
                f"File operation is not permitted: {operation.value}"
            )

        resolved_path = Path(path).expanduser().resolve()

        if not any(
            resolved_path.is_relative_to(root)
            for root in self.allowed_roots
        ):
            raise PermissionError(
                "Path is outside the allowed directories."
            )

        return resolved_path

    def validate_file_size(self, size_bytes: int) -> None:
        if size_bytes < 0:
            raise ValueError("File size cannot be negative.")

        if size_bytes > self.max_file_size_bytes:
            raise PermissionError(
                "File exceeds the configured size limit."
            )
