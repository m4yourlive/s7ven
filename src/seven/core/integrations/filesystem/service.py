
from pathlib import Path

from seven.core.integrations.filesystem.content import (
    BinaryFileContent,
    FileContent,
    read_binary_file,
    read_text_file,
)
from seven.core.integrations.filesystem.metadata import (
    DirectoryEntry,
    FileMetadata,
    list_directory,
    read_file_metadata,
)
from seven.core.integrations.filesystem.operations import (
    FileWriteResult,
    create_directory,
    delete_empty_directory,
    delete_file,
    write_binary_file,
    write_text_file,
)
from seven.core.integrations.filesystem.policy import (
    FileOperation,
    FileSystemAccessPolicy,
)


class FileSystemService:
    """Provide lifecycle-managed, policy-controlled filesystem operations."""

    def __init__(self, policy: FileSystemAccessPolicy) -> None:
        self._policy = policy
        self._is_running = False

    @property
    def name(self) -> str:
        return "filesystem"

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def policy(self) -> FileSystemAccessPolicy:
        return self._policy

    def start(self) -> None:
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False

    def _ensure_running(self) -> None:
        if not self._is_running:
            raise RuntimeError("Filesystem service is not running.")

    def get_metadata(self, path: str | Path) -> FileMetadata:
        self._ensure_running()
        authorized_path = self._policy.authorize(path, FileOperation.READ)
        metadata = read_file_metadata(authorized_path)
        self._policy.validate_file_size(metadata.size_bytes)
        return metadata

    def list_directory(
        self,
        path: str | Path,
    ) -> tuple[DirectoryEntry, ...]:
        self._ensure_running()
        authorized_path = self._policy.authorize(path, FileOperation.READ)
        return list_directory(authorized_path)

    def read_text(
        self,
        path: str | Path,
        encoding: str = "utf-8",
    ) -> FileContent:
        self._ensure_running()
        authorized_path = self._policy.authorize(path, FileOperation.READ)
        metadata = read_file_metadata(authorized_path)

        if not metadata.is_file:
            raise IsADirectoryError(authorized_path)

        self._policy.validate_file_size(metadata.size_bytes)
        return read_text_file(authorized_path, encoding=encoding)

    def read_binary(self, path: str | Path) -> BinaryFileContent:
        self._ensure_running()
        authorized_path = self._policy.authorize(path, FileOperation.READ)
        metadata = read_file_metadata(authorized_path)

        if not metadata.is_file:
            raise IsADirectoryError(authorized_path)

        self._policy.validate_file_size(metadata.size_bytes)
        return read_binary_file(authorized_path)

    def write_text(
        self,
        path: str | Path,
        content: str,
        encoding: str = "utf-8",
        overwrite: bool = False,
        create_parents: bool = False,
    ) -> FileWriteResult:
        self._ensure_running()
        data_size = len(content.encode(encoding))
        self._policy.validate_file_size(data_size)
        authorized_path = self._policy.authorize(path, FileOperation.WRITE)

        return write_text_file(
            authorized_path,
            content,
            encoding=encoding,
            overwrite=overwrite,
            create_parents=create_parents,
        )

    def write_binary(
        self,
        path: str | Path,
        content: bytes,
        overwrite: bool = False,
        create_parents: bool = False,
    ) -> FileWriteResult:
        self._ensure_running()
        self._policy.validate_file_size(len(content))
        authorized_path = self._policy.authorize(path, FileOperation.WRITE)

        return write_binary_file(
            authorized_path,
            content,
            overwrite=overwrite,
            create_parents=create_parents,
        )

    def create_directory(
        self,
        path: str | Path,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> Path:
        self._ensure_running()
        authorized_path = self._policy.authorize(path, FileOperation.WRITE)

        return create_directory(
            authorized_path,
            parents=parents,
            exist_ok=exist_ok,
        )

    def delete(self, path: str | Path) -> Path:
        self._ensure_running()
        authorized_path = self._policy.authorize(path, FileOperation.DELETE)

        if authorized_path in self._policy.allowed_roots:
            raise PermissionError("Deleting an allowed root is forbidden.")

        return delete_file(authorized_path)

    def delete_directory(self, path: str | Path) -> Path:
        self._ensure_running()
        authorized_path = self._policy.authorize(path, FileOperation.DELETE)

        if authorized_path in self._policy.allowed_roots:
            raise PermissionError("Deleting an allowed root is forbidden.")

        return delete_empty_directory(authorized_path)
