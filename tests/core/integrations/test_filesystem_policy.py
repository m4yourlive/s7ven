
from pathlib import Path

import pytest

from seven.core.integrations.filesystem import (
    FileOperation,
    FileSystemAccessPolicy,
)


def create_policy(root: Path) -> FileSystemAccessPolicy:
    return FileSystemAccessPolicy(
        allowed_roots=(root,),
        allow_read=True,
        allow_write=True,
        allow_delete=False,
        max_file_size_bytes=1024,
    )


def test_policy_authorizes_path_inside_allowed_root(
    tmp_path: Path,
) -> None:
    policy = create_policy(tmp_path)
    target = tmp_path / "document.txt"

    result = policy.authorize(target, FileOperation.READ)

    assert result == target.resolve()


def test_policy_rejects_path_outside_allowed_root(
    tmp_path: Path,
) -> None:
    policy = create_policy(tmp_path / "allowed")
    target = tmp_path / "outside.txt"

    with pytest.raises(PermissionError):
        policy.authorize(target, FileOperation.READ)


def test_policy_rejects_disabled_write_operation(
    tmp_path: Path,
) -> None:
    policy = FileSystemAccessPolicy(
        allowed_roots=(tmp_path,),
        allow_write=False,
    )

    with pytest.raises(PermissionError):
        policy.authorize(
            tmp_path / "document.txt",
            FileOperation.WRITE,
        )


def test_policy_rejects_disabled_delete_operation(
    tmp_path: Path,
) -> None:
    policy = create_policy(tmp_path)

    with pytest.raises(PermissionError):
        policy.authorize(
            tmp_path / "document.txt",
            FileOperation.DELETE,
        )


def test_policy_accepts_file_within_size_limit(
    tmp_path: Path,
) -> None:
    policy = create_policy(tmp_path)

    policy.validate_file_size(1024)


def test_policy_rejects_file_exceeding_size_limit(
    tmp_path: Path,
) -> None:
    policy = create_policy(tmp_path)

    with pytest.raises(PermissionError):
        policy.validate_file_size(1025)


def test_policy_rejects_negative_file_size(
    tmp_path: Path,
) -> None:
    policy = create_policy(tmp_path)

    with pytest.raises(ValueError):
        policy.validate_file_size(-1)


def test_policy_rejects_invalid_size_limit(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        FileSystemAccessPolicy(
            allowed_roots=(tmp_path,),
            max_file_size_bytes=0,
        )


def test_policy_is_immutable(
    tmp_path: Path,
) -> None:
    policy = create_policy(tmp_path)

    with pytest.raises(AttributeError):
        policy.allow_write = False
