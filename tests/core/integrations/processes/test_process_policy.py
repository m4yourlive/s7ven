
"""Tests for process access policy."""

import pytest

from seven.core.integrations.processes.policy import ProcessAccessPolicy


def test_policy_allows_discovery_by_default():
    policy = ProcessAccessPolicy()

    policy.authorize_discovery()


def test_policy_allows_inspection_by_default():
    policy = ProcessAccessPolicy()

    policy.authorize_inspection(1234)


def test_policy_denies_termination_by_default():
    policy = ProcessAccessPolicy()

    with pytest.raises(PermissionError):
        policy.authorize_termination(1234)


def test_policy_allows_termination_when_enabled():
    policy = ProcessAccessPolicy(allow_termination=True)

    policy.authorize_termination(1234)


def test_policy_denies_discovery_when_disabled():
    policy = ProcessAccessPolicy(allow_discovery=False)

    with pytest.raises(PermissionError):
        policy.authorize_discovery()


def test_policy_denies_inspection_when_disabled():
    policy = ProcessAccessPolicy(allow_inspection=False)

    with pytest.raises(PermissionError):
        policy.authorize_inspection(1234)


def test_policy_denies_protected_process_termination():
    policy = ProcessAccessPolicy(
        allow_termination=True,
        protected_pids=frozenset({1234}),
    )

    with pytest.raises(PermissionError):
        policy.authorize_termination(1234)


def test_policy_allows_termination_for_allowlisted_pid():
    policy = ProcessAccessPolicy(
        allow_termination=True,
        allowed_termination_pids=frozenset({1234}),
    )

    policy.authorize_termination(1234)


def test_policy_denies_termination_for_non_allowlisted_pid():
    policy = ProcessAccessPolicy(
        allow_termination=True,
        allowed_termination_pids=frozenset({1234}),
    )

    with pytest.raises(PermissionError):
        policy.authorize_termination(5678)


def test_policy_accepts_any_pid_when_allowlist_is_none():
    policy = ProcessAccessPolicy(allow_termination=True)

    policy.authorize_termination(5678)


@pytest.mark.parametrize("pid", [0, -1])
def test_policy_rejects_invalid_pid(pid):
    policy = ProcessAccessPolicy()

    with pytest.raises(ValueError):
        policy.authorize_inspection(pid)


def test_policy_rejects_non_integer_pid():
    policy = ProcessAccessPolicy()

    with pytest.raises(TypeError):
        policy.authorize_inspection(True)


def test_policy_rejects_invalid_protected_pid():
    with pytest.raises(ValueError):
        ProcessAccessPolicy(
            protected_pids=frozenset({0}),
        )


def test_policy_rejects_non_frozenset_protected_pids():
    with pytest.raises(TypeError):
        ProcessAccessPolicy(
            protected_pids={1234},
        )


def test_policy_rejects_invalid_allowlist():
    with pytest.raises(ValueError):
        ProcessAccessPolicy(
            allowed_termination_pids=frozenset({-1}),
        )


def test_policy_rejects_non_boolean_permissions():
    with pytest.raises(TypeError):
        ProcessAccessPolicy(allow_termination=1)
