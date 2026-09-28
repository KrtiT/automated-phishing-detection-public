"""Operator authority uses committed policy bytes and the unchanged closed binder."""

from dataclasses import replace
from pathlib import Path

import pytest
import study_execution_fixtures as original
from study_execution_fixtures import execution_case as execution_case
from study_operator_fixtures import (
    effective_profile,
    install_policy,
    policy_bytes,
    seal,
)

from automated_phishing_detection import _study_execution_policy as legacy
from automated_phishing_detection import _study_policy_variants as variants
from automated_phishing_detection._checkpoint_codec import canonical_bytes


def bound(case, monkeypatch):
    install_policy(case, monkeypatch)
    return original.bind(case, seal(effective_profile(case)))


def test_complete_operator_authority_keeps_all_candidates_closed(
    execution_case, monkeypatch
):
    result = bound(execution_case, monkeypatch)
    assert result.base is execution_case.base
    assert result.policy_bytes == policy_bytes()
    assert result.deadlines == legacy.DEADLINES
    assert result.base.protected_evaluation_ready is False
    assert result.external.protected_evaluation_ready is False
    assert result.operational.protected_evaluation_ready is False
    original.api().recheck_study_execution(result)


@pytest.mark.parametrize("change", ["missing", "changed", "symlink", "wrong_name"])
def test_operator_policy_must_be_committed_under_exact_fixed_path(
    execution_case, monkeypatch, change
):
    install_policy(execution_case, monkeypatch)
    previous = original.preflight._git

    def altered(root, *arguments):
        result = previous(root, *arguments)
        if arguments[0] == "cat-file":
            return result + b" " if change == "changed" else result
        if change == "missing":
            return b""
        if change == "symlink":
            return result.replace(b"100644", b"120000")
        return (
            result.replace(b"policy-v2", b"policy-v1")
            if change == "wrong_name"
            else result
        )

    monkeypatch.setattr(original.preflight, "_git", altered)
    with pytest.raises(ValueError):
        original.bind(execution_case, seal(effective_profile(execution_case)))


@pytest.mark.parametrize(
    "slot", ["method", "advisor_requirement_waiver", "profile", "access"]
)
def test_missing_operator_decisions_reject_before_even_public_binding(
    execution_case, monkeypatch, slot
):
    install_policy(execution_case, monkeypatch)
    value = seal(effective_profile(execution_case))
    del value["decisions"][slot]

    def forbidden(*args, **kwargs):
        pytest.fail("incomplete operator decision reached binding")

    monkeypatch.setattr(original.preflight, "bind_execution", forbidden)
    with pytest.raises(ValueError):
        original.bind(execution_case, value)


def test_operator_recheck_rejects_mutated_envelope_or_binding(
    execution_case, monkeypatch
):
    result = bound(execution_case, monkeypatch)
    with pytest.raises(ValueError):
        original.api().recheck_study_execution(
            replace(result, policy_bytes=legacy.policy_bytes())
        )
    execution_case.envelope_path.write_bytes(canonical_bytes({"revoked": True}))
    with pytest.raises(ValueError):
        original.api().recheck_study_execution(result)


def test_operator_binding_projects_paths_without_stat_or_resolve(
    execution_case, monkeypatch
):
    install_policy(execution_case, monkeypatch)
    value = seal(effective_profile(execution_case))
    pin = original.write_envelope(execution_case, value)

    def forbidden(*args, **kwargs):
        pytest.fail("operator authority inspected a private path")

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "stat", forbidden)
        scoped.setattr(Path, "resolve", forbidden)
        result = original.api().bind_study_execution(
            execution_case.root,
            expected_revision=execution_case.base.revision,
            envelope_path=execution_case.envelope_path,
            expected_envelope_sha256=pin,
        )
    assert result.envelope_sha256 == pin


def test_unknown_policy_digest_does_not_select_an_arbitrary_path():
    with pytest.raises(ValueError):
        variants.select_policy("0" * 64)
