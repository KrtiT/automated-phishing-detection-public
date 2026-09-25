"""Prepared scoring has no source-path fallback or access override."""

from dataclasses import fields
from pathlib import Path

import pytest

from automated_phishing_detection import _external_source_records as records
from automated_phishing_detection import external_source_runner as runner
from automated_phishing_detection.execution_preflight import ExecutionBinding


def test_prepared_paths_exclude_original_sources():
    assert hasattr(records, "PreparedExternalRunPaths")
    assert [member.name for member in fields(records.PreparedExternalRunPaths)] == [
        "preparation",
        "artifacts",
        "secondary_artifacts",
        "drift_artifacts",
        "attempt",
        "public_summary",
    ]


@pytest.mark.parametrize("internal_ready", [False, True])
def test_each_gate_precedes_all_supplied_preparation_paths(monkeypatch, internal_ready):
    assert callable(getattr(runner, "run_prepared_external_evaluation", None))
    binding = ExecutionBinding(Path("/invented"), "c" * 40, "d" * 64, (), "{}")
    monkeypatch.setattr(runner, "bind_execution", lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        ExecutionBinding,
        "protected_evaluation_ready",
        property(lambda _: internal_ready),
    )
    profile = type(
        "ClosedProfile", (), {"protected_evaluation_ready": not internal_ready}
    )()
    monkeypatch.setattr(
        runner.body, "resolve_external_source_profile", lambda _: profile
    )
    with pytest.raises(runner.ExternalSourceExecutionError, match="freeze_incomplete"):
        runner.run_prepared_external_evaluation(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            paths=object(),
            internal_transport=object(),
            expected_handoff_sha256=object(),
            expected_preparation_reservation_sha256=object(),
            expected_preparation_completion_sha256=object(),
        )


@pytest.mark.parametrize("internal_ready", [False, True])
def test_parent_pair_has_two_independently_closed_gates(monkeypatch, internal_ready):
    from automated_phishing_detection import prepared_external_process as pair

    binding = ExecutionBinding(Path("/invented"), "c" * 40, "d" * 64, (), "{}")
    monkeypatch.setattr(pair, "bind_execution", lambda *args, **kwargs: binding)
    monkeypatch.setattr(
        ExecutionBinding,
        "protected_evaluation_ready",
        property(lambda _: internal_ready),
    )
    profile = type(
        "ClosedProfile", (), {"protected_evaluation_ready": not internal_ready}
    )()
    monkeypatch.setattr(
        pair.process, "resolve_external_source_profile", lambda _: profile
    )
    with pytest.raises(pair.ExternalSourceExecutionError, match="freeze_incomplete"):
        pair.run_prepared_internal_external_process(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            internal_paths=object(),
            external_paths=object(),
            expected_preparation_reservation_sha256=object(),
            expected_preparation_completion_sha256=object(),
        )
