"""Actual owned internal scoring precedes external handoff and actual external exit."""

import json

import pytest
from prepared_external_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    prepared_case,
    runner,
)
from prepared_external_pair_fixtures import configure_pair, internal_paths, run_pair

from automated_phishing_detection import _study_preparation_files as files
from automated_phishing_detection import study_preparation_transport as transport

__all__ = ["inputs", "preparation_api", "preparation_case", "prepared_case", "runner"]


def test_both_actual_children_complete_with_original_inputs_absent(
    prepared_case, inputs, monkeypatch
):
    case = prepared_case
    paths = internal_paths(case, inputs)
    configure_pair(case, monkeypatch)
    result = run_pair(case, paths)
    assert (
        result.internal.worker.exit.exit_code
        == result.external.worker.exit.exit_code
        == 0
    )
    assert result.internal.worker.exit.pid != result.external.worker.exit.pid
    handoff = json.loads(result.handoff.handoff_bytes)
    assert handoff["worker"]["command_sha256"] == result.internal.worker.command_sha256
    assert len(result.internal.snapshot.payloads) == 35
    assert len(result.external.snapshot.payloads) == 76


def test_final_preparation_mutation_rejects_but_retains_both_actual_children(
    prepared_case, inputs, monkeypatch
):
    case = prepared_case
    paths = internal_paths(case, inputs)
    configure_pair(case, monkeypatch)
    original = transport.files.check

    def recheck(*args):
        if case.paths.public_summary.exists():
            raise files.StudyPreparationError("invented_final_rejection")
        return original(*args)

    monkeypatch.setattr(transport.files, "check", recheck)
    with pytest.raises(transport.StudyPreparationTransportError) as caught:
        run_pair(case, paths)
    assert caught.value.source_internal.worker.exit.exit_code == 0
    failure = caught.value.external_failure
    assert failure.worker.exit.exit_code == 0
    assert len(failure.candidate_snapshot.payloads) == 76
    assert failure.stage == "preparation_finalization"
