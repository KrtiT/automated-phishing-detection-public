"""Prepared saved acceptance guards file ownership, not scientific reconstruction."""

import os
import signal

import pytest
from study_preparation_runner_fixtures import (
    inputs,
    preparation_api,
    preparation_case,
    runner,
)
from test_prepared_internal_completion import prepared_published, verifier
from test_prepared_internal_io_interruptions import arm_call
from test_study_preparation_io_interruptions import signal_after_open

from automated_phishing_detection import (
    execution_receipt,
    source_completion,
    source_runner,
)

__all__ = [
    "inputs",
    "preparation_api",
    "preparation_case",
    "runner",
    "prepared_published",
]


def verification_scope(monkeypatch, stage):
    if stage == "public":
        return "sources.json", None
    if stage == "evidence":
        return "predictions.jsonl", None
    if stage == "entry":
        return "/", arm_call(monkeypatch, source_completion, "_verify_outputs")
    if stage == "check":
        return "/", arm_call(monkeypatch, execution_receipt._Directory, "check")
    calls = []
    monkeypatch.setattr(
        source_runner, "recheck_binding", lambda binding: calls.append(binding)
    )
    return "/", lambda: len(calls) == 2


@pytest.mark.parametrize(
    "stage", ["public", "evidence", "entry", "check", "final_check"]
)
def test_prepared_verifier_closes_interrupted_borrowed_descriptors(
    prepared_published, monkeypatch, stage
):
    case = prepared_published
    target, enabled = verification_scope(monkeypatch, stage)
    with signal_after_open(monkeypatch, target, enabled) as (opened, closed):
        with pytest.raises(KeyboardInterrupt):
            verifier()(
                case.binding,
                case.paths,
                preparation=case.preparation,
                producer_exit_code=0,
            )
        assert len(opened) == 1 and opened[0] in closed
        with pytest.raises(OSError):
            os.fstat(opened[0])


def test_saved_scientific_reconstruction_is_not_signal_deferred(
    prepared_published, monkeypatch
):
    case, continued = prepared_published, []

    def reconstruct(*args, **kwargs):
        signal.raise_signal(signal.SIGINT)
        continued.append(True)

    monkeypatch.setattr(
        source_completion, "reconstruct_internal_evidence_and_population", reconstruct
    )
    with pytest.raises(KeyboardInterrupt):
        verifier()(
            case.binding, case.paths, preparation=case.preparation, producer_exit_code=0
        )
    assert continued == []
