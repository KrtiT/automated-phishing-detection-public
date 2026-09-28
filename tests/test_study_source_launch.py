"""A study worker uses live admission, never the standalone launch route."""

from types import SimpleNamespace

import pytest
from test_prepared_internal_process import (
    inputs,
    preparation_api,
    preparation_case,
    process_case,
    runner,
)

from automated_phishing_detection import prepared_internal_runner as internal
from automated_phishing_detection._owned_process_exit import OwnedProcessExit
from automated_phishing_detection._process_support import command_hash
from automated_phishing_detection.owned_worker import WorkerObservation

__all__ = ["inputs", "preparation_api", "preparation_case", "process_case", "runner"]


def observer(calls):
    def observe(command, ledger, role, **extra):
        calls.append((command, ledger, role))
        return WorkerObservation(
            command_hash(command), OwnedProcessExit(77, True, 0), "a" * 64, "b" * 64
        )

    return observe


def test_study_internal_uses_admitted_observer(process_case, monkeypatch):
    case, calls = process_case, []
    admissions = SimpleNamespace(
        authorization=SimpleNamespace(
            base=case.binding,
            envelope_path=case.paths.attempt.parent / "approval.json",
            envelope_sha256="c" * 64,
        )
    )

    monkeypatch.setattr(
        internal, "observe_worker", lambda command: pytest.fail("legacy observer used")
    )
    monkeypatch.setattr(internal, "_observe_admitted", observer(calls), raising=False)
    monkeypatch.setattr(
        internal,
        "verify_prepared_internal_completion_snapshot",
        lambda *args, **kwargs: object(),
    )
    result = internal._run_observed_prepared_internal(
        case.binding,
        case.paths,
        preparation=case.preparation,
        study_admissions=admissions,
    )
    assert result.worker.exit.pid == 77 and len(calls) == 1
    assert calls[0][0][1].endswith("scripts/run_study_child.py")
    assert calls[0][1:] == (admissions, "internal")
