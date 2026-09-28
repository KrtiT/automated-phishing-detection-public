"""External observation binds the live predecessor instead of a standalone CLI."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from automated_phishing_detection import external_source_process as external
from automated_phishing_detection._owned_process_exit import OwnedProcessExit
from automated_phishing_detection._process_support import command_hash
from automated_phishing_detection.owned_worker import WorkerObservation


def fixture():
    binding = SimpleNamespace(root=Path("/invented/repo"), revision="a" * 40)
    authorization = SimpleNamespace(
        base=binding,
        envelope_path=Path("/invented/approval.json"),
        envelope_sha256="b" * 64,
    )
    return (
        SimpleNamespace(binding=binding, preparation=object(), handoff=object()),
        SimpleNamespace(
            directory=Path("/invented/handoff"), expected_handoff_sha256="c" * 64
        ),
        SimpleNamespace(authorization=authorization),
    )


def test_external_uses_live_admission_and_retains_same_expected_preparation(
    monkeypatch,
):
    state, transport, admissions = fixture()
    seen = []

    def observe(command, ledger, role, **extra):
        seen.append((command, ledger, role, extra))
        return WorkerObservation(
            command_hash(command), OwnedProcessExit(77, True, 0), "a" * 64, "b" * 64
        )

    def verify(*args, **kwargs):
        assert kwargs["expected_preparation"] is state.preparation
        assert kwargs["command"] is state.command
        return "invented_snapshot"

    monkeypatch.setattr(
        external, "observe_worker", lambda command: pytest.fail("legacy observer used")
    )
    monkeypatch.setattr(external, "_observe_admitted", observe, raising=False)
    monkeypatch.setattr(external, "verify_external_completion_snapshot", verify)
    external._observe_and_verify(
        state, object(), transport, study_admissions=admissions
    )
    assert state.snapshot == "invented_snapshot"
    assert seen[0][0][1].endswith("scripts/run_study_child.py")
    assert seen[0][1:] == (admissions, "external", {"predecessor_sha256": "c" * 64})
