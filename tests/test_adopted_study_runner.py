"""The adopted root preserves whole-study holds and durable admission ordering."""

import json

import adopted_study_fixtures as fixtures
import pytest
from study_run_record_fixtures import (
    REQUIREMENTS,
    capacity,
    isolated_shortage,
    prepared,
)

__all__ = ["prepared"]


@pytest.mark.parametrize("requirement", REQUIREMENTS)
def test_each_shortage_retains_adopted_hold_without_source_admission(
    tmp_path, prepared, monkeypatch, requirement
):
    case = fixtures.setup(
        tmp_path, prepared, monkeypatch, isolated_shortage(prepared, requirement)
    )
    result = fixtures.execute(case)
    public = json.loads(result.snapshot.payload("public-summary.json"))
    assert public["protocol"] == "adopted-study-root-v1"
    assert public["status"] == "whole_study_hold"
    assert len(result.snapshot.payloads) == 10
    assert len(result.cells) == 125
    assert all(slot.status == "unattempted" for slot in result.cells)
    accounting = json.loads(result.snapshot.payload("attempt/study-accounting.json"))
    assert accounting["authorization_ledger"]["admissions"] == []
    assert not case.paths.internal.attempt.exists()
    assert not case.paths.external.attempt.exists()


def test_pending_barrier_bytes_never_issue_admission(tmp_path, prepared, monkeypatch):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    retention = fixtures.api("study_root_retention")
    original = retention._RootWriter.append

    def fail(writer, name, content):
        if name == "prediction-barrier.json":
            writer._contents[name] = content
            raise OSError("invented append failure")
        return original(writer, name, content)

    monkeypatch.setattr(retention._RootWriter, "append", fail)
    with pytest.raises(case.module.StudyRunError) as caught:
        fixtures.execute(case)
    retained = case.module.adopted_study_failure(caught.value)
    ledger = json.loads(retained.authorization_ledger)
    assert ledger["barrier_sha256"] is None
    assert ledger["admissions"] == []
    assert not case.paths.public_summary.exists()


def test_root_intent_pins_profile_and_complete_envelope(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    result = fixtures.execute(case)
    intent = json.loads(result.snapshot.payload("attempt/study-intent.json"))
    assert (
        intent["execution"]["study_profile_sha256"] == case.authorization.profile_sha256
    )
    assert (
        intent["execution"]["adoption_envelope_sha256"]
        == case.authorization.envelope_sha256
    )
