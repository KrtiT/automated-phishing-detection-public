import asyncio
import base64
import json
from dataclasses import replace

import pytest
from external_composition_fixtures import composition_inputs, producer_module


def retained(progress):
    return {
        name: base64.b64decode(content)
        for name, content in progress["retained_checkpoints_base64"].items()
    }


@pytest.mark.parametrize(
    "name",
    [
        "retained-test.jsonl",
        "bindings.json",
        "primary-scores.jsonl",
        "secondary-seed-44.json",
        "all-scores.jsonl",
        "routing.json",
        "monitors.json",
        "predictions.jsonl",
        "secondary.json",
    ],
)
def test_failed_writer_preserves_attempted_bytes_and_never_advances(monkeypatch, name):
    module = producer_module()
    prepared, session, _, _ = composition_inputs(monkeypatch)
    writes = []

    def retain(checkpoint, content):
        writes.append((checkpoint, content))
        if checkpoint == name:
            raise OSError("private output directory")

    with pytest.raises(module.ExternalProducerError) as caught:
        module.produce_external_evidence(prepared, session, retain=retain)
    progress = json.loads(caught.value.progress)
    assert retained(progress) == dict(writes)
    assert progress["failed_checkpoint"] == name
    assert progress["retention_status"] == "failed_or_ambiguous"
    assert writes[-1][0] == name
    assert "private" not in str(caught.value)
    if name in ("retained-test.jsonl", "bindings.json"):
        assert session.evaluation.primary.scorer.urls == []


@pytest.mark.parametrize("phase", ["primary", "replay", "summary"])
def test_phase_error_keeps_predecessors_without_success_marker(monkeypatch, phase):
    module = producer_module()
    prepared, session, _, _ = composition_inputs(monkeypatch)
    writes = {}

    def fail(*args, **kwargs):
        raise ValueError("private failing input")

    if phase == "primary":
        session.evaluation.primary.scorer.fail_at = 2
    elif phase == "replay":
        monkeypatch.setattr(module.external_replay, "replay_external_scores", fail)
    else:
        monkeypatch.setattr(module.external_metrics, "summarize_external", fail)
    with pytest.raises(module.ExternalProducerError) as caught:
        module.produce_external_evidence(
            prepared,
            session,
            retain=lambda name, content: writes.__setitem__(name, content),
        )
    progress = json.loads(caught.value.progress)
    assert retained(progress) == writes
    assert progress["status"] == "failed"
    assert "secondary.json" not in writes
    assert ("all-scores.jsonl" in writes) == (phase != "primary")
    if phase == "primary":
        child = json.loads(progress["phase_progress_json"])
        assert len(child["completed_rows"]) == 2


def test_cancelled_writer_preserves_interruption_identity_and_checkpoints(monkeypatch):
    module = producer_module()
    prepared, session, _, _ = composition_inputs(monkeypatch)
    cancellation = asyncio.CancelledError("private-source")

    def retain(name, content):
        if name == "routing.json":
            raise cancellation

    with pytest.raises(asyncio.CancelledError) as caught:
        module.produce_external_evidence(prepared, session, retain=retain)
    assert caught.value is cancellation
    assert "private" not in str(cancellation)
    progress = json.loads(cancellation.progress)
    assert "all-scores.jsonl" in retained(progress)
    assert "routing.json" in retained(progress)


def test_missing_retained_drift_evidence_rejects_before_primary(monkeypatch):
    module = producer_module()
    prepared, session, _, _ = composition_inputs(monkeypatch)
    session = replace(session, drift=replace(session.drift, public_inputs=()))
    with pytest.raises(module.ExternalProducerError):
        module.produce_external_evidence(prepared, session)
    assert session.evaluation.primary.scorer.urls == []
