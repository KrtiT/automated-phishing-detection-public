"""Twelve-column restoration binds exact member, row, and completion identities."""

from hashlib import sha256

import pytest
from saved_external_restore_fixtures import (
    forbid_execution,
    phase_inputs,
    restore_module,
    rewrite,
)

from automated_phishing_detection import _external_secondary_checkpoints as checkpoints
from automated_phishing_detection.evaluation_producer import _json_bytes
from automated_phishing_detection.saved_evidence import SavedEvidenceError


@pytest.mark.parametrize("count", [0, 1, 5])
def test_restores_every_completed_column_without_scoring_or_io(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    module = restore_module("columns")
    phase = phase_inputs(monkeypatch, count)
    forbid_execution(monkeypatch)
    actual = module.restore_secondary(phase.outputs, phase.primary, phase.bindings)
    assert actual == phase.secondary


def test_saved_binding_projection_preserves_existing_producer_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phase = phase_inputs(monkeypatch)
    assert hasattr(checkpoints, "project_member_bindings"), "pure projection missing"
    projected = checkpoints.project_member_bindings(phase.bindings["secondary"])
    assert projected == checkpoints.member_bindings(phase.bound)


@pytest.mark.parametrize("index", range(7))
def test_projection_preserves_all_tabular_fields(
    monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    phase = phase_inputs(monkeypatch, 0)
    member = phase.bound.tabular[index]
    expected = {
        "kind": "tabular",
        "name": member.name,
        "artifact_sha256": member.artifact_sha256,
        "threshold": member.threshold,
        "accepted_report_sha256": dict(phase.bound.report_hashes)["tabular"],
    }
    actual = checkpoints.project_member_bindings(phase.bindings["secondary"])[index]
    assert _json_bytes(actual) == _json_bytes(expected)


@pytest.mark.parametrize("index", range(5))
def test_projection_preserves_all_seed_fields(
    monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    phase = phase_inputs(monkeypatch, 0)
    member = phase.bound.seeds[index]
    expected = {
        "kind": "seed",
        "seed": member.seed,
        "weights_sha256": member.weights_sha256,
        "transformer_threshold": member.transformer_threshold,
        "half_width": member.half_width,
        "stage1_threshold": phase.bound.stage1_threshold,
        "reuses_primary": member.reuses_primary,
        "vocabulary_sha256": sha256(phase.bound.vocabulary_bytes).hexdigest(),
        "device_type": phase.bound.device_type,
        "accepted_report_sha256": dict(phase.bound.report_hashes)["seeds"],
    }
    actual = checkpoints.project_member_bindings(phase.bindings["secondary"])[7 + index]
    assert _json_bytes(actual) == _json_bytes(expected)


@pytest.mark.parametrize("name", checkpoints.SECONDARY_CHECKPOINTS)
def test_empty_stream_still_requires_every_column(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    module = restore_module("columns")
    phase = phase_inputs(monkeypatch, 0)
    del phase.outputs[name]
    with pytest.raises(SavedEvidenceError):
        module.restore_secondary(phase.outputs, phase.primary, phase.bindings)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("schema_version",), True),
        (("private_field",), "private.invalid"),
        (("primary_scores_sha256",), "0" * 64),
        (("record_ids",), ["private.invalid"] * 5),
        (("binding", "threshold"), 0.125),
        (("binding", "accepted_report_sha256"), "0" * 64),
        (("column", "singleton_calls"), True),
        (("column", "scores", 0, "decision"), True),
        (("column", "scores", 0, "private_field"), "private.invalid"),
        (("column", "scores", 0, "probability"), 10**400),
    ],
)
def test_tabular_column_rejects_forged_envelope_or_shape(
    monkeypatch: pytest.MonkeyPatch, path: tuple, value: object
) -> None:
    module = restore_module("columns")
    phase = phase_inputs(monkeypatch)
    rewrite(phase.outputs, checkpoints.SECONDARY_CHECKPOINTS[0], path, value)
    with pytest.raises(SavedEvidenceError) as caught:
        module.restore_secondary(phase.outputs, phase.primary, phase.bindings)
    assert "private.invalid" not in str(caught.value)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("binding", "reuses_primary"), False),
        (("binding", "stage1_threshold"), 0.125),
        (("column", "seed"), True),
        (("column", "transformer_singleton_calls"), 5),
        (("column", "reused_primary_transformer_scores"), 0),
        (("column", "scores", 0, "band_selected"), 1),
    ],
)
def test_seed_column_rejects_forged_binding_or_counts(
    monkeypatch: pytest.MonkeyPatch, path: tuple, value: object
) -> None:
    module = restore_module("columns")
    phase = phase_inputs(monkeypatch)
    rewrite(phase.outputs, "secondary-seed-42.json", path, value)
    with pytest.raises(SavedEvidenceError):
        module.restore_secondary(phase.outputs, phase.primary, phase.bindings)


@pytest.mark.parametrize("name", ["secondary-completion.json", "all-scores.jsonl"])
def test_secondary_rejects_mismatched_join_or_receipt(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    module = restore_module("columns")
    phase = phase_inputs(monkeypatch)
    phase.outputs[name] += b"\n"
    with pytest.raises(SavedEvidenceError):
        module.restore_secondary(phase.outputs, phase.primary, phase.bindings)


@pytest.mark.parametrize("name", ["secondary-completion.json", "all-scores.jsonl"])
@pytest.mark.parametrize("mutable_type", [bytearray, memoryview])
def test_secondary_requires_immutable_join_and_receipt(
    monkeypatch: pytest.MonkeyPatch, name: str, mutable_type: type
) -> None:
    module = restore_module("columns")
    phase = phase_inputs(monkeypatch)
    phase.outputs[name] = mutable_type(phase.outputs[name])
    with pytest.raises(SavedEvidenceError):
        module.restore_secondary(phase.outputs, phase.primary, phase.bindings)
