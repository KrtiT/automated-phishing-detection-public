"""Preparation and primary restoration reject malformed retained checkpoints."""

from dataclasses import replace

import pytest
from saved_external_restore_fixtures import (
    forbid_execution,
    phase_inputs,
    restore_module,
    rewrite,
    rewrite_row,
)

from automated_phishing_detection.saved_evidence import SavedEvidenceError


@pytest.mark.parametrize("count", [0, 1, 5])
def test_restores_preparation_and_primary_without_scoring_or_io(
    monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    module = restore_module("inputs")
    phase = phase_inputs(monkeypatch, count)
    forbid_execution(monkeypatch)
    prepared = module.restore_preparation(phase.outputs)
    primary = module.restore_primary(phase.outputs, prepared, phase.bindings)
    assert prepared == phase.prepared
    assert primary == phase.primary
    assert prepared.quarantine[0].reason_codes == ("invalid_or_missing_url",)
    if count == 5:
        assert prepared.retained[-1].is_phishing is None


@pytest.mark.parametrize("content", [b"\n", b"{}", b"{}\n\n", b'{"x":1,"x":2}\n'])
@pytest.mark.parametrize("name", ["retained-test.jsonl", "quarantine.jsonl"])
def test_preparation_rejects_noncanonical_or_malformed_jsonl(
    monkeypatch: pytest.MonkeyPatch, name: str, content: bytes
) -> None:
    module = restore_module("inputs")
    phase = phase_inputs(monkeypatch)
    phase.outputs[name] = content
    with pytest.raises(SavedEvidenceError):
        module.restore_preparation(phase.outputs)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("private_field",), "private.invalid"),
        (("file_position",), True),
        (("is_phishing",), True),
        (("role",), "private.invalid"),
    ],
)
def test_preparation_rejects_exact_field_and_type_mutations(
    monkeypatch: pytest.MonkeyPatch, path: tuple, value: object
) -> None:
    module = restore_module("inputs")
    phase = phase_inputs(monkeypatch)
    rewrite_row(phase.outputs, "retained-test.jsonl", path, value)
    with pytest.raises(SavedEvidenceError) as caught:
        module.restore_preparation(phase.outputs)
    assert "private.invalid" not in str(caught.value)


@pytest.mark.parametrize("name", ["retained-test.jsonl", "inventory.json"])
def test_preparation_requires_even_empty_inventory_members(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    module = restore_module("inputs")
    phase = phase_inputs(monkeypatch, 0)
    del phase.outputs[name]
    with pytest.raises(SavedEvidenceError):
        module.restore_preparation(phase.outputs)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("primary", "features"), "private.invalid"),
        (("primary", "inference_counts", "completed_requests"), True),
        (("primary", "private_field"), "private.invalid"),
        (("record", "record_id"), "private.invalid"),
        (("primary", "transformer_probability"), 10**400),
    ],
)
def test_primary_rejects_score_shape_and_identity_mutations(
    monkeypatch: pytest.MonkeyPatch, path: tuple, value: object
) -> None:
    module = restore_module("inputs")
    phase = phase_inputs(monkeypatch)
    rewrite_row(phase.outputs, "primary-scores.jsonl", path, value)
    with pytest.raises(SavedEvidenceError) as caught:
        module.restore_primary(phase.outputs, phase.prepared, phase.bindings)
    assert "private.invalid" not in str(caught.value)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("thresholds", "transformer"), 0.125),
        (("row_count",), True),
        (("inference_counts", "failed_requests"), 1),
        (("primary_scores_sha256",), "0" * 64),
        (("private_field",), "private.invalid"),
    ],
)
def test_primary_requires_exact_bound_completion_receipt(
    monkeypatch: pytest.MonkeyPatch, path: tuple, value: object
) -> None:
    module = restore_module("inputs")
    phase = phase_inputs(monkeypatch)
    rewrite(phase.outputs, "primary-completion.json", path, value)
    with pytest.raises(SavedEvidenceError):
        module.restore_primary(phase.outputs, phase.prepared, phase.bindings)


def test_primary_requires_complete_prepared_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = restore_module("inputs")
    phase = phase_inputs(monkeypatch)
    prepared = replace(
        phase.prepared, retained=tuple(reversed(phase.prepared.retained))
    )
    with pytest.raises(SavedEvidenceError):
        module.restore_primary(phase.outputs, prepared, phase.bindings)
