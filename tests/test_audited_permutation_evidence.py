"""Disk evidence must agree with pinned inputs without rerunning a fit or score."""

import json
from hashlib import sha256

import numpy as np
import pytest
from test_audited_permutation_member import member as member
from test_audited_permutation_member import supplied as supplied


@pytest.fixture
def completed(member, supplied):
    member.run_reserved_member(**supplied)
    return supplied


def _verify(completed):
    from automated_phishing_detection import audited_permutation_evidence

    return audited_permutation_evidence.verify_completed_member(**completed)


def test_verification_replays_inputs_without_fitting_or_scoring(
    completed, member, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("verifier must not fit or score")

    monkeypatch.setattr(member.secondary_tabular.LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(member.secondary_tabular, "_score_state", forbidden)
    result = _verify(completed)
    assert result["status"] == "evidence_consistent"
    assert result["seed"] == 42
    assert result["training_rows"] == 64
    assert result["validation_rows"] == 12
    assert result["historical_fit_verified"] is False
    assert 0 < result["permuted_label_changes"] < 64


@pytest.mark.parametrize(
    "relative",
    [
        "fit-call-prepared.json",
        "fit-call-returned.json",
        "fit-state.json",
        "evidence/model.json",
        "evidence/fit-linkage.json",
        "evidence/validation-predictions.jsonl",
        "outcome.json",
        "member-started.json",
    ],
)
def test_modified_evidence_is_rejected(completed, relative):
    path = completed["attempt"].directory / relative
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        _verify(completed)


def _reseal(completed, relative, content):
    root = completed["attempt"].directory
    (root / relative).write_bytes(content)
    summary = json.loads(completed["public_path"].read_bytes())
    outcome = json.loads((root / "outcome.json").read_bytes())
    name = relative.removeprefix("evidence/")
    if relative.startswith("evidence/"):
        summary["private_sha256"][name] = sha256(content).hexdigest()
        outcome["private_sha256"][name] = sha256(content).hexdigest()
    else:
        summary["receipt_sha256"][name] = sha256(content).hexdigest()
        linkage = json.loads((root / "evidence/fit-linkage.json").read_bytes())
        linkage["receipt_sha256"][name] = sha256(content).hexdigest()
        linked = (
            json.dumps(linkage, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        (root / "evidence/fit-linkage.json").write_bytes(linked)
        summary["private_sha256"]["fit-linkage.json"] = sha256(linked).hexdigest()
        outcome["private_sha256"]["fit-linkage.json"] = sha256(linked).hexdigest()
    public = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    completed["public_path"].write_bytes(public)
    outcome["public_summary_sha256"] = sha256(public).hexdigest()
    (root / "outcome.json").write_bytes(
        json.dumps(outcome, sort_keys=True, separators=(",", ":")).encode()
    )


@pytest.mark.parametrize(
    "field",
    [
        "ordered_record_ids",
        "consumed_labels",
        "source_features",
        "scaled_features",
        "parameters",
    ],
)
def test_rehashed_but_wrong_fit_receipt_is_rejected(completed, field):
    root = completed["attempt"].directory
    prepared = json.loads((root / "fit-call-prepared.json").read_bytes())
    if field == "ordered_record_ids":
        prepared[field] = prepared[field][::-1]
    elif field == "consumed_labels":
        prepared[field][0] = 1 - prepared[field][0]
        prepared["consumed_labels_sha256"] = sha256(
            np.asarray(prepared[field], dtype="<i8").tobytes(order="C")
        ).hexdigest()
    elif field == "parameters":
        prepared[field]["C"] = 99.0
    else:
        prepared[field]["sha256"] = "0" * 64
    content = (
        json.dumps(prepared, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    _reseal(completed, "fit-call-prepared.json", content)
    returned = {
        "schema_version": 1,
        "event": "fit_call_returned",
        "prepared_receipt_sha256": sha256(content).hexdigest(),
    }
    _reseal(
        completed,
        "fit-call-returned.json",
        (json.dumps(returned, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    with pytest.raises(ValueError):
        _verify(completed)


def test_rehashed_wrong_prediction_identity_is_rejected(completed):
    root = completed["attempt"].directory
    rows = [
        json.loads(line)
        for line in (root / "evidence/validation-predictions.jsonl")
        .read_bytes()
        .splitlines()
    ]
    rows[0]["record_id"] = rows[1]["record_id"]
    content = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    _reseal(completed, "evidence/validation-predictions.jsonl", content)
    with pytest.raises(ValueError):
        _verify(completed)


def test_wrong_seed_and_input_pins_are_rejected(completed):
    completed["seed"] = 43
    with pytest.raises(ValueError):
        _verify(completed)


@pytest.mark.parametrize(
    "relative", ["finalize.claim", "outcome.json", "fit-call-returned.json"]
)
def test_boolean_schema_version_is_not_integer_one(completed, relative):
    path = completed["attempt"].directory / relative
    record = json.loads(path.read_bytes())
    record["schema_version"] = True
    content = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    if relative == "fit-call-returned.json":
        _reseal(completed, relative, content + b"\n")
    else:
        path.write_bytes(content)
    with pytest.raises(ValueError):
        _verify(completed)
