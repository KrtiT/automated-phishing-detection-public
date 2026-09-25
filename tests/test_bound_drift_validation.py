"""Public accepted records and invented private-byte boundaries only."""

import json
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_bound_drift import drift as drift
from test_bound_drift import prepared as prepared

from automated_phishing_detection import bound_drift, secondary_development

ROOT = Path(__file__).resolve().parents[1]


def public_fixture():
    profile = json.loads(
        (ROOT / "data/execution-binding-contract-v3.json").read_bytes()
    )
    binding = SimpleNamespace(
        root=ROOT, source_hashes=tuple(profile["public_file_sha256"].items())
    )
    contents = bound_drift._public_inputs(binding)
    report = json.loads(contents[bound_drift._REPORT])
    pins = secondary_development.DevelopmentPins(
        **report["completion"]["execution"]["pins"]
    )
    models = SimpleNamespace(
        artifact_hashes=(
            ("logistic-l1.json", pins.logistic_l1_artifact_sha256),
            ("gmm.json", pins.gmm_artifact_sha256),
        )
    )
    return binding, contents, report, pins, models


def test_real_public_accepted_chain_works_with_closed_v3():
    binding, contents, report, pins, models = public_fixture()
    assert bound_drift._pins(report, binding, models) == pins
    drift_summary, reference_hash, audit_hash = (
        bound_drift.seed_probe_execution._retained_drift(report, pins)
    )
    assert json.loads(drift_summary)["private_sha256"] == {
        "training-reference.json": reference_hash,
        "validation-audit.json": audit_hash,
    }
    assert (
        bound_drift._preparation(
            contents[bound_drift._SOURCE],
            contents[bound_drift._PREPARATION],
            dict(binding.source_hashes)[bound_drift._SOURCE],
            pins,
        )
        == json.loads(contents[bound_drift._PREPARATION])["splits"]["train"][
            "row_count"
        ]
    )


@pytest.mark.parametrize(
    "field",
    [
        "train_sha256",
        "validation_sha256",
        "source_csv_sha256",
        "suffix_rules_sha256",
    ],
)
def test_real_public_preparation_rejects_changed_development_pins(field):
    binding, contents, _, pins, _ = public_fixture()
    with pytest.raises(bound_drift.BoundDriftError):
        bound_drift._preparation(
            contents[bound_drift._SOURCE],
            contents[bound_drift._PREPARATION],
            dict(binding.source_hashes)[bound_drift._SOURCE],
            replace(pins, **{field: "0" * 64}),
        )


@pytest.mark.parametrize(
    "field",
    [
        "preparation_summary_sha256",
        "baseline_contract_sha256",
        "gmm_contract_sha256",
        "logistic_l1_artifact_sha256",
        "gmm_artifact_sha256",
    ],
)
def test_real_public_report_pins_must_match_bound_models_and_profile(field):
    binding, _, report, pins, models = public_fixture()
    report["completion"]["execution"]["pins"] = asdict(
        replace(pins, **{field: "0" * 64})
    )
    with pytest.raises(
        bound_drift.BoundDriftError, match="accepted_drift_binding_mismatch"
    ):
        bound_drift._pins(report, binding, models)


def test_missing_private_input_is_sanitized(drift, prepared):
    prepared.paths.training_reference.unlink()
    with pytest.raises(
        drift.BoundDriftError, match="^invalid_bound_drift_evidence$"
    ) as failure:
        drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
    assert failure.value.__suppress_context__
    assert prepared.events == [
        "accepted_report",
        "public_preparation",
        "accepted_models",
    ]


def test_private_symlink_is_not_followed(drift, prepared, tmp_path):
    original = prepared.paths.training_reference
    original.unlink()
    target = tmp_path / "private-sensitive"
    target.write_bytes(prepared.reference_bytes)
    original.symlink_to(target)
    with pytest.raises(drift.BoundDriftError, match="^invalid_bound_drift_evidence$"):
        drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
    assert "retained_reference" not in prepared.events


def test_private_reads_are_single_pass(drift, prepared, monkeypatch):
    seen = []
    read = drift._read_file_once

    def record(path):
        seen.append(path)
        return read(path)

    monkeypatch.setattr(drift, "_read_file_once", record)
    drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
    assert seen == [prepared.paths.training_reference, prepared.paths.validation_audit]


def test_count_must_match_loaded_primary(drift, prepared):
    prepared.retained.psi.training_row_count += 1
    with pytest.raises(drift.BoundDriftError, match="primary_drift_state_mismatch"):
        drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)


def test_accepted_state_failure_precedes_any_private_access(
    drift, prepared, monkeypatch
):
    def reject(*args):
        raise ValueError("private-sensitive value")

    monkeypatch.setattr(drift.secondary_development, "_accepted_states", reject)
    monkeypatch.setattr(
        drift,
        "_read_file_once",
        lambda _: pytest.fail("private access"),
    )
    with pytest.raises(drift.BoundDriftError, match="^invalid_bound_drift_evidence$"):
        drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)


def test_loader_receives_exact_accepted_snapshot_hashes(drift, prepared, monkeypatch):
    def load(reference, audit, **kwargs):
        assert kwargs["expected_reference_sha256"] == sha256(reference).hexdigest()
        assert kwargs["expected_audit_sha256"] == sha256(audit).hexdigest()
        assert kwargs["expected_drift_summary"] == {}
        return prepared.retained

    monkeypatch.setattr(drift.retained_drift, "load_retained_drift_reference", load)
    result = drift.load_bound_drift(prepared.binding, prepared.paths, prepared.models)
    assert "reference fixture" not in repr(result)
