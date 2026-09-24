"""Invented retained snapshots only; no source rows, model reads or fitting."""

import copy
import importlib.util
import inspect
import json
import os
from dataclasses import FrozenInstanceError, asdict, replace
from hashlib import sha256

import pytest
from test_transformer_pipeline import SOURCE_SHA256, _preparation_summary

from automated_phishing_detection import (
    development_completion,
    gmm_monitor,
    secondary_development,
    secondary_drift,
)


def _bytes(value):
    return gmm_monitor._canonical_json_bytes(value)


def _public_bytes(value):
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _snapshots(*, training_count=256, constant=False, validation_count=640):
    # Metadata declarations are invented, not records or partition content.
    summary = _preparation_summary(
        [{"is_phishing": index % 2} for index in range(training_count)],
        [{"is_phishing": index % 2} for index in range(validation_count)],
        b"invented train identity",
        b"invented validation identity",
    )
    preparation = _public_bytes(summary)
    pins = secondary_development.DevelopmentPins(
        train_sha256=summary["output_hashes"]["train.jsonl"],
        validation_sha256=summary["output_hashes"]["validation.jsonl"],
        source_csv_sha256=SOURCE_SHA256,
        preparation_summary_sha256=sha256(preparation).hexdigest(),
        suffix_rules_sha256=summary["declared_sources"]["public_suffix_list"]["sha256"],
        logistic_l1_artifact_sha256="a" * 64,
        baseline_contract_sha256="b" * 64,
        gmm_artifact_sha256="c" * 64,
        gmm_contract_sha256="d" * 64,
    )
    identities = [
        f"phiusiil-row-v1:{SOURCE_SHA256}:{index + 1:016x}"
        for index in range(training_count)
    ]
    domains = [f"train-{index}.example" for index in range(training_count)]
    selected = sorted(
        range(training_count),
        key=lambda index: (
            sha256(
                (secondary_drift._REFERENCE_PREFIX + domains[index]).encode("ascii")
            ).digest(),
            domains[index],
        ),
    )[:256]
    reason = (
        "fewer_than_256_training_domains"
        if training_count < 256
        else "no_positive_reference_distance"
        if constant
        else None
    )

    def feature(edges, value, counts):
        return {
            "internal_edges": edges,
            "constant": value,
            "training_counts": counts,
            "training_proportions": [
                (count + 0.5) / (training_count + 0.5 * len(counts)) for count in counts
            ],
        }

    features = [feature([], 0.0, [0, training_count, 0]) for _ in range(26)]
    if not constant:
        features[0] = feature(
            [0.0, 0.5, 1.0],
            None,
            [0, training_count // 2, 0, training_count // 2],
        )
    reference = {
        "schema_version": 1,
        "contract_id": "secondary-development-v1",
        "input_hashes": asdict(pins),
        "training_record_ids": identities,
        "training_domains": domains,
        "scaler": {"mean": [0.0] * 26, "scale": [1.0] * 26},
        "portable_state_sha256": "e" * 64,
        "validation_declaration_sha256": sha256(
            _bytes(summary["splits"]["validation"])
        ).hexdigest(),
        "mmd": {
            "values": [
                [0.0 if constant else float(index % 2)] + [0.0] * 25
                for index in selected
            ],
            "domains": [domains[index] for index in selected],
            "stable_ids": [identities[index] for index in selected],
            "bandwidth_squared": None if reason else 1.0,
            "reason": reason,
        },
        "psi": {
            "features": features,
            "training_row_count": training_count,
            "reason": None,
        },
    }
    validation_ids = [
        f"phiusiil-row-v1:{SOURCE_SHA256}:{index + 1000:016x}"
        for index in range(validation_count)
    ]
    validation_domains = [
        f"validation-{index}.example" for index in range(validation_count)
    ]
    allocation = gmm_monitor.allocate_validation_domains(validation_domains)
    streams = {}
    for stream, positions in allocation.items():
        ends = list(range(256, len(positions) + 1, 64))
        entry = {
            "input_row_positions": list(positions),
            "record_ids": [validation_ids[index] for index in positions],
            "domains": [validation_domains[index] for index in positions],
            "window_end_positions": ends,
        }
        for method in ("mmd", "psi"):
            unavailable = (
                "no_complete_256_row_window"
                if not ends
                else reason
                if method == "mmd"
                else None
            )
            scores = (
                []
                if unavailable
                else ([0.25, 0.75] if stream == "calibration" else [0.1, 1.0])
            )
            entry[method] = {
                "window_end_positions": ends,
                "scores": scores,
                "feature_scores": [[score] + [0.0] * 25 for score in scores]
                if method == "psi"
                else [],
                "reason": unavailable,
            }
        streams[stream] = entry
    methods, results = {}, {}
    for method in ("mmd", "psi"):
        calibration = secondary_drift.DriftWindowScores(
            tuple(streams["calibration"][method]["window_end_positions"]),
            tuple(streams["calibration"][method]["scores"]),
            reason=streams["calibration"][method]["reason"],
        )
        audit = secondary_drift.DriftWindowScores(
            tuple(streams["audit"][method]["window_end_positions"]),
            tuple(streams["audit"][method]["scores"]),
            reason=streams["audit"][method]["reason"],
        )
        methods[method], results[method] = secondary_development._method_result(
            calibration, audit, reference[method]["reason"]
        )
    audit = {
        "schema_version": 1,
        "contract_id": "secondary-development-v1",
        "training_reference_sha256": sha256(_bytes(reference)).hexdigest(),
        "input_hashes": asdict(pins),
        "window_length": 256,
        "window_stride": 64,
        "streams": streams,
        "results": results,
    }
    available = sum(method["status"] == "estimated" for method in methods.values())
    public = {
        "schema_version": 1,
        "contract_id": "secondary-development-v1",
        "status": "completed_development_validation"
        if available == 2
        else ("partially_estimable" if available else "not_estimable"),
        "analysis_stage": "development_validation_only",
        "analysis_role": "secondary_descriptive_only",
        "protected_evaluation_authorized": False,
        "input_binding": "caller_supplied_expected_pins_not_authorization",
        "input_hashes": asdict(pins),
        "input_counts": {
            "training_rows": training_count,
            "training_domain_count": training_count,
            "validation_rows": validation_count,
            "validation_domain_count": validation_count,
            **{f"{name}_rows": len(indices) for name, indices in allocation.items()},
        },
        "methods": methods,
        "private_sha256": {},
    }
    return _rebind(reference, audit, public, preparation, pins)


def _rebind(reference, audit, public, preparation, pins):
    reference_bytes = _bytes(reference)
    audit["training_reference_sha256"] = sha256(reference_bytes).hexdigest()
    audit_bytes = _bytes(audit)
    public["private_sha256"] = {
        "training-reference.json": sha256(reference_bytes).hexdigest(),
        "validation-audit.json": sha256(audit_bytes).hexdigest(),
    }
    return {
        "training_reference": reference_bytes,
        "validation_audit": audit_bytes,
        "expected_reference_sha256": sha256(reference_bytes).hexdigest(),
        "expected_audit_sha256": sha256(audit_bytes).hexdigest(),
        "pins": pins,
        "preparation_summary": preparation,
        "expected_drift_summary": public,
    }


@pytest.fixture
def loader():
    from automated_phishing_detection import retained_drift

    return retained_drift


def test_retained_snapshot_loader_is_available():
    assert importlib.util.find_spec("automated_phishing_detection.retained_drift")


# SP-CORR-01: exact hash-bound producer-format public JSON is accepted.
def test_pretty_public_preparation_json_is_accepted(loader):
    data = _snapshots()
    assert data["preparation_summary"].startswith(b"{\n  ")
    result = loader.load_retained_drift_reference(**data)
    assert result.preparation_summary_sha256 == data["pins"].preparation_summary_sha256


def test_load_returns_immutable_saved_state_and_original_audit_identities(loader):
    data = _snapshots()
    before = copy.deepcopy(data)
    result = loader.load_retained_drift_reference(**data)
    saved = json.loads(data["training_reference"])
    audit = json.loads(data["validation_audit"])
    assert data == before
    assert result.pins == data["pins"]
    assert result.scaler_mean == tuple(saved["scaler"]["mean"])
    assert result.scaler_scale == tuple(saved["scaler"]["scale"])
    assert result.portable_state_sha256 == saved["portable_state_sha256"]
    assert result.mmd == secondary_drift.MMDReference(
        tuple(tuple(row) for row in saved["mmd"]["values"]),
        tuple(saved["mmd"]["domains"]),
        tuple(saved["mmd"]["stable_ids"]),
        saved["mmd"]["bandwidth_squared"],
        saved["mmd"]["reason"],
    )
    assert result.psi.training_row_count == 256
    assert all(
        isinstance(feature.internal_edges, tuple) for feature in result.psi.features
    )
    for method in ("mmd", "psi"):
        assert getattr(
            result, f"{method}_calibration"
        ) == secondary_drift.DriftCalibration(**audit["results"][method]["calibration"])
    assert result.audit_validation_positions == tuple(
        audit["streams"]["audit"]["input_row_positions"]
    )
    assert result.audit_record_ids == tuple(audit["streams"]["audit"]["record_ids"])
    assert result.audit_domains == tuple(audit["streams"]["audit"]["domains"])
    assert (
        tuple(
            result.validation_record_ids[index]
            for index in result.audit_validation_positions
        )
        == result.audit_record_ids
    )
    assert result.training_reference_sha256 == data["expected_reference_sha256"]
    assert result.validation_audit_sha256 == data["expected_audit_sha256"]
    assert result.preparation_summary_sha256 == data["pins"].preparation_summary_sha256
    assert not isinstance(result, secondary_development.TrainingReference)
    assert not hasattr(result, "_marker")
    with pytest.raises(FrozenInstanceError):
        result.scaler_scale = ()
    with pytest.raises(FrozenInstanceError):
        result.psi.features[0].constant = 1.0


@pytest.mark.parametrize(
    ("options", "reason"),
    [
        ({"training_count": 128}, "fewer_than_256_training_domains"),
        ({"constant": True}, "no_positive_reference_distance"),
        ({"validation_count": 128}, "no_complete_256_row_window"),
    ],
)
def test_unavailable_boundaries_remain_none_with_original_reason(
    loader, options, reason
):
    result = loader.load_retained_drift_reference(**_snapshots(**options))
    assert result.mmd_calibration.threshold is None
    assert result.mmd_calibration.reason == reason
    if options.get("validation_count"):
        assert result.psi_calibration.threshold is None
        assert result.psi_calibration.reason == reason
    else:
        assert result.mmd.reason == reason
        assert result.psi_calibration.threshold is not None


def test_snapshot_loading_has_no_raw_input_model_read_fit_or_replay(
    loader, monkeypatch
):
    data = _snapshots()

    def forbidden(*args, **kwargs):
        raise AssertionError("snapshot loader attempted raw-data work")

    for module, name in (
        (secondary_development, "build_training_reference"),
        (secondary_development, "evaluate_validation"),
        (secondary_development, "_partition"),
        (secondary_development, "_standardized_features"),
        (secondary_drift, "fit_mmd_reference"),
        (secondary_drift, "fit_psi_reference"),
        (secondary_drift, "mmd_window_scores"),
        (secondary_drift, "psi_window_scores"),
        (gmm_monitor, "load_gmm_artifact_bytes"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    with monkeypatch.context() as no_files:
        no_files.setattr("builtins.open", forbidden)
        for name in ("open", "stat", "lstat", "access"):
            no_files.setattr(os, name, forbidden)
        result = loader.load_retained_drift_reference(**data)
    assert result.mmd_calibration.threshold == 0.725
    assert (
        "train_content"
        not in inspect.signature(loader.load_retained_drift_reference).parameters
    )


@pytest.mark.parametrize(
    "name", ["training_reference", "validation_audit", "preparation_summary"]
)
def test_wrong_byte_hash_is_rejected_before_semantic_verification(
    loader, name, monkeypatch
):
    data = _snapshots()
    data[name] += b" "

    def forbidden(*args, **kwargs):
        raise AssertionError("mismatched bytes reached semantic checking")

    monkeypatch.setattr(development_completion, "_drift", forbidden)
    with pytest.raises(loader.RetainedDriftError, match="hash_mismatch"):
        loader.load_retained_drift_reference(**data)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ref, audit, public: ref["mmd"]["stable_ids"].reverse(),
        lambda ref, audit, public: ref["mmd"].update(bandwidth_squared=2.0),
        lambda ref, audit, public: ref["mmd"]["values"][0].__setitem__(0, True),
        lambda ref, audit, public: ref["psi"]["features"][0][
            "training_counts"
        ].__setitem__(1, 1),
        lambda ref, audit, public: ref["psi"]["features"][0][
            "training_proportions"
        ].__setitem__(0, 0.0),
        lambda ref, audit, public: ref["scaler"]["scale"].__setitem__(0, 0.0),
        lambda ref, audit, public: ref.update(validation_declaration_sha256="f" * 64),
        lambda ref, audit, public: ref["input_hashes"].update(
            gmm_artifact_sha256="f" * 64
        ),
        lambda ref, audit, public: audit["input_hashes"].update(
            gmm_contract_sha256="f" * 64
        ),
        lambda ref, audit, public: audit["streams"]["audit"][
            "input_row_positions"
        ].reverse(),
        lambda ref, audit, public: audit["streams"]["audit"]["record_ids"].reverse(),
        lambda ref, audit, public: audit["streams"]["audit"][
            "window_end_positions"
        ].__setitem__(0, 255),
        lambda ref, audit, public: audit["results"]["mmd"]["calibration"].update(
            threshold=0.0
        ),
        lambda ref, audit, public: audit["results"]["psi"]["audit"].update(
            alert_count=0
        ),
        lambda ref, audit, public: public["methods"]["mmd"].update(threshold=0.0),
        lambda ref, audit, public: ref.update(mmd=None),
        lambda ref, audit, public: audit.update(streams=[]),
        lambda ref, audit, public: ref.update(scaler={}),
    ],
)
def test_semantic_mutations_fail_even_after_repairing_content_hashes(loader, mutation):
    data = _snapshots()
    reference, audit = (
        json.loads(data[key]) for key in ("training_reference", "validation_audit")
    )
    mutation(reference, audit, data["expected_drift_summary"])
    changed = _rebind(
        reference,
        audit,
        data["expected_drift_summary"],
        data["preparation_summary"],
        data["pins"],
    )
    with pytest.raises(loader.RetainedDriftError):
        loader.load_retained_drift_reference(**changed)


@pytest.mark.parametrize(
    "field", list(secondary_development.DevelopmentPins.__annotations__)
)
def test_every_supplied_pin_is_bound_to_retained_evidence(loader, field):
    data = _snapshots()
    data["pins"] = replace(data["pins"], **{field: "f" * 64})
    with pytest.raises(loader.RetainedDriftError):
        loader.load_retained_drift_reference(**data)


# SP-CORR-02: private canonicality and strict duplicate/nonfinite parsing remain closed.
@pytest.mark.parametrize("name", ["training_reference", "validation_audit"])
def test_private_json_rejects_noncanonical_bytes(loader, name):
    data = _snapshots()
    content = data[name] + b" "
    data[name] = content
    digest = sha256(content).hexdigest()
    if name == "training_reference":
        data["expected_reference_sha256"] = digest
    else:
        data["expected_audit_sha256"] = digest
    with pytest.raises(loader.RetainedDriftError, match="noncanonical_private_json"):
        loader.load_retained_drift_reference(**data)


@pytest.mark.parametrize(
    "name", ["training_reference", "validation_audit", "preparation_summary"]
)
@pytest.mark.parametrize("kind", ["duplicate", "nonfinite"])
def test_strict_json_rejects_duplicate_and_nonfinite_bytes(loader, name, kind):
    data = _snapshots()
    content = data[name]
    if kind == "duplicate":
        content = b'{"schema_version":1,' + content[1:]
    else:
        content = b'{"nonfinite":NaN,' + content[1:]
    data[name] = content
    digest = sha256(content).hexdigest()
    if name == "training_reference":
        data["expected_reference_sha256"] = digest
    elif name == "validation_audit":
        data["expected_audit_sha256"] = digest
    else:
        data["pins"] = replace(data["pins"], preparation_summary_sha256=digest)
    with pytest.raises(loader.RetainedDriftError):
        loader.load_retained_drift_reference(**data)


def test_audit_must_link_exact_reference_even_with_repaired_audit_digest(loader):
    data = _snapshots()
    audit = json.loads(data["validation_audit"])
    audit["training_reference_sha256"] = "f" * 64
    data["validation_audit"] = _bytes(audit)
    data["expected_audit_sha256"] = sha256(data["validation_audit"]).hexdigest()
    data["expected_drift_summary"]["private_sha256"]["validation-audit.json"] = data[
        "expected_audit_sha256"
    ]
    with pytest.raises(loader.RetainedDriftError, match="invalid_drift_audit_identity"):
        loader.load_retained_drift_reference(**data)


def test_unavailable_mmd_cannot_be_replaced_by_forged_zero_boundary(loader):
    data = _snapshots(training_count=128)
    reference = json.loads(data["training_reference"])
    audit = json.loads(data["validation_audit"])
    audit["results"]["mmd"]["calibration"].update(threshold=0.0, reason=None)
    changed = _rebind(
        reference,
        audit,
        data["expected_drift_summary"],
        data["preparation_summary"],
        data["pins"],
    )
    with pytest.raises(
        loader.RetainedDriftError, match="drift_boundary_or_alert_mismatch"
    ):
        loader.load_retained_drift_reference(**changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_reference_sha256", "A" * 64),
        ("expected_audit_sha256", "f" * 63),
        ("training_reference", bytearray(b"{}")),
        ("validation_audit", "{}"),
        ("preparation_summary", None),
        ("expected_drift_summary", []),
        ("pins", {}),
    ],
)
def test_invalid_argument_types_and_hashes_have_symbolic_errors(loader, field, value):
    data = _snapshots()
    data[field] = value
    with pytest.raises(loader.RetainedDriftError) as rejected:
        loader.load_retained_drift_reference(**data)
    assert all(
        character.islower() or character == "_" for character in str(rejected.value)
    )


@pytest.mark.parametrize("name", ["source", "suffix", "train", "validation"])
def test_preparation_metadata_must_match_pins_after_repairing_its_digest(loader, name):
    data = _snapshots()
    preparation = json.loads(data["preparation_summary"])
    if name == "source":
        preparation["declared_sources"]["phiusiil"]["csv_sha256"] = "f" * 64
    elif name == "suffix":
        preparation["declared_sources"]["public_suffix_list"]["sha256"] = "f" * 64
    else:
        preparation["output_hashes"][f"{name}.jsonl"] = "f" * 64
    data["preparation_summary"] = _bytes(preparation)
    data["pins"] = replace(
        data["pins"],
        preparation_summary_sha256=sha256(data["preparation_summary"]).hexdigest(),
    )
    with pytest.raises(loader.RetainedDriftError, match="preparation_.*_mismatch"):
        loader.load_retained_drift_reference(**data)
