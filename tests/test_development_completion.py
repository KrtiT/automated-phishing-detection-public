"""Retained synthetic evidence, without source reads or model fitting."""

import json
import os
import warnings
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest
import threadpoolctl
from test_secondary_development import _fixture

from automated_phishing_detection import (
    baselines,
    execution_receipt,
    secondary_development,
    secondary_tabular,
    source_runner,
)

STEPS = (
    "drift",
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)


def _bytes(value):
    return execution_receipt._json_bytes(value, "fixture")


def _load(path):
    return json.loads(path.read_bytes())


def _write(path, value):
    path.write_bytes(_bytes(value))


def _model(member, count):
    kind = "permutation" if member.startswith("permutation_") else member
    seed = int(member[-2:]) if kind == "permutation" else 42
    width = len(secondary_tabular._feature_names(kind))
    state = {
        "scaler": {
            "mean": [0.0] * width,
            "scale": [1.0] * width,
            "variance": [1.0] * width,
            "n_samples_seen": count,
        },
        "coefficients": [[0.0] * width],
        "intercept": [0.0],
        "n_iter": [1],
    }
    if kind == "random_forest":
        state = {
            "trees": [
                {
                    "children_left": [-1],
                    "children_right": [-1],
                    "feature": [-2],
                    "threshold": [-2.0],
                    "value": [[0.5, 0.5]],
                    "random_state": i,
                }
                for i in range(100)
            ]
        }
    return secondary_tabular._json_bytes(
        {
            "schema_version": 1,
            "contract_id": "secondary-development-v1",
            "artifact_type": "secondary-tabular-model",
            "method_version": "secondary-tabular-v1",
            "analysis_stage": "development_validation_only",
            "protected_evaluation_authorized": False,
            "model_kind": kind,
            "features": secondary_tabular._feature_names(kind),
            "classes": [0, 1],
            "parameters": secondary_tabular._parameters(kind),
            "state": state,
            "scoring": secondary_tabular._RF_SCORING
            if kind == "random_forest"
            else secondary_tabular._LOGISTIC_SCORING,
            "permutation": {"bit_generator": "PCG64", "seed": seed}
            if kind == "permutation"
            else None,
            "training_row_count": count,
            "software_versions": secondary_tabular._VERSIONS,
        }
    )


@pytest.fixture(scope="module")
def material():
    data = _fixture(secondary_development)
    reference = secondary_development.build_training_reference(**data["arguments"])
    drift = secondary_development.evaluate_validation(
        reference, data["validation_content"]
    )
    rows = [
        {"record_id": row["record_id"], "label": row["is_phishing"], "probability": 0.5}
        for row in data["validation"]
    ]
    threshold = baselines.select_validation_threshold(
        [row["probability"] for row in rows], [row["label"] for row in rows]
    )
    return data, drift, rows, threshold


@pytest.fixture
def verifier():
    from automated_phishing_detection import development_completion

    return development_completion


@pytest.fixture
def published(verifier, material, tmp_path, monkeypatch):
    from automated_phishing_detection import development_execution, development_runner

    data, drift, rows, threshold = material
    root = tmp_path / "source"
    root.mkdir()
    base = SimpleNamespace(
        root=root,
        revision="a" * 40,
        contract_sha256="b" * 64,
        runtime_json='{"fixture":true}',
    )
    binding = development_execution.DevelopmentExecutionBinding(
        base,
        "c" * 64,
        "d" * 64,
        data["arguments"]["pins"],
        data["arguments"]["preparation_summary"],
    )
    paths = development_runner.DevelopmentRunPaths(
        *(
            tmp_path / name
            for name in (
                "train",
                "validation",
                "suffix",
                "logistic",
                "gmm",
                "attempt",
                "summary.json",
            )
        )
    )
    rechecks = []
    monkeypatch.setattr(
        development_execution,
        "recheck_development_binding",
        lambda value: rechecks.append(value),
    )
    identity = {
        "kind": "secondary_development",
        "revision": base.revision,
        "execution_contract_sha256": base.contract_sha256,
        "development_profile_sha256": binding.profile_sha256,
        "methods_contract_sha256": binding.methods_sha256,
        "runtime_sha256": sha256(base.runtime_json.encode()).hexdigest(),
        "pins": asdict(binding.pins),
        "ordered_steps": list(STEPS),
    }
    root_attempt = execution_receipt.reserve_attempt(paths.attempt, identity=identity)
    members = []
    for member in STEPS:
        attempt = execution_receipt.reserve_attempt(
            paths.attempt / member,
            identity={
                "root_reservation_sha256": root_attempt.reservation_sha256,
                "member": member,
            },
        )
        if member == "drift":
            outputs, result = drift.private_outputs, drift.public_summary
        else:
            kind = "permutation" if member.startswith("permutation_") else member
            audit = {
                "batch_size": 1,
                "warning_records": [],
                "portable_exact_parity": True,
                "threshold_role": "secondary_descriptive_operating_point",
            }
            if kind != "random_forest":
                audit.update(
                    platform_identity=baselines._platform_identity(),
                    max_absolute_decision_difference=0.0,
                    max_absolute_probability_difference=0.0,
                )
            outputs = {
                "model.json": _model(member, len(data["train"])),
                "validation-predictions.jsonl": b"".join(
                    secondary_tabular._json_bytes(row) for row in rows
                ),
                "threshold.json": secondary_tabular._json_bytes(threshold),
                "scoring-audit.json": secondary_tabular._json_bytes(audit),
            }
            result = {
                "model_kind": kind,
                "seed": int(member[-2:]) if kind == "permutation" else 42,
                "row_count": len(rows),
                "class_counts": {"0": 320, "1": 320},
                "validation_threshold": threshold,
                "scoring_audit": audit,
                "analysis_role": "descriptive_secondary_not_primary",
                "score_metrics": {"average_precision": 0.5, "roc_auc": 0.5},
            }
        summary = {
            "schema_version": 1,
            "status": "development_member_completed",
            "member": member,
            "root_reservation_sha256": root_attempt.reservation_sha256,
            "reservation_sha256": attempt.reservation_sha256,
            "private_sha256": {
                name: sha256(content).hexdigest() for name, content in outputs.items()
            },
            "result": result,
        }
        marker = paths.attempt / f"{member}.json"
        execution_receipt.publish_completion(
            attempt, private_outputs=outputs, public_summary=summary, public_path=marker
        )
        members.append(
            {
                "member": member,
                "public_summary_sha256": sha256(marker.read_bytes()).hexdigest(),
                "summary": summary,
            }
        )
    execution = {**identity, "reservation_sha256": root_attempt.reservation_sha256}
    outputs = {
        "bindings.json": _bytes(execution),
        "members.json": _bytes({"schema_version": 1, "members": members}),
    }
    summary = {
        "schema_version": 1,
        "status": "development_evidence_published",
        "protected_evaluation_authorized": False,
        "execution": execution,
        "members": members,
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in outputs.items()
        },
    }
    execution_receipt.publish_completion(
        root_attempt,
        private_outputs=outputs,
        public_summary=summary,
        public_path=paths.public_summary,
    )
    return binding, paths, rechecks


def _relink(paths):
    members = []
    for member in STEPS:
        marker = paths.attempt / f"{member}.json"
        summary = _load(marker)
        evidence = paths.attempt / member / "evidence"
        summary["private_sha256"] = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in evidence.iterdir()
        }
        if member == "drift":
            summary["result"]["private_sha256"] = summary["private_sha256"]
        _write(marker, summary)
        outcome = _load(paths.attempt / member / "outcome.json")
        outcome.update(
            private_sha256=summary["private_sha256"],
            public_summary_sha256=sha256(marker.read_bytes()).hexdigest(),
        )
        _write(paths.attempt / member / "outcome.json", outcome)
        members.append(
            {
                "member": member,
                "public_summary_sha256": sha256(marker.read_bytes()).hexdigest(),
                "summary": summary,
            }
        )
    _write(
        paths.attempt / "evidence" / "members.json",
        {"schema_version": 1, "members": members},
    )
    summary = _load(paths.public_summary)
    summary["members"] = members
    summary["private_sha256"] = {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in (paths.attempt / "evidence").iterdir()
    }
    _write(paths.public_summary, summary)
    outcome = _load(paths.attempt / "outcome.json")
    outcome.update(
        private_sha256=summary["private_sha256"],
        public_summary_sha256=sha256(paths.public_summary.read_bytes()).hexdigest(),
    )
    _write(paths.attempt / "outcome.json", outcome)


def test_accepts_all_eight_members_without_reading_original_inputs(
    verifier, published, monkeypatch
):
    binding, paths, rechecks = published
    original = source_runner._read_file_once
    reads = Counter()

    def read(path, **kwargs):
        assert path == paths.public_summary or path.is_relative_to(paths.attempt)
        reads[path] += 1
        return original(path, **kwargs)

    monkeypatch.setattr(source_runner, "_read_file_once", read)
    for name in ("fit_formatting", "fit_label_permutation", "fit_random_forest"):
        monkeypatch.setattr(
            secondary_tabular, name, lambda *a, **k: pytest.fail("fit invoked")
        )
    monkeypatch.setattr(
        secondary_tabular.SecondaryModel,
        "score_urls",
        lambda *a: pytest.fail("URL scoring invoked"),
    )
    assert verifier.verify_development_completion(
        binding, paths, producer_exit_code=0
    ) == _load(paths.public_summary)
    assert len(reads) == 68
    assert set(reads.values()) == {1}
    assert len(rechecks) == 2


@pytest.mark.parametrize("code", [True, False, None, "0", 0.0, 1, -9])
def test_requires_observed_exact_zero_before_access(verifier, monkeypatch, code):
    monkeypatch.setattr(
        source_runner, "_read_file_once", lambda *a, **k: pytest.fail("read")
    )
    with pytest.raises(verifier.DevelopmentCompletionError, match="producer_exit"):
        verifier.verify_development_completion(None, None, producer_exit_code=code)


@pytest.mark.parametrize(
    "record,field,value",
    [
        ("reservation.json", "schema_version", True),
        ("finalize.claim", "operation", "failure"),
        ("outcome.json", "status", "failed"),
        ("formatting/reservation.json", "schema_version", True),
        ("formatting/finalize.claim", "operation", "failure"),
        ("formatting/outcome.json", "status", "failed"),
    ],
)
def test_exact_receipts(verifier, published, record, field, value):
    binding, paths, _ = published
    path = paths.attempt / record
    changed = _load(path)
    changed[field] = value
    _write(path, changed)
    _relink(paths)
    with pytest.raises(verifier.DevelopmentCompletionError):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "member,field,value",
    [
        ("formatting", "seed", 43),
        ("permutation_42", "seed", 43),
        ("random_forest", "model_kind", "formatting"),
        ("formatting", "row_count", True),
        ("formatting", "extra", "https://private.invalid"),
    ],
)
def test_member_semantics_survive_repaired_hashes(
    verifier, published, member, field, value
):
    binding, paths, _ = published
    marker = paths.attempt / f"{member}.json"
    changed = _load(marker)
    changed["result"][field] = value
    _write(marker, changed)
    _relink(paths)
    with pytest.raises(verifier.DevelopmentCompletionError):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "change",
    [
        "bool_label",
        "probability",
        "extra",
        "order",
        "dropped",
        "foreign_id",
        "label_disagreement",
    ],
)
def test_prediction_semantics_survive_repaired_hashes(verifier, published, change):
    binding, paths, _ = published
    path = paths.attempt / "formatting/evidence/validation-predictions.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if change == "bool_label":
        rows[0]["label"] = False
    elif change == "probability":
        rows[0]["probability"] = 1.1
    elif change == "extra":
        rows[0]["raw_url"] = "https://private.invalid"
    elif change == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif change == "dropped":
        rows.pop()
    elif change == "foreign_id":
        rows[0]["record_id"] = "unbound"
    else:
        rows[0]["label"], rows[-1]["label"] = rows[-1]["label"], rows[0]["label"]
    path.write_bytes(b"".join(secondary_tabular._json_bytes(row) for row in rows))
    _relink(paths)
    with pytest.raises(verifier.DevelopmentCompletionError):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


def test_recomputes_threshold_without_refitting(verifier, published):
    binding, paths, _ = published
    marker = paths.attempt / "formatting.json"
    summary = _load(marker)
    summary["result"]["validation_threshold"]["threshold"] = 0.5
    _write(marker, summary)
    (paths.attempt / "formatting/evidence/threshold.json").write_bytes(
        secondary_tabular._json_bytes(summary["result"]["validation_threshold"])
    )
    _relink(paths)
    with pytest.raises(verifier.DevelopmentCompletionError, match="threshold"):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("field", ["average_precision", "roc_auc"])
def test_recomputes_rank_metrics_from_retained_predictions(verifier, published, field):
    binding, paths, _ = published
    marker = paths.attempt / "permutation_42.json"
    summary = _load(marker)
    summary["result"]["score_metrics"][field] = 0.99
    _write(marker, summary)
    _relink(paths)
    with pytest.raises(verifier.DevelopmentCompletionError, match="score_metrics"):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("change", ["seed", "kind", "training_count", "noncanonical"])
def test_saved_model_identity_and_state_checked(verifier, published, change):
    binding, paths, _ = published
    path = paths.attempt / "permutation_42/evidence/model.json"
    model = _load(path)
    if change == "seed":
        model["permutation"]["seed"] = 43
    elif change == "kind":
        path.write_bytes(_model("random_forest", 256))
    elif change == "training_count":
        model["training_row_count"] = 255
        model["state"]["scaler"]["n_samples_seen"] = 255
    if change != "kind":
        path.write_bytes(
            json.dumps(model).encode()
            if change == "noncanonical"
            else secondary_tabular._json_bytes(model)
        )
    _relink(paths)
    with pytest.raises(verifier.DevelopmentCompletionError):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize(
    "change", ["allocation", "window", "alert", "feature_score", "public_extra"]
)
def test_drift_trace_semantics_survive_repaired_hashes(verifier, published, change):
    binding, paths, _ = published
    path = paths.attempt / "drift/evidence/validation-audit.json"
    audit = _load(path)
    if change == "allocation":
        audit["streams"]["audit"]["input_row_positions"][0] = 0
    elif change == "window":
        audit["streams"]["audit"]["window_end_positions"][0] = 255
    elif change == "alert":
        audit["results"]["mmd"]["audit"]["alert_count"] = 999
    elif change == "feature_score":
        audit["streams"]["audit"]["psi"]["feature_scores"][0][0] = 999.0
    else:
        marker = paths.attempt / "drift.json"
        summary = _load(marker)
        summary["result"]["raw_url"] = "https://private.invalid"
        _write(marker, summary)
    path.write_bytes(secondary_tabular._json_bytes(audit))
    _relink(paths)
    with pytest.raises(verifier.DevelopmentCompletionError):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


def test_negative_nonmax_psi_feature_rejected_with_repaired_hashes(verifier, published):
    binding, paths, _ = published
    path = paths.attempt / "drift/evidence/validation-audit.json"
    audit = _load(path)
    trace = audit["streams"]["audit"]["psi"]
    features = trace["feature_scores"][0]
    maximum = trace["scores"][0]
    position = next(index for index, value in enumerate(features) if value < maximum)
    features[position] = -1.0
    assert max(features) == maximum
    path.write_bytes(secondary_tabular._json_bytes(audit))
    _relink(paths)
    with pytest.raises(
        verifier.DevelopmentCompletionError, match="invalid_psi_feature_scores"
    ):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


@pytest.mark.parametrize("method,score", [("psi", -0.0), ("mmd", -1e-16)])
def test_trace_preserves_psi_signed_zero_and_mmd_roundoff(verifier, method, score):
    trace = {
        "window_end_positions": [256],
        "scores": [score],
        "feature_scores": [[score] * 26] if method == "psi" else [],
        "reason": None,
    }
    assert verifier._trace(trace, [256], method, None).scores == (score,)


@pytest.mark.parametrize("change", ["missing", "extra", "symlink", "hardlink"])
def test_exact_nonaliased_outputs(verifier, published, change, tmp_path):
    binding, paths, _ = published
    target = paths.attempt / "formatting/evidence/threshold.json"
    if change == "missing":
        target.unlink()
    elif change == "extra":
        (paths.attempt / "unrequested").write_bytes(b"unexpected")
    elif change == "hardlink":
        os.link(target, tmp_path / "alias")
    else:
        saved = tmp_path / "saved"
        target.rename(saved)
        target.symlink_to(saved)
    with pytest.raises(verifier.DevelopmentCompletionError):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


def test_directory_swap_during_read_rejected(verifier, published, monkeypatch):
    binding, paths, _ = published
    original = source_runner._read_file_once
    target = paths.attempt / "formatting/evidence"
    swapped = False

    def read(path, **kwargs):
        nonlocal swapped
        content = original(path, **kwargs)
        if path.parent == target and not swapped:
            swapped = True
            target.rename(target.with_name("old-evidence"))
            target.mkdir()
        return content

    monkeypatch.setattr(source_runner, "_read_file_once", read)
    with pytest.raises(verifier.DevelopmentCompletionError):
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)


def test_final_binding_failure_rejects_completion_without_private_details(
    verifier, published, monkeypatch
):
    from automated_phishing_detection import development_execution

    binding, paths, _ = published
    calls = []

    def recheck(value):
        calls.append(value)
        if len(calls) == 2:
            raise ValueError("private path must not leak")

    monkeypatch.setattr(development_execution, "recheck_development_binding", recheck)
    with pytest.raises(verifier.DevelopmentCompletionError) as caught:
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)
    assert "private path" not in str(caught.value)


def test_rederived_identity_rejects_profile_change(verifier, published):
    binding, paths, _ = published
    with pytest.raises(verifier.DevelopmentCompletionError):
        verifier.verify_development_completion(
            replace(binding, profile_sha256="e" * 64), paths, producer_exit_code=0
        )


def test_recomputation_restores_strict_runtime_guard(verifier, published, monkeypatch):
    binding, paths, _ = published
    original = baselines.select_validation_threshold
    observed = []

    def threshold(scores, labels):
        assert set(np.geterr().values()) == {"raise"}
        assert all(pool["num_threads"] == 1 for pool in threadpoolctl.threadpool_info())
        assert warnings.filters[0][0] == "error"
        observed.append(True)
        return original(scores, labels)

    monkeypatch.setattr(baselines, "select_validation_threshold", threshold)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filters = list(warnings.filters)
        pools = threadpoolctl.threadpool_info()
        verifier.verify_development_completion(binding, paths, producer_exit_code=0)
        assert set(np.geterr().values()) == {"ignore"}
        assert warnings.filters == filters
        assert threadpoolctl.threadpool_info() == pools
    assert len(observed) == 7


def test_numerical_warning_rejects_and_restores_caller_state(
    verifier, published, monkeypatch
):
    from automated_phishing_detection import secondary_metrics

    binding, paths, _ = published

    def warned(*args):
        warnings.warn("private values must not leak", RuntimeWarning)
        return 0.5

    monkeypatch.setattr(secondary_metrics, "average_precision_score", warned)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filters = list(warnings.filters)
        with pytest.raises(verifier.DevelopmentCompletionError) as caught:
            verifier.verify_development_completion(binding, paths, producer_exit_code=0)
        assert set(np.geterr().values()) == {"ignore"}
        assert warnings.filters == filters
        assert "private values" not in str(caught.value)
