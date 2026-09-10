import json
from hashlib import sha256

import numpy as np
import pytest
from scipy.special import expit

from automated_phishing_detection import baselines, fixed_cascade
from automated_phishing_detection.url_features import (
    FEATURE_NAMES,
    extract_url_features,
)


def _threshold_record(status="selected", threshold=0.5, *, positive=1, negative=400):
    if status == "target_not_met":
        return {
            "status": status,
            "threshold": None,
            "candidate_count": 3,
            "counts": None,
            "recall": None,
            "observed_fpr": None,
            "fpr_upper_95": None,
        }
    return {
        "status": status,
        "threshold": threshold,
        "candidate_count": 3,
        "counts": {
            "true_positive": positive,
            "false_positive": 0,
            "true_negative": negative,
            "false_negative": 0,
            "positive": positive,
            "negative": negative,
        },
        "recall": 1.0,
        "observed_fpr": 0.0,
        "fpr_upper_95": baselines.clopper_pearson_upper(0, negative),
    }


def _artifact(contract_sha="0" * 64):
    width = len(FEATURE_NAMES)
    coefficients = [0.0] * width
    coefficients[0] = 0.1
    return {
        "schema_version": 2,
        "artifact_type": "rq1-baseline-model",
        "analysis_stage": "development_validation_only",
        "contract_id": "rq1-baselines-v2",
        "contract_sha256": contract_sha,
        "model_name": "Logistic-L1",
        "features": list(FEATURE_NAMES),
        "classes": [0, 1],
        "scaler": {
            "config": {
                "class": "StandardScaler",
                "fit_partition": "train",
                "with_mean": True,
                "with_std": True,
            },
            "mean": [0.0] * width,
            "scale": [1.0] * width,
            "variance": [1.0] * width,
            "n_samples_seen": 20,
        },
        "classifier": {
            "config": {
                "class": "LogisticRegression",
                "penalty": "l1",
                "solver": "saga",
                "C": 1.0,
                "class_weight": "balanced",
                "fit_intercept": True,
                "max_iter": 5000,
                "tol": 0.0001,
                "random_state": 42,
            },
            "coefficients": [coefficients],
            "intercept": [0.0],
            "n_iter": [12],
        },
        "validation_scoring_audit": {
            "platform_identity": {
                "sys_platform": "darwin",
                "platform_machine": "arm64",
                "numpy_blas_name": "accelerate",
            },
            "warning_records": [],
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
        },
        "validation_threshold": _threshold_record(),
        "input_hashes": {
            "train": "1" * 64,
            "validation": "2" * 64,
            "preparation_summary": "3" * 64,
            "contract": contract_sha,
        },
        "software_versions": {
            "numpy": "2.2.6",
            "scikit-learn": "1.7.2",
            "scipy": "1.15.3",
        },
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
    }


def _write_artifact(tmp_path, artifact=None):
    payload = _artifact() if artifact is None else artifact
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii")
    path = tmp_path / "logistic-l1.json"
    path.write_bytes(content)
    return path, sha256(content).hexdigest()


def _selected_threshold(scores, labels):
    selected = baselines.select_validation_threshold(scores, labels)
    assert selected["status"] == "selected"
    return selected


def _loaded_stage1(tmp_path, threshold_record, *, probability_difference=0.0):
    artifact = _artifact()
    artifact["validation_threshold"] = threshold_record
    artifact["validation_scoring_audit"]["max_absolute_probability_difference"] = (
        probability_difference
    )
    path, digest = _write_artifact(tmp_path, artifact)
    return fixed_cascade.load_logistic_l1_artifact(
        path, expected_sha256=digest, expected_contract_sha256="0" * 64
    )


def test_loader_binds_exact_bytes_contract_and_schema(tmp_path):
    path, digest = _write_artifact(tmp_path)

    loaded = fixed_cascade.load_logistic_l1_artifact(
        path, expected_sha256=digest, expected_contract_sha256="0" * 64
    )

    assert loaded.feature_names == FEATURE_NAMES
    assert loaded.validation_threshold_record == _threshold_record()
    with pytest.raises(fixed_cascade.FixedCascadeError, match="SHA-256 mismatch"):
        fixed_cascade.load_logistic_l1_artifact(
            path, expected_sha256="f" * 64, expected_contract_sha256="0" * 64
        )


def test_strict_content_loader_matches_the_public_path_loader(tmp_path):
    path, digest = _write_artifact(tmp_path)
    content = path.read_bytes()

    from_path = fixed_cascade.load_logistic_l1_artifact(
        path, expected_sha256=digest, expected_contract_sha256="0" * 64
    )
    from_content = fixed_cascade._load_logistic_l1_artifact_bytes(
        content,
        expected_sha256=digest,
        expected_contract_sha256="0" * 64,
    )

    assert from_content == from_path
    assert from_content.score_urls(["https://example.test/path"]) == (
        from_path.score_urls(["https://example.test/path"])
    )


def test_strict_content_loader_enforces_the_same_hash_and_schema_rules(tmp_path):
    path, digest = _write_artifact(tmp_path)
    content = path.read_bytes()

    with pytest.raises(fixed_cascade.FixedCascadeError, match="SHA-256 mismatch"):
        fixed_cascade._load_logistic_l1_artifact_bytes(
            content,
            expected_sha256="f" * 64,
            expected_contract_sha256="0" * 64,
        )
    malformed = content.replace(
        b'"model_name": "Logistic-L1"', b'"model_name": "length-only"'
    )
    with pytest.raises(fixed_cascade.FixedCascadeError, match="model_name"):
        fixed_cascade._load_logistic_l1_artifact_bytes(
            malformed,
            expected_sha256=sha256(malformed).hexdigest(),
            expected_contract_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(model_name="length-only"),
        lambda value: value["features"].reverse(),
        lambda value: value["classes"].reverse(),
        lambda value: value["scaler"].update(extra=True),
        lambda value: value["classifier"].update(intercept=[float("inf")]),
        lambda value: value["classifier"].update(intercept=[10**1000]),
        lambda value: value["validation_scoring_audit"]["warning_records"].append(
            {
                "stage": [],
                "category": "RuntimeWarning",
                "message": "overflow encountered in matmul",
            }
        ),
    ),
)
def test_loader_rejects_schema_or_nonfinite_state(tmp_path, mutation):
    artifact = _artifact()
    mutation(artifact)
    path, digest = _write_artifact(tmp_path, artifact)

    with pytest.raises(fixed_cascade.FixedCascadeError):
        fixed_cascade.load_logistic_l1_artifact(
            path, expected_sha256=digest, expected_contract_sha256="0" * 64
        )


def test_portable_model_scores_existing_raw_url_features_without_fit(tmp_path):
    path, digest = _write_artifact(tmp_path)
    loaded = fixed_cascade.load_logistic_l1_artifact(
        path, expected_sha256=digest, expected_contract_sha256="0" * 64
    )
    urls = ("https://example.test/a", "http://safe.test/path?q=1")

    scores = loaded.score_urls(urls)

    expected = tuple(expit(0.1 * extract_url_features(url)[0]) for url in urls)
    assert scores == pytest.approx(expected)


@pytest.mark.parametrize("urls", ([], "https://example.test", [None]))
def test_portable_model_rejects_invalid_url_inputs(tmp_path, urls):
    path, digest = _write_artifact(tmp_path)
    loaded = fixed_cascade.load_logistic_l1_artifact(
        path, expected_sha256=digest, expected_contract_sha256="0" * 64
    )

    with pytest.raises(fixed_cascade.FixedCascadeError):
        loaded.score_urls(urls)


def test_scoring_uses_inclusive_band_and_selected_path_score_and_decision():
    scored = fixed_cascade.score_fixed_cascade(
        np.array([0.25, 0.50, 0.75, 0.90]),
        np.array([0.80, 0.20, 0.40, 0.10]),
        stage1_threshold=0.50,
        transformer_threshold=0.60,
        half_width=0.25,
    )

    assert scored.transformer_invoked == (True, True, True, False)
    assert scored.probabilities == pytest.approx((0.80, 0.20, 0.40, 0.90))
    assert scored.decisions == (1, 0, 0, 1)


def test_candidates_are_exact_sorted_unique_validation_distances():
    assert fixed_cascade.cascade_candidate_half_widths(
        np.array([0.9, 0.4, 0.6, 0.4]), 0.5
    ) == pytest.approx((0.1, 0.4))


@pytest.mark.parametrize(
    "operation",
    (
        lambda values: fixed_cascade.score_fixed_cascade(
            values,
            np.array([0.1, 0.9]),
            stage1_threshold=0.5,
            transformer_threshold=0.5,
            half_width=0.1,
        ),
        lambda values: fixed_cascade.cascade_candidate_half_widths(values, 0.5),
    ),
)
def test_public_probability_paths_reject_complex_arrays(operation):
    with pytest.raises(fixed_cascade.FixedCascadeError, match="numeric vector"):
        operation(np.array([0.1 + 0.0j, 0.9 + 0.0j]))


def test_calibration_rejects_complex_arrays(tmp_path):
    model = _loaded_stage1(tmp_path, _threshold_record("target_not_met"))
    with pytest.raises(fixed_cascade.FixedCascadeError, match="numeric vector"):
        fixed_cascade.calibrate_fixed_cascade(
            model,
            np.array([0.1 + 0.0j, 0.9 + 0.0j]),
            np.array([0.1, 0.9]),
            np.array([0, 1], dtype=np.int8),
            transformer_threshold_record=_threshold_record("target_not_met"),
        )


def test_calibration_selects_fewest_invocations_that_meet_both_gates(tmp_path):
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(10, dtype=np.int8)))
    stage1 = np.concatenate(
        (
            np.array([0.49]),
            np.full(399, 0.01),
            np.full(5, 0.9),
            np.full(3, 0.45),
            np.full(2, 0.3),
        )
    )
    transformer = np.concatenate((np.full(400, 0.01), np.full(10, 0.9)))
    model = _loaded_stage1(tmp_path, _selected_threshold(stage1, labels))

    result = fixed_cascade.calibrate_fixed_cascade(
        model,
        stage1,
        transformer,
        labels,
        transformer_threshold_record=_selected_threshold(transformer, labels),
    )

    assert result["status"] == "selected"
    assert result["accepted_cascade"] is True
    assert result["half_width"] == pytest.approx(0.6)
    assert result["transformer_invocations"] == 11
    assert result["recall"] == 1.0
    assert result["fpr_upper_95"] == baselines.clopper_pearson_upper(0, 400)


def test_candidate_tie_is_resolved_by_smaller_half_width():
    chosen = fixed_cascade._choose_candidate(
        [
            {"half_width": 0.2, "transformer_invocations": 4},
            {"half_width": 0.1, "transformer_invocations": 4},
        ]
    )
    assert chosen["half_width"] == 0.1


def test_exact_two_percent_recall_loss_is_accepted_without_float_drift(tmp_path):
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(50, dtype=np.int8)))
    stage1 = np.concatenate((np.full(400, 0.1), np.full(6, 0.9), np.full(44, 0.1)))
    transformer = np.concatenate((np.full(400, 0.1), np.full(7, 0.9), np.full(43, 0.1)))
    model = _loaded_stage1(tmp_path, _selected_threshold(stage1, labels))

    result = fixed_cascade.calibrate_fixed_cascade(
        model,
        stage1,
        transformer,
        labels,
        transformer_threshold_record=_selected_threshold(transformer, labels),
    )

    assert result["status"] == "selected"
    assert result["half_width"] == 0.0
    assert result["counts"]["true_positive"] == 6
    assert result["minimum_recall"] == pytest.approx(0.12)


def test_target_not_met_threshold_is_propagated_without_calibration(tmp_path):
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold_record = baselines.select_validation_threshold(scores, labels)
    assert threshold_record["status"] == "target_not_met"
    model = _loaded_stage1(tmp_path, threshold_record)
    result = fixed_cascade.calibrate_fixed_cascade(
        model,
        scores,
        scores,
        labels,
        transformer_threshold_record=threshold_record,
    )

    assert result == {
        "schema_version": 1,
        "status": "target_not_met",
        "accepted_cascade": False,
        "reason": "threshold_target_not_met",
        "threshold_statuses": {
            "stage1": "target_not_met",
            "transformer": "target_not_met",
        },
        "candidate_count": 0,
        "half_width": None,
        "transformer_invocations": None,
        "transformer_invocation_rate": None,
        "counts": None,
        "recall": None,
        "observed_fpr": None,
        "fpr_upper_95": None,
        "minimum_recall": None,
        "maximum_fpr_upper_95": 0.01,
    }


def test_selected_threshold_must_pass_the_frozen_fpr_gate(tmp_path):
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(1, dtype=np.int8)))
    scores = np.concatenate((np.full(400, 0.1), np.array([0.9])))
    model = _loaded_stage1(tmp_path, _selected_threshold(scores, labels))
    record = _threshold_record()
    record["counts"] = {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 399,
        "false_negative": 0,
        "positive": 1,
        "negative": 400,
    }
    record["observed_fpr"] = 1 / 400
    record["fpr_upper_95"] = baselines.clopper_pearson_upper(1, 400)

    with pytest.raises(fixed_cascade.FixedCascadeError, match="frozen FPR gate"):
        fixed_cascade.calibrate_fixed_cascade(
            model,
            scores,
            scores,
            labels,
            transformer_threshold_record=record,
        )


def test_selected_threshold_population_must_match_calibration_labels(tmp_path):
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(2, dtype=np.int8)))
    scores = np.concatenate((np.full(400, 0.1), np.full(2, 0.9)))
    model = _loaded_stage1(tmp_path, _threshold_record())

    with pytest.raises(fixed_cascade.FixedCascadeError, match="scores"):
        fixed_cascade.calibrate_fixed_cascade(
            model,
            scores,
            scores,
            labels,
            transformer_threshold_record=_selected_threshold(scores, labels),
        )


def test_threshold_record_must_match_scores_not_only_class_totals(tmp_path):
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(10, dtype=np.int8)))
    stage1 = np.concatenate((np.full(400, 0.1), np.full(5, 0.9), np.full(5, 0.1)))
    transformer_reference = np.concatenate((np.full(400, 0.1), np.full(10, 0.9)))
    transformer_wrong = np.full(410, 0.1)
    model = _loaded_stage1(tmp_path, _selected_threshold(stage1, labels))

    with pytest.raises(fixed_cascade.FixedCascadeError, match="scores"):
        fixed_cascade.calibrate_fixed_cascade(
            model,
            stage1,
            transformer_wrong,
            labels,
            transformer_threshold_record=_selected_threshold(
                transformer_reference, labels
            ),
        )


def test_stage1_threshold_drift_is_limited_by_artifact_audit(tmp_path):
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(10, dtype=np.int8)))
    reference = np.concatenate((np.full(400, 0.1), np.full(10, 0.9)))
    portable = np.concatenate((np.full(400, 0.1), np.full(10, 0.9000005)))
    transformer_record = _selected_threshold(reference, labels)
    artifact_record = _selected_threshold(reference, labels)

    accepted_model = _loaded_stage1(
        tmp_path, artifact_record, probability_difference=0.000001
    )
    result = fixed_cascade.calibrate_fixed_cascade(
        accepted_model,
        portable,
        reference,
        labels,
        transformer_threshold_record=transformer_record,
    )
    assert result["status"] == "selected"
    assert accepted_model.validation_threshold_record["threshold"] == 0.9

    rejected_model = _loaded_stage1(
        tmp_path, artifact_record, probability_difference=0.0000001
    )
    with pytest.raises(fixed_cascade.FixedCascadeError, match="scoring audit"):
        fixed_cascade.calibrate_fixed_cascade(
            rejected_model,
            portable,
            reference,
            labels,
            transformer_threshold_record=transformer_record,
        )


def test_transformer_threshold_must_exactly_match_recomputed_record(tmp_path):
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(10, dtype=np.int8)))
    stage1 = np.concatenate((np.full(400, 0.1), np.full(5, 0.9), np.full(5, 0.1)))
    transformer = np.concatenate((np.full(400, 0.1), np.full(10, 0.9)))
    model = _loaded_stage1(tmp_path, _selected_threshold(stage1, labels))
    transformer_record = _selected_threshold(transformer, labels)
    transformer_record["threshold"] = float(
        np.nextafter(transformer_record["threshold"], 0.0)
    )

    with pytest.raises(fixed_cascade.FixedCascadeError, match="exactly match"):
        fixed_cascade.calibrate_fixed_cascade(
            model,
            stage1,
            transformer,
            labels,
            transformer_threshold_record=transformer_record,
        )


@pytest.mark.parametrize(
    ("stage1", "transformer", "labels"),
    (
        ([0.1, np.nan], [0.1, 0.9], [0, 1]),
        ([0.1, 0.9], [0.1], [0, 1]),
        ([0.1, 0.9], [0.1, 0.9], [0, 2]),
        ([[0.1], [0.9]], [0.1, 0.9], [0, 1]),
        ([[0.1], [0.9, 0.8]], [0.1, 0.9], [0, 1]),
        ([0.1, 0.9], [0.1, 0.9], [[0], [1, 0]]),
    ),
)
def test_calibration_rejects_nonfinite_nonbinary_or_misshaped_inputs(
    tmp_path, stage1, transformer, labels
):
    model = _loaded_stage1(tmp_path, _threshold_record())
    with pytest.raises(fixed_cascade.FixedCascadeError):
        fixed_cascade.calibrate_fixed_cascade(
            model,
            stage1,
            transformer,
            labels,
            transformer_threshold_record=_threshold_record(),
        )
