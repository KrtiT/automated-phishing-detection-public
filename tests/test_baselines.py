import json
import math
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from automated_phishing_detection import baselines

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data" / "rq1-baseline-contract-v2.json"
ACCELERATE_ENVIRONMENT = {
    "sys_platform": "darwin",
    "platform_machine": "arm64",
    "numpy_blas_name": "accelerate",
}


def _validated_contract():
    return baselines._validate_contract(json.loads(CONTRACT.read_text()))


def test_shared_scoring_audit_preserves_complete_synthetic_baseline_artifact(
    monkeypatch,
):
    rng = np.random.default_rng(521)
    train = rng.normal(size=(100, 25))
    train_labels = np.tile(np.asarray([0, 1], dtype=np.int8), 50)
    validation = rng.normal(size=(420, 25))
    labels = np.asarray([0] * 400 + [1] * 20, dtype=np.int8)
    hashes = {
        name: "0" * 64
        for name in ("contract", "train", "validation", "preparation_summary")
    }
    contract = _validated_contract()
    original = baselines._fit_model(
        "Logistic-L1", train, train_labels, validation, labels, hashes, contract
    )
    scorer = baselines._audited_validation_scores

    def inline_reference(model_name, scaled, classifier, *, policy, environment):
        decisions, decision_warnings = baselines._score_with_warning_policy(
            "decision_function",
            lambda: classifier.decision_function(scaled),
            policy=policy,
            environment=environment,
        )
        probabilities, probability_warnings = baselines._score_with_warning_policy(
            "predict_proba",
            lambda: classifier.predict_proba(scaled),
            policy=policy,
            environment=environment,
        )
        decision_difference, probability_difference = (
            baselines._reference_score_differences(
                model_name, scaled, classifier, decisions, probabilities, policy
            )
        )
        expected_audit = {
            "platform_identity": environment,
            "warning_records": decision_warnings + probability_warnings,
            "max_absolute_decision_difference": decision_difference,
            "max_absolute_probability_difference": probability_difference,
        }
        scores, audit = scorer(
            model_name, scaled, classifier, policy=policy, environment=environment
        )
        np.testing.assert_array_equal(scores, probabilities[:, 1])
        assert audit == expected_audit
        return probabilities[:, 1], expected_audit

    monkeypatch.setattr(baselines, "_audited_validation_scores", inline_reference)
    repeated = baselines._fit_model(
        "Logistic-L1", train, train_labels, validation, labels, hashes, contract
    )
    assert repeated == original


def _warning_operation(
    message="divide by zero encountered in matmul",
    *,
    category=RuntimeWarning,
    module="sklearn.utils.extmath",
):
    def operation():
        warnings.warn_explicit(
            message,
            category,
            filename="synthetic-extmath.py",
            lineno=1,
            module=module,
        )
        return np.asarray([1.0])

    return operation


@pytest.mark.parametrize(
    "message",
    (
        "divide by zero encountered in matmul",
        "overflow encountered in matmul",
        "invalid value encountered in matmul",
    ),
)
@pytest.mark.parametrize("stage", ("decision_function", "predict_proba"))
def test_scoring_policy_captures_each_exact_accelerate_warning(message, stage):
    policy = _validated_contract()["scoring_integrity"]

    value, records = baselines._score_with_warning_policy(
        stage,
        _warning_operation(message),
        policy=policy,
        environment=ACCELERATE_ENVIRONMENT,
    )

    np.testing.assert_array_equal(value, np.asarray([1.0]))
    assert records == [
        {"stage": stage, "category": "RuntimeWarning", "message": message}
    ]


@pytest.mark.parametrize(
    ("stage", "environment", "message", "category", "module"),
    (
        (
            "decision_function",
            {**ACCELERATE_ENVIRONMENT, "sys_platform": "linux"},
            "divide by zero encountered in matmul",
            RuntimeWarning,
            "sklearn.utils.extmath",
        ),
        (
            "decision_function",
            {**ACCELERATE_ENVIRONMENT, "platform_machine": "x86_64"},
            "divide by zero encountered in matmul",
            RuntimeWarning,
            "sklearn.utils.extmath",
        ),
        (
            "decision_function",
            {**ACCELERATE_ENVIRONMENT, "numpy_blas_name": "openblas"},
            "divide by zero encountered in matmul",
            RuntimeWarning,
            "sklearn.utils.extmath",
        ),
        (
            "decision_function",
            ACCELERATE_ENVIRONMENT,
            "divide by zero encountered in matmul!",
            RuntimeWarning,
            "sklearn.utils.extmath",
        ),
        (
            "decision_function",
            ACCELERATE_ENVIRONMENT,
            "divide by zero encountered in matmul",
            UserWarning,
            "sklearn.utils.extmath",
        ),
        (
            "decision_function",
            ACCELERATE_ENVIRONMENT,
            "divide by zero encountered in matmul",
            RuntimeWarning,
            "sklearn.utils.extmath.extra",
        ),
        (
            "fit",
            ACCELERATE_ENVIRONMENT,
            "divide by zero encountered in matmul",
            RuntimeWarning,
            "sklearn.utils.extmath",
        ),
    ),
)
def test_scoring_policy_rejects_every_near_miss(
    stage, environment, message, category, module
):
    with pytest.raises(Warning):
        baselines._score_with_warning_policy(
            stage,
            _warning_operation(message, category=category, module=module),
            policy=_validated_contract()["scoring_integrity"],
            environment=environment,
        )


@pytest.mark.parametrize("column", (0, 1))
def test_reference_check_rejects_disagreement_in_either_probability_column(column):
    scaled = np.asarray([[0.0], [1.0]], dtype=np.float64)
    classifier = SimpleNamespace(
        coef_=np.asarray([[1.0]], dtype=np.float64),
        intercept_=np.asarray([0.0], dtype=np.float64),
    )
    decision = np.asarray([0.0, 1.0], dtype=np.float64)
    positive = baselines.expit(decision)
    probabilities = np.column_stack((1.0 - positive, positive))
    probabilities[0, column] += 1e-4

    with pytest.raises(baselines.BaselineError, match="probability outputs disagree"):
        baselines._reference_score_differences(
            "fixture",
            scaled,
            classifier,
            decision,
            probabilities,
            _validated_contract()["scoring_integrity"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("decision", np.asarray([0.0, math.nan])),
        (
            "probability",
            np.asarray([[0.5, 0.5], [math.inf, -math.inf]]),
        ),
    ),
)
def test_reference_check_rejects_nonfinite_outputs(field, value):
    scaled = np.asarray([[0.0], [1.0]], dtype=np.float64)
    classifier = SimpleNamespace(
        coef_=np.asarray([[1.0]], dtype=np.float64),
        intercept_=np.asarray([0.0], dtype=np.float64),
    )
    decision = np.asarray([0.0, 1.0], dtype=np.float64)
    positive = 1.0 / (1.0 + np.exp(-decision))
    probabilities = np.column_stack((1.0 - positive, positive))

    with pytest.raises(baselines.BaselineError, match="nonfinite"):
        baselines._reference_score_differences(
            "fixture",
            scaled,
            classifier,
            value if field == "decision" else decision,
            value if field == "probability" else probabilities,
            _validated_contract()["scoring_integrity"],
        )


@pytest.mark.parametrize(
    ("decisions", "probabilities", "message"),
    (
        (
            np.asarray([[0.0], [1.0]]),
            np.asarray([[0.5, 0.5], [0.25, 0.75]]),
            "decision-score shape",
        ),
        (
            np.asarray([0.0, 1.0]),
            np.asarray([0.5, 0.75]),
            "probability shape",
        ),
    ),
)
def test_reference_check_rejects_output_shape_changes(
    decisions, probabilities, message
):
    classifier = SimpleNamespace(
        coef_=np.asarray([[1.0]], dtype=np.float64),
        intercept_=np.asarray([0.0], dtype=np.float64),
    )

    with pytest.raises(baselines.BaselineError, match=message):
        baselines._reference_score_differences(
            "fixture",
            np.asarray([[0.0], [1.0]], dtype=np.float64),
            classifier,
            decisions,
            probabilities,
            _validated_contract()["scoring_integrity"],
        )


def test_clopper_pearson_upper_matches_closed_form_boundaries():
    assert baselines.clopper_pearson_upper(0, 400) == pytest.approx(
        1.0 - 0.05 ** (1.0 / 400.0), rel=1e-13
    )
    assert baselines.clopper_pearson_upper(400, 400) == 1.0
    assert baselines.clopper_pearson_upper(1, 400) == pytest.approx(
        0.01180430445398703, rel=1e-12
    )


@pytest.mark.parametrize(
    ("false_positives", "negative_count"),
    ((-1, 400), (401, 400), (0, 0), (True, 400), (0, 400.0)),
)
def test_clopper_pearson_upper_rejects_invalid_exact_counts(
    false_positives, negative_count
):
    with pytest.raises(baselines.BaselineError):
        baselines.clopper_pearson_upper(false_positives, negative_count)


def test_threshold_selection_handles_score_ties_as_one_candidate():
    negative_scores = np.concatenate(
        (np.full(399, 0.1), np.array([0.8], dtype=np.float64))
    )
    positive_scores = np.array([0.2, 0.8, 0.8, 0.9], dtype=np.float64)
    scores = np.concatenate((negative_scores, positive_scores))
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(4, dtype=np.int8)))

    selected = baselines.select_validation_threshold(scores, labels)

    assert selected == {
        "status": "selected",
        "threshold": 0.9,
        "candidate_count": 5,
        "counts": {
            "true_positive": 1,
            "false_positive": 0,
            "true_negative": 400,
            "false_negative": 3,
            "positive": 4,
            "negative": 400,
        },
        "recall": 0.25,
        "observed_fpr": 0.0,
        "fpr_upper_95": pytest.approx(1.0 - 0.05 ** (1.0 / 400.0)),
    }


def test_threshold_selection_uses_smaller_bound_then_higher_threshold_for_ties():
    scores = np.concatenate(
        (
            np.full(400, 0.1),
            np.array([0.3, 0.4, 0.4, 0.7], dtype=np.float64),
        )
    )
    labels = np.concatenate((np.zeros(400, dtype=np.int8), np.ones(4, dtype=np.int8)))

    selected = baselines.select_validation_threshold(scores, labels)

    assert selected["threshold"] == 0.3
    assert selected["recall"] == 1.0
    assert selected["counts"]["false_positive"] == 0


def test_threshold_selection_reports_small_negative_infeasibility():
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 1, 1], dtype=np.int8)

    selected = baselines.select_validation_threshold(scores, labels)

    assert selected == {
        "status": "target_not_met",
        "threshold": None,
        "candidate_count": 5,
        "counts": None,
        "recall": None,
        "observed_fpr": None,
        "fpr_upper_95": None,
    }


@pytest.mark.parametrize(
    ("scores", "labels"),
    (
        ([0.1, math.nan], [0, 1]),
        ([0.1, math.inf], [0, 1]),
        ([-0.1, 0.8], [0, 1]),
        ([0.1, 1.1], [0, 1]),
        ([0.1], [0, 1]),
        ([[0.1], [0.2]], [0, 1]),
        ([0.1, 0.2], [0, 2]),
        ([0.1, 0.2], [0, 0]),
    ),
)
def test_threshold_selection_rejects_invalid_scores_and_labels(scores, labels):
    with pytest.raises(baselines.BaselineError):
        baselines.select_validation_threshold(np.asarray(scores), np.asarray(labels))
