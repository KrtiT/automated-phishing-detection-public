import ast
import importlib.util
import inspect
import json
import struct
import sys
import warnings
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from automated_phishing_detection.url_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "rq1_saga_convergence_diagnostic.py"


@pytest.fixture
def diagnostic():
    spec = importlib.util.spec_from_file_location("rq1_saga_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


def _synthetic_training():
    generator = np.random.default_rng(20260904)
    labels = np.tile(np.asarray([0, 1], dtype=np.int8), 64)
    features = generator.normal(scale=0.25, size=(labels.size, len(FEATURE_NAMES)))
    features[:, 0] += np.where(labels == 1, 2.0, -2.0)
    return features, labels


def _encoded_float(values, shape):
    flat = np.asarray(values, dtype=np.dtype("<f8")).reshape(-1)
    return {
        "dtype": "<f8",
        "hex": flat.tobytes(order="C").hex(),
        "shape": list(shape),
    }


def _encoded_int(values, shape):
    flat = np.asarray(values, dtype=np.dtype("<i8")).reshape(-1)
    return {
        "dtype": "<i8",
        "hex": flat.tobytes(order="C").hex(),
        "shape": list(shape),
    }


def _context_record(diagnostic, environment):
    return {
        "configuration": {
            "classifier": diagnostic.CLASSIFIER_CONFIG,
            "scaler": diagnostic.SCALER_CONFIG,
        },
        "diagnostic_id": "rq1-saga-convergence-v2",
        "environment": environment,
        "input_hashes": diagnostic.INPUT_HASHES,
        "model_features": {
            name: list(names) for name, names in diagnostic.MODEL_FEATURES.items()
        },
        "schema_version": 1,
        "scope": "training_only",
    }


def _planned_record(diagnostic, environment, models):
    return {
        **_context_record(diagnostic, environment),
        "models": models,
    }


def _model_aggregates(diagnostic):
    return {
        model_name: {
            "elapsed_seconds": float(len(feature_names)),
            "feature_count": len(feature_names),
            "allowed_warnings": [],
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
            "n_iter": len(feature_names) + 1,
            "nonzero_coefficient_count": len(feature_names),
            "state_sha256": f"{len(feature_names):064x}",
        }
        for model_name, feature_names in diagnostic.MODEL_FEATURES.items()
    }


def _successful_run(diagnostic, environment):
    return {
        **_planned_record(diagnostic, environment, _model_aggregates(diagnostic)),
        "run_passed": True,
    }


def test_script_has_only_the_frozen_training_input_surface(diagnostic):
    assert tuple(inspect.signature(diagnostic.main).parameters) == ()
    assert diagnostic.TRAIN_PATH == ROOT / "data/processed/phiusiil-v1/train.jsonl"
    assert diagnostic.PREPARATION_SUMMARY_PATH == (
        ROOT / "reports/phiusiil-preparation-summary.json"
    )
    assert diagnostic.CONTRACT_PATH == ROOT / "data/rq1-baseline-contract.json"
    assert diagnostic.UV_LOCK_SHA256 == (
        "15fadb4ad1f3c702a902b40d587a55294e7a26c33f268e8708ba8d941e6a51f0"
    )
    assert diagnostic.INPUT_PATHS == {
        "train": diagnostic.TRAIN_PATH,
        "preparation_summary": diagnostic.PREPARATION_SUMMARY_PATH,
    }
    assert diagnostic.INPUT_HASHES == {
        "contract": (
            "594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4"
        ),
        "preparation_summary": (
            "1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e"
        ),
        "train": ("575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0"),
    }
    assert not any(
        forbidden in str(path).lower()
        for path in diagnostic.INPUT_PATHS.values()
        for forbidden in ("validation", "group_test", "group-test", "phishvn")
    )


def test_script_has_no_file_output_interface(diagnostic):
    path_constants = {
        name
        for name, value in vars(diagnostic).items()
        if name.isupper() and isinstance(value, Path)
    }
    assert path_constants == {
        "CONTRACT_PATH",
        "PREPARATION_SUMMARY_PATH",
        "REPOSITORY",
        "TRAIN_PATH",
        "UV_LOCK_PATH",
    }

    syntax = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    open_modes = []
    mutating_methods = {
        "dump",
        "hardlink_to",
        "mkdir",
        "rename",
        "replace",
        "save",
        "savetxt",
        "symlink_to",
        "tofile",
        "touch",
        "unlink",
        "write",
        "write_bytes",
        "write_text",
    }
    for node in ast.walk(syntax):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            pytest.fail("built-in open is not permitted")
        if not isinstance(node.func, ast.Attribute):
            continue
        assert node.func.attr not in mutating_methods
        if node.func.attr == "open":
            assert node.args and isinstance(node.args[0], ast.Constant)
            open_modes.append(node.args[0].value)
    assert open_modes == ["rb"]


def test_contract_hash_is_checked_before_training_data_is_opened(
    diagnostic, monkeypatch
):
    monkeypatch.setattr(diagnostic, "_sha256_path", lambda path: "0" * 64)
    monkeypatch.setattr(
        diagnostic.baselines,
        "_open_input_streams",
        lambda *args: pytest.fail("data must not open after a contract hash failure"),
    )

    with pytest.raises(RuntimeError, match="contract hash does not match"):
        diagnostic._load_training_partition()


def test_candidate_configuration_and_model_features_are_exact(diagnostic):
    assert diagnostic.DIAGNOSTIC_ID == "rq1-saga-convergence-v2"
    assert diagnostic.SCALER_CONFIG == {
        "class": "StandardScaler",
        "with_mean": True,
        "with_std": True,
    }
    assert diagnostic.CLASSIFIER_CONFIG == {
        "class": "LogisticRegression",
        "penalty": "l1",
        "solver": "saga",
        "C": 1.0,
        "class_weight": "balanced",
        "fit_intercept": True,
        "max_iter": 5000,
        "tol": 1e-4,
        "random_state": 42,
    }
    assert tuple(diagnostic.MODEL_FEATURES) == ("length-only", "Logistic-L1")
    assert diagnostic.MODEL_FEATURES["length-only"] == ("raw_url_codepoint_length",)
    assert diagnostic.MODEL_FEATURES["Logistic-L1"] == FEATURE_NAMES


@pytest.mark.parametrize(
    ("sys_platform", "machine", "blas_name", "expected"),
    (
        ("darwin", "arm64", "accelerate", True),
        ("linux", "arm64", "accelerate", False),
        ("darwin", "aarch64", "accelerate", False),
        ("darwin", "arm64", "Accelerate", False),
        ("darwin", "arm64", "openblas", False),
    ),
)
def test_scoring_warning_platform_predicate_is_exact(
    diagnostic, sys_platform, machine, blas_name, expected
):
    environment = {
        "sys_platform": sys_platform,
        "platform_machine": machine,
        "numpy_blas_name": blas_name,
    }

    assert diagnostic._allows_accelerate_scoring_warnings(environment) is expected


def test_git_identity_reports_platform_machine_and_numpy_blas(diagnostic, monkeypatch):
    def git_output(arguments):
        if arguments[0] == "rev-parse":
            return "a" * 40
        if arguments[0] == "status":
            return ""
        return arguments[-1]

    monkeypatch.setattr(diagnostic, "_git_output", git_output)
    monkeypatch.setattr(
        diagnostic, "_sha256_path", lambda path: diagnostic.UV_LOCK_SHA256
    )
    monkeypatch.setattr(diagnostic.sys, "platform", "darwin")
    monkeypatch.setattr(diagnostic.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(diagnostic, "_numpy_blas_name", lambda: "accelerate")

    assert diagnostic._git_identity() == {
        "git_head": "a" * 40,
        "numpy_blas_name": "accelerate",
        "platform_machine": "arm64",
        "sys_platform": "darwin",
        "tracked_worktree_clean": True,
        "uv_lock_sha256": diagnostic.UV_LOCK_SHA256,
    }


def test_canonical_state_digest_uses_fixed_order_and_normalized_bytes(diagnostic):
    scaler = SimpleNamespace(
        mean_=np.asarray([1.5], dtype=">f4"),
        scale_=np.asarray([2.25], dtype=">f4"),
        var_=np.asarray([5.0625], dtype=">f4"),
        n_samples_seen_=np.asarray(3, dtype=">i4"),
    )
    classifier = SimpleNamespace(
        classes_=np.asarray([0, 1], dtype=">i2"),
        coef_=np.asarray([[-2.5]], dtype=">f4"),
        intercept_=np.asarray([0.125], dtype=">f4"),
        n_iter_=np.asarray([17], dtype=">i4"),
    )
    ordered_state = [
        {"name": "model_name", "value": "length-only"},
        {
            "name": "features",
            "value": ["raw_url_codepoint_length"],
        },
        {"name": "scaler_config", "value": diagnostic.SCALER_CONFIG},
        {"name": "classifier_config", "value": diagnostic.CLASSIFIER_CONFIG},
        {"name": "input_hashes", "value": diagnostic.INPUT_HASHES},
        {
            "name": "scaler.mean_",
            "value": _encoded_float([1.5], (1,)),
        },
        {
            "name": "scaler.scale_",
            "value": _encoded_float([2.25], (1,)),
        },
        {
            "name": "scaler.var_",
            "value": _encoded_float([5.0625], (1,)),
        },
        {
            "name": "scaler.n_samples_seen_",
            "value": _encoded_int([3], ()),
        },
        {
            "name": "classifier.classes_",
            "value": _encoded_int([0, 1], (2,)),
        },
        {
            "name": "classifier.coef_",
            "value": _encoded_float([-2.5], (1, 1)),
        },
        {
            "name": "classifier.intercept_",
            "value": _encoded_float([0.125], (1,)),
        },
        {
            "name": "classifier.n_iter_",
            "value": _encoded_int([17], (1,)),
        },
    ]
    canonical = json.dumps(
        {"ordered_state": ordered_state, "schema_version": 1},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")

    actual = diagnostic._state_sha256(
        "length-only",
        ("raw_url_codepoint_length",),
        scaler,
        classifier,
    )

    assert actual == sha256(canonical).hexdigest()
    assert actual == "1e1afd7ac2fe320f29d9470d8a2ee89feafae84c1009ec676bc1bd960a156158"
    assert _encoded_float([1.5], (1,))["hex"] == struct.pack("<d", 1.5).hex()


def test_fit_candidate_constructs_exact_estimators_and_returns_aggregates(
    diagnostic, monkeypatch
):
    real_scaler = StandardScaler
    real_classifier = LogisticRegression
    observed = {}

    def scaler_factory(**kwargs):
        observed["scaler"] = kwargs
        return real_scaler(**kwargs)

    def classifier_factory(**kwargs):
        observed["classifier"] = kwargs
        return real_classifier(**kwargs)

    monkeypatch.setattr(diagnostic, "StandardScaler", scaler_factory)
    monkeypatch.setattr(diagnostic, "LogisticRegression", classifier_factory)
    features, labels = _synthetic_training()

    result = diagnostic._fit_candidate("length-only", features, labels)

    assert observed == {
        "scaler": {"with_mean": True, "with_std": True},
        "classifier": {
            "penalty": "l1",
            "solver": "saga",
            "C": 1.0,
            "class_weight": "balanced",
            "fit_intercept": True,
            "max_iter": 5000,
            "tol": 1e-4,
            "random_state": 42,
        },
    }
    assert set(result) == {
        "allowed_warnings",
        "elapsed_seconds",
        "feature_count",
        "max_absolute_decision_difference",
        "max_absolute_probability_difference",
        "n_iter",
        "nonzero_coefficient_count",
        "state_sha256",
    }
    assert result["elapsed_seconds"] >= 0.0
    assert result["feature_count"] == 1
    assert result["allowed_warnings"] == []
    assert result["max_absolute_decision_difference"] <= 1e-12
    assert result["max_absolute_probability_difference"] <= 1e-12
    assert 0 < result["n_iter"] < 5000
    assert 0 <= result["nonzero_coefficient_count"] <= 1
    assert len(result["state_sha256"]) == 64


def test_fit_candidate_treats_every_warning_as_fatal(diagnostic, monkeypatch):
    class WarningClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, features, labels):
            warnings.warn("forced convergence warning", ConvergenceWarning)

    monkeypatch.setattr(diagnostic, "LogisticRegression", WarningClassifier)
    features, labels = _synthetic_training()

    with pytest.raises(
        ConvergenceWarning, match="forced convergence warning"
    ) as raised:
        diagnostic._fit_candidate("length-only", features, labels)
    assert raised.value.failure_stage == "fit"


def test_scaling_warning_is_fatal_even_when_it_matches_scoring_allowlist(
    diagnostic, monkeypatch
):
    class WarningScaler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, features):
            warnings.warn_explicit(
                "invalid value encountered in matmul",
                RuntimeWarning,
                filename="sklearn/utils/extmath.py",
                lineno=1,
                module="sklearn.utils.extmath",
            )

    monkeypatch.setattr(diagnostic, "StandardScaler", WarningScaler)
    features, labels = _synthetic_training()

    with pytest.raises(RuntimeWarning, match="invalid value") as raised:
        diagnostic._fit_candidate(
            "length-only",
            features,
            labels,
            {
                "sys_platform": "darwin",
                "platform_machine": "arm64",
                "numpy_blas_name": "accelerate",
            },
        )
    assert raised.value.failure_stage == "scaling"


@pytest.mark.parametrize("stage", ("decision_function", "predict_proba"))
@pytest.mark.parametrize(
    "message",
    (
        "divide by zero encountered in matmul",
        "overflow encountered in matmul",
        "invalid value encountered in matmul",
    ),
)
def test_exact_accelerate_scoring_warnings_are_captured(diagnostic, stage, message):
    def operation():
        warnings.warn_explicit(
            message,
            RuntimeWarning,
            filename="sklearn/utils/extmath.py",
            lineno=1,
            module="sklearn.utils.extmath",
        )
        return np.asarray([1.0])

    value, allowed_warnings = diagnostic._score_with_warning_policy(
        stage, operation, allow_accelerate_warning=True
    )

    assert value.tolist() == [1.0]
    assert allowed_warnings == [
        {
            "category": "RuntimeWarning",
            "message": message,
            "stage": stage,
        }
    ]


@pytest.mark.parametrize(
    ("stage", "message", "category", "module", "allow", "expected_category"),
    (
        (
            "decision_function",
            "unexpected matmul warning",
            RuntimeWarning,
            "sklearn.utils.extmath",
            True,
            RuntimeWarning,
        ),
        (
            "decision_function",
            "invalid value encountered in matmul",
            RuntimeWarning,
            "sklearn.linear_model._base",
            True,
            RuntimeWarning,
        ),
        (
            "predict_proba",
            "invalid value encountered in matmul",
            UserWarning,
            "sklearn.utils.extmath",
            True,
            UserWarning,
        ),
        (
            "fit",
            "invalid value encountered in matmul",
            RuntimeWarning,
            "sklearn.utils.extmath",
            True,
            RuntimeWarning,
        ),
        (
            "decision_function",
            "invalid value encountered in matmul",
            RuntimeWarning,
            "sklearn.utils.extmath",
            False,
            RuntimeWarning,
        ),
    ),
)
def test_every_nonallowed_warning_remains_fatal(
    diagnostic, stage, message, category, module, allow, expected_category
):
    def operation():
        warnings.warn_explicit(
            message,
            category,
            filename="warning-origin.py",
            lineno=1,
            module=module,
        )
        return np.asarray([1.0])

    with pytest.raises(expected_category, match=message) as raised:
        diagnostic._score_with_warning_policy(
            stage, operation, allow_accelerate_warning=allow
        )
    assert raised.value.failure_stage == stage


@pytest.mark.parametrize(
    "disagreement",
    ("decision", "probability_column_0", "probability_column_1"),
)
def test_reference_scoring_disagreement_is_fatal(diagnostic, disagreement):
    scaled = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    classifier = SimpleNamespace(
        coef_=np.asarray([[2.0]], dtype=np.float64),
        intercept_=np.asarray([0.5], dtype=np.float64),
    )
    reference_decision = (
        np.einsum("ij,j->i", scaled, classifier.coef_[0], optimize=False)
        + classifier.intercept_[0]
    )
    decision_scores = reference_decision.copy()
    positive_probability = expit(reference_decision)
    probabilities = np.column_stack((1.0 - positive_probability, positive_probability))
    if disagreement == "decision":
        decision_scores[0] += 1e-6
    else:
        probability_column = int(disagreement[-1])
        probabilities[0, probability_column] += 1e-6

    expected_output = "decision" if disagreement == "decision" else "probability"
    with pytest.raises(RuntimeError, match=f"{expected_output}.*disagree") as raised:
        diagnostic._reference_score_differences(
            "length-only",
            scaled,
            classifier,
            decision_scores,
            probabilities,
        )
    assert raised.value.failure_stage == "reference_check"


def test_probability_difference_aggregate_uses_both_columns(diagnostic):
    scaled = np.asarray([[-1.0], [0.0], [1.0]], dtype=np.float64)
    classifier = SimpleNamespace(
        coef_=np.asarray([[2.0]], dtype=np.float64),
        intercept_=np.asarray([0.5], dtype=np.float64),
    )
    decision_scores = (
        np.einsum("ij,j->i", scaled, classifier.coef_[0], optimize=False)
        + classifier.intercept_[0]
    )
    positive_probability = expit(decision_scores)
    probabilities = np.column_stack((1.0 - positive_probability, positive_probability))
    probabilities[0, 0] += 5e-13
    probabilities[0, 1] += 1e-13

    _, probability_difference = diagnostic._reference_score_differences(
        "length-only",
        scaled,
        classifier,
        decision_scores,
        probabilities,
    )

    assert probability_difference == abs(
        probabilities[0, 0] - (1.0 - positive_probability[0])
    )


def test_allowed_scoring_warning_is_retained_in_model_aggregate(
    diagnostic, monkeypatch
):
    class WarningClassifier(LogisticRegression):
        def decision_function(self, features):
            if not getattr(self, "_warning_emitted", False):
                self._warning_emitted = True
                warnings.warn_explicit(
                    "overflow encountered in matmul",
                    RuntimeWarning,
                    filename="sklearn/utils/extmath.py",
                    lineno=1,
                    module="sklearn.utils.extmath",
                )
            return super().decision_function(features)

    monkeypatch.setattr(diagnostic, "LogisticRegression", WarningClassifier)
    features, labels = _synthetic_training()

    result = diagnostic._fit_candidate(
        "length-only",
        features,
        labels,
        {
            "sys_platform": "darwin",
            "platform_machine": "arm64",
            "numpy_blas_name": "accelerate",
        },
    )

    assert result["allowed_warnings"] == [
        {
            "category": "RuntimeWarning",
            "message": "overflow encountered in matmul",
            "stage": "decision_function",
        }
    ]
    assert not any(isinstance(value, np.ndarray) for value in result.values())


def test_later_scoring_failure_retains_prior_allowed_warning(diagnostic, monkeypatch):
    class WarningClassifier(LogisticRegression):
        def decision_function(self, features):
            warnings.warn_explicit(
                "overflow encountered in matmul",
                RuntimeWarning,
                filename="sklearn/utils/extmath.py",
                lineno=1,
                module="sklearn.utils.extmath",
            )
            return super().decision_function(features)

        def predict_proba(self, features):
            warnings.warn("fatal probability warning", UserWarning)

    monkeypatch.setattr(diagnostic, "LogisticRegression", WarningClassifier)
    features, labels = _synthetic_training()

    with pytest.raises(UserWarning, match="fatal probability warning") as raised:
        diagnostic._fit_candidate(
            "length-only",
            features,
            labels,
            {
                "sys_platform": "darwin",
                "platform_machine": "arm64",
                "numpy_blas_name": "accelerate",
            },
        )

    assert raised.value.failure_stage == "predict_proba"
    assert raised.value.allowed_warnings == [
        {
            "category": "RuntimeWarning",
            "message": "overflow encountered in matmul",
            "stage": "decision_function",
        }
    ]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("classes_", np.asarray([1, 0]), "classes"),
        ("n_iter_", np.asarray([5000]), "below max_iter"),
        ("coef_", np.asarray([[1.0, 2.0]]), "coefficient shape"),
        ("intercept_", np.asarray([np.inf]), "nonfinite"),
    ),
)
def test_candidate_validation_rejects_invalid_fitted_state(
    diagnostic, field, value, message
):
    scaler = SimpleNamespace(
        mean_=np.asarray([0.0]),
        scale_=np.asarray([1.0]),
        var_=np.asarray([1.0]),
        n_samples_seen_=np.asarray(4),
    )
    classifier = SimpleNamespace(
        classes_=np.asarray([0, 1]),
        coef_=np.asarray([[1.0]]),
        intercept_=np.asarray([0.0]),
        n_iter_=np.asarray([2]),
    )
    setattr(classifier, field, value)

    with pytest.raises(RuntimeError, match=message):
        diagnostic._candidate_summary(
            "length-only",
            ("raw_url_codepoint_length",),
            scaler,
            classifier,
            np.zeros((4, 1)),
            np.zeros(4),
            np.full((4, 2), 0.5),
            1.0,
            [],
            0.0,
            0.0,
        )


def test_execute_reports_two_model_aggregate_and_stable_git_identity(
    diagnostic, monkeypatch
):
    features, labels = _synthetic_training()
    identity = {
        "git_head": "a" * 40,
        "tracked_worktree_clean": True,
        "uv_lock_sha256": "b" * 64,
    }
    identity_calls = []
    model_calls = []

    def git_identity():
        identity_calls.append(True)
        return identity.copy()

    def fit_candidate(model_name, actual_features, actual_labels, environment):
        assert actual_features is features
        assert actual_labels is labels
        assert environment is identity or environment == identity
        model_calls.append(model_name)
        width = len(diagnostic.MODEL_FEATURES[model_name])
        return {
            "allowed_warnings": [],
            "elapsed_seconds": float(width),
            "feature_count": width,
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
            "n_iter": width + 1,
            "nonzero_coefficient_count": width,
            "state_sha256": f"{width:064x}",
        }

    monkeypatch.setattr(diagnostic, "_git_identity", git_identity)
    monkeypatch.setattr(
        diagnostic, "_load_training_partition", lambda: (features, labels)
    )
    monkeypatch.setattr(diagnostic, "_fit_candidate", fit_candidate)

    result = diagnostic._execute_single_run()

    assert identity_calls == [True, True]
    assert model_calls == ["length-only", "Logistic-L1"]
    expected_models = {
        "length-only": {
            "allowed_warnings": [],
            "elapsed_seconds": 1.0,
            "feature_count": 1,
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
            "n_iter": 2,
            "nonzero_coefficient_count": 1,
            "state_sha256": f"{1:064x}",
        },
        "Logistic-L1": {
            "allowed_warnings": [],
            "elapsed_seconds": 25.0,
            "feature_count": 25,
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
            "n_iter": 26,
            "nonzero_coefficient_count": 25,
            "state_sha256": f"{25:064x}",
        },
    }
    assert result == {
        **_planned_record(diagnostic, identity, expected_models),
        "run_passed": True,
    }
    assert "status" not in result


def test_model_failure_retains_planned_and_initial_environment_context(
    diagnostic, monkeypatch
):
    features, labels = _synthetic_training()
    identity = {
        "git_head": "a" * 40,
        "tracked_worktree_clean": True,
        "uv_lock_sha256": diagnostic.UV_LOCK_SHA256,
    }
    completed = {
        "allowed_warnings": [],
        "elapsed_seconds": 1.0,
        "feature_count": 1,
        "max_absolute_decision_difference": 0.0,
        "max_absolute_probability_difference": 0.0,
        "n_iter": 2,
        "nonzero_coefficient_count": 1,
        "state_sha256": "c" * 64,
    }

    def fit_candidate(model_name, actual_features, actual_labels, environment):
        assert actual_features is features
        assert actual_labels is labels
        if model_name == "Logistic-L1":
            error = ConvergenceWarning("forced full-model warning")
            error.failure_stage = "fit"
            raise error
        return completed

    monkeypatch.setattr(diagnostic, "_git_identity", lambda: identity.copy())
    monkeypatch.setattr(
        diagnostic, "_load_training_partition", lambda: (features, labels)
    )
    monkeypatch.setattr(diagnostic, "_fit_candidate", fit_candidate)

    result = diagnostic._execute_single_run()

    assert result == {
        **_planned_record(diagnostic, identity, {"length-only": completed}),
        "failure": {
            "message": "forced full-model warning",
            "model_name": "Logistic-L1",
            "stage": "fit",
            "type": "ConvergenceWarning",
        },
        "run_passed": False,
    }
    assert "status" not in result


def test_execute_rejects_git_identity_changes_during_fit(diagnostic, monkeypatch):
    identities = [
        {
            "git_head": "a" * 40,
            "tracked_worktree_clean": True,
            "uv_lock_sha256": "b" * 64,
        },
        {
            "git_head": "c" * 40,
            "tracked_worktree_clean": True,
            "uv_lock_sha256": "b" * 64,
        },
    ]
    features, labels = _synthetic_training()
    monkeypatch.setattr(diagnostic, "_git_identity", lambda: identities.pop(0))
    monkeypatch.setattr(
        diagnostic, "_load_training_partition", lambda: (features, labels)
    )
    monkeypatch.setattr(
        diagnostic,
        "_fit_candidate",
        lambda *args: {
            "allowed_warnings": [],
            "elapsed_seconds": 1.0,
            "feature_count": 1,
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
            "n_iter": 1,
            "nonzero_coefficient_count": 1,
            "state_sha256": "0" * 64,
        },
    )

    result = diagnostic._execute_single_run()

    assert result["run_passed"] is False
    assert "status" not in result
    assert result["environment"] == {
        "git_head": "a" * 40,
        "tracked_worktree_clean": True,
        "uv_lock_sha256": "b" * 64,
    }
    assert result["failure"] == {
        "message": "Git identity changed during the diagnostic",
        "stage": "environment",
        "type": "RuntimeError",
    }


def test_coordinator_runs_exactly_two_fresh_processes_before_passing(diagnostic):
    identity = {
        "git_head": "a" * 40,
        "tracked_worktree_clean": True,
        "uv_lock_sha256": diagnostic.UV_LOCK_SHA256,
    }
    expected_runs = [
        _successful_run(diagnostic, identity.copy()),
        _successful_run(diagnostic, identity.copy()),
    ]
    pending_runs = iter(expected_runs)
    calls = []

    def fresh_process():
        calls.append(len(calls) + 1)
        return next(pending_runs)

    result = diagnostic._coordinate_fresh_runs(fresh_process)

    assert calls == [1, 2]
    assert result == {
        **_context_record(diagnostic, identity),
        "runs": expected_runs,
        "status": "passed",
    }
    assert all("status" not in run for run in result["runs"])


@pytest.mark.parametrize(
    ("model_name", "field", "replacement"),
    (
        ("length-only", "n_iter", 91),
        ("length-only", "state_sha256", "a" * 64),
        ("Logistic-L1", "n_iter", 92),
        ("Logistic-L1", "state_sha256", "b" * 64),
    ),
)
def test_coordinator_fails_for_each_required_fresh_run_mismatch(
    diagnostic, model_name, field, replacement
):
    identity = {
        "git_head": "a" * 40,
        "tracked_worktree_clean": True,
        "uv_lock_sha256": diagnostic.UV_LOCK_SHA256,
    }
    first_run = _successful_run(diagnostic, identity.copy())
    second_run = _successful_run(diagnostic, identity.copy())
    first_value = first_run["models"][model_name][field]
    second_run["models"][model_name][field] = replacement
    pending_runs = iter((first_run, second_run))

    result = diagnostic._coordinate_fresh_runs(lambda: next(pending_runs))

    assert result["status"] == "failed"
    assert result["failure"] == {
        "message": "fresh-run model aggregates do not match",
        "mismatches": [
            {
                "field": field,
                "model_name": model_name,
                "run_1": first_value,
                "run_2": replacement,
            }
        ],
        "type": "FreshRunMismatch",
    }


def test_coordinator_propagates_child_failure_after_both_processes(diagnostic):
    identity = {
        "git_head": "a" * 40,
        "tracked_worktree_clean": True,
        "uv_lock_sha256": diagnostic.UV_LOCK_SHA256,
    }
    first_run = _successful_run(diagnostic, identity.copy())
    child_failure = {
        **_planned_record(diagnostic, identity.copy(), {}),
        "failure": {
            "message": "forced child failure",
            "type": "RuntimeError",
        },
        "run_passed": False,
    }
    pending_runs = iter((first_run, child_failure))
    calls = []

    def fresh_process():
        calls.append(True)
        return next(pending_runs)

    result = diagnostic._coordinate_fresh_runs(fresh_process)

    assert calls == [True, True]
    assert result["runs"] == [first_run, child_failure]
    assert result["failure"] == {
        "message": "one or more fresh runs failed",
        "runs": [
            {
                "failure": child_failure["failure"],
                "run_number": 2,
            }
        ],
        "type": "FreshRunFailure",
    }
    assert result["status"] == "failed"


def test_fresh_process_runner_explicitly_uses_spawn_context(diagnostic, monkeypatch):
    identity = {
        "git_head": "a" * 40,
        "tracked_worktree_clean": True,
        "uv_lock_sha256": diagnostic.UV_LOCK_SHA256,
    }
    expected = _successful_run(diagnostic, identity)
    events = []

    class ReceivingConnection:
        def recv(self):
            events.append("recv")
            return expected

        def close(self):
            events.append("receiving-close")

    class SendingConnection:
        def close(self):
            events.append("sending-close")

    class SpawnedProcess:
        exitcode = 0

        def start(self):
            events.append("start")

        def join(self):
            events.append("join")

    class SpawnContext:
        def Pipe(self, *, duplex):
            assert duplex is False
            return ReceivingConnection(), SendingConnection()

        def Process(self, *, target, args):
            assert target is diagnostic._child_process_entry
            assert len(args) == 1
            events.append("process-created")
            return SpawnedProcess()

    def get_context(method):
        events.append(("context", method))
        return SpawnContext()

    monkeypatch.setattr(diagnostic.multiprocessing, "get_context", get_context)

    result = diagnostic._run_fresh_process()

    assert result == expected
    assert events == [
        ("context", "spawn"),
        "process-created",
        "start",
        "sending-close",
        "recv",
        "receiving-close",
        "join",
    ]


def test_git_identity_requires_a_clean_tracked_worktree(diagnostic, monkeypatch):
    calls = []

    def git_output(arguments):
        calls.append(tuple(arguments))
        if arguments[0] == "rev-parse":
            return "a" * 40
        return " M docs/result.md"

    monkeypatch.setattr(diagnostic, "_git_output", git_output)
    monkeypatch.setattr(diagnostic, "_sha256_path", lambda path: "b" * 64)

    with pytest.raises(RuntimeError, match="tracked worktree must be clean"):
        diagnostic._git_identity()

    assert calls == [
        ("rev-parse", "HEAD"),
        ("status", "--porcelain=v1", "--untracked-files=no"),
    ]


def test_git_identity_requires_the_executable_and_lock_in_head(diagnostic, monkeypatch):
    calls = []

    def git_output(arguments):
        calls.append(tuple(arguments))
        if arguments[0] == "rev-parse":
            return "a" * 40
        if arguments[0] == "status":
            return ""
        return ""

    monkeypatch.setattr(diagnostic, "_git_output", git_output)
    monkeypatch.setattr(diagnostic, "_sha256_path", lambda path: "b" * 64)

    with pytest.raises(RuntimeError, match="diagnostic executable must be tracked"):
        diagnostic._git_identity()

    assert calls == [
        ("rev-parse", "HEAD"),
        ("status", "--porcelain=v1", "--untracked-files=no"),
        (
            "ls-files",
            "--error-unmatch",
            "--",
            "scripts/rq1_saga_convergence_diagnostic.py",
        ),
    ]


def test_git_identity_rejects_a_nonfrozen_uv_lock(diagnostic, monkeypatch):
    def git_output(arguments):
        if arguments[0] == "rev-parse":
            return "a" * 40
        if arguments[0] == "status":
            return ""
        return arguments[-1]

    monkeypatch.setattr(diagnostic, "_git_output", git_output)
    monkeypatch.setattr(diagnostic, "_sha256_path", lambda path: "b" * 64)

    with pytest.raises(RuntimeError, match="uv.lock hash does not match"):
        diagnostic._git_identity()


def test_main_rejects_arguments_before_loading_inputs(diagnostic, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--train", "anything"])
    monkeypatch.setattr(
        diagnostic,
        "_execute_diagnostic",
        lambda: pytest.fail("arguments must be rejected before execution"),
    )

    return_code = diagnostic.main()
    captured = capsys.readouterr()

    assert return_code == 2
    assert captured.err == ""
    expected = {
        **_context_record(diagnostic, None),
        "failure": {
            "message": "this diagnostic accepts no arguments",
            "type": "UsageError",
        },
        "runs": [],
        "status": "failed",
    }
    assert json.loads(captured.out) == expected
    assert captured.out == (
        json.dumps(json.loads(captured.out), separators=(",", ":"), sort_keys=True)
        + "\n"
    )
