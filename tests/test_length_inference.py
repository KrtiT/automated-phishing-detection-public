import json
from copy import deepcopy
from hashlib import sha256

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from automated_phishing_detection import baselines, fixed_cascade
from automated_phishing_detection.url_features import (
    FEATURE_NAMES,
    extract_url_features,
)


def _artifact():
    return {
        "schema_version": 2,
        "artifact_type": "rq1-baseline-model",
        "analysis_stage": "development_validation_only",
        "contract_id": "rq1-baselines-v2",
        "contract_sha256": "0" * 64,
        "model_name": "length-only",
        "features": ["raw_url_codepoint_length"],
        "classes": [0, 1],
        "scaler": {
            "config": dict(baselines._SCALER_CONFIG),
            "mean": [20.0],
            "scale": [4.0],
            "variance": [16.0],
            "n_samples_seen": 20,
        },
        "classifier": {
            "config": dict(baselines._CLASSIFIER_CONFIG),
            "coefficients": [[0.8]],
            "intercept": [-0.3],
            "n_iter": [12],
        },
        "validation_scoring_audit": {
            "platform_identity": baselines._platform_identity(),
            "warning_records": [],
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
        },
        "validation_threshold": {
            "status": "selected",
            "threshold": 0.5,
            "candidate_count": 3,
            "counts": {
                "true_positive": 1,
                "false_positive": 0,
                "true_negative": 400,
                "false_negative": 0,
                "positive": 1,
                "negative": 400,
            },
            "recall": 1.0,
            "observed_fpr": 0.0,
            "fpr_upper_95": baselines.clopper_pearson_upper(0, 400),
        },
        "input_hashes": {
            "train": "1" * 64,
            "validation": "2" * 64,
            "preparation_summary": "3" * 64,
            "contract": "0" * 64,
        },
        "software_versions": baselines._software_versions(),
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
    }


def _content(artifact=None):
    value = _artifact() if artifact is None else artifact
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("ascii")


def _load(content=None):
    from automated_phishing_detection import length_inference

    content = _content() if content is None else content
    return length_inference._load_length_only_artifact_bytes(
        content,
        expected_sha256=sha256(content).hexdigest(),
        expected_contract_sha256="0" * 64,
    )


def test_length_loader_preserves_hash_bound_state_and_threshold():
    model = _load()

    assert model.artifact_sha256 == sha256(_content()).hexdigest()
    assert model.contract_sha256 == "0" * 64
    assert model.feature_names == ("raw_url_codepoint_length",)
    assert model.validation_threshold_record == _artifact()["validation_threshold"]
    changed = model.validation_threshold_record
    changed["counts"]["true_positive"] = 999
    assert model.validation_threshold_record == _artifact()["validation_threshold"]


def test_public_loader_matches_explicit_synthetic_bytes_and_binds_defaults(tmp_path):
    from automated_phishing_detection import length_inference

    path = tmp_path / "length-only.json"
    path.write_bytes(_content())

    model = length_inference.load_length_only_artifact(
        path,
        expected_sha256=sha256(_content()).hexdigest(),
        expected_contract_sha256="0" * 64,
    )

    assert model == _load()
    with pytest.raises(length_inference.LengthInferenceError, match="SHA-256 mismatch"):
        length_inference.load_length_only_artifact(path)
    assert length_inference.OFFICIAL_LENGTH_ONLY_SHA256 == (
        "b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799"
    )


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_public_loader_rejects_nonregular_paths(tmp_path, kind):
    from automated_phishing_detection import length_inference

    path = tmp_path / "length-only.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        target = tmp_path / "fixture.json"
        target.write_bytes(_content())
        path.symlink_to(target)
    with pytest.raises(length_inference.LengthInferenceError):
        length_inference.load_length_only_artifact(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update(schema_version=True),
        lambda value: value.update(model_name="Logistic-L1"),
        lambda value: value.update(features=list(FEATURE_NAMES)),
        lambda value: value.update(features=["raw_url_utf8_byte_length"]),
        lambda value: value.update(classes=[False, True]),
        lambda value: value.update(extra=True),
        lambda value: value["scaler"].update(mean=[0.0, 0.0]),
        lambda value: value["scaler"].update(scale=[0.0]),
        lambda value: value["scaler"].update(variance=[-1.0]),
        lambda value: value["scaler"].update(n_samples_seen=True),
        lambda value: value["scaler"]["config"].update(with_mean=False),
        lambda value: value["classifier"]["config"].update(penalty="l2"),
        lambda value: value["classifier"].update(coefficients=[[0.1, 0.2]]),
        lambda value: value["classifier"].update(intercept=[float("inf")]),
        lambda value: value["classifier"].update(intercept=[10**1000]),
        lambda value: value["classifier"].update(n_iter=[5000]),
        lambda value: value["validation_threshold"].update(threshold=float("nan")),
        lambda value: value["validation_threshold"].update(recall=0.0),
        lambda value: value["validation_threshold"].update(fpr_upper_95=0.02),
        lambda value: value["input_hashes"].update(train="not-a-hash"),
        lambda value: value["input_hashes"].update(contract="f" * 64),
        lambda value: value["software_versions"].update(numpy=""),
        lambda value: value["access"].update(group_test_accessed=True),
        lambda value: value["validation_scoring_audit"].update(
            max_absolute_probability_difference=-1.0
        ),
        lambda value: value["validation_scoring_audit"].update(
            warning_records=[{"stage": "fit", "category": "Warning", "message": "x"}]
        ),
    ],
)
def test_loader_rejects_changed_schema_or_invalid_state(mutate):
    from automated_phishing_detection import length_inference

    artifact = _artifact()
    mutate(artifact)

    with pytest.raises(length_inference.LengthInferenceError):
        _load(_content(artifact))


@pytest.mark.parametrize(
    "content", [b"\xff", b"{", b'{"model_name":"length-only","model_name":"x"}']
)
def test_loader_rejects_malformed_encoding_json_and_duplicate_keys(content):
    from automated_phishing_detection import length_inference

    with pytest.raises(length_inference.LengthInferenceError):
        _load(content)


def test_loaded_length_artifact_cannot_be_accepted_as_logistic_l1():
    content = _content()
    with pytest.raises(fixed_cascade.FixedCascadeError, match="model_name"):
        fixed_cascade._load_logistic_l1_artifact_bytes(
            content,
            expected_sha256=sha256(content).hexdigest(),
            expected_contract_sha256="0" * 64,
        )


def test_shared_validator_rejects_unrecognized_model_names():
    with pytest.raises(fixed_cascade.FixedCascadeError, match="model_name"):
        fixed_cascade._validate_artifact(
            _artifact(), expected_contract_sha256="0" * 64, model_name="other"
        )


def _restore_reference():
    artifact = _artifact()
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(artifact["scaler"]["mean"])
    scaler.scale_ = np.asarray(artifact["scaler"]["scale"])
    scaler.var_ = np.asarray(artifact["scaler"]["variance"])
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = artifact["scaler"]["n_samples_seen"]
    classifier = LogisticRegression(
        **{
            key: value
            for key, value in baselines._CLASSIFIER_CONFIG.items()
            if key != "class"
        }
    )
    classifier.coef_ = np.asarray(artifact["classifier"]["coefficients"])
    classifier.intercept_ = np.asarray(artifact["classifier"]["intercept"])
    classifier.classes_ = np.asarray([0, 1])
    classifier.n_features_in_ = 1
    classifier.n_iter_ = np.asarray(artifact["classifier"]["n_iter"])
    return scaler, classifier


def test_scoring_matches_restored_sklearn_without_fit(monkeypatch):
    from automated_phishing_detection import length_inference

    urls = (
        "https://example.test/a",
        "https://example.test/\u00e9",
        "HTTP://other.test/path?q=1#fragment",
    )
    scaler, classifier = _restore_reference()
    matrix = np.asarray([extract_url_features(url) for url in urls])[:, np.asarray([0])]
    expected = classifier.predict_proba(scaler.transform(matrix))[:, 1]

    def forbidden_fit(*args, **kwargs):
        pytest.fail("no-fit scoring attempted to fit or select a threshold")

    monkeypatch.setattr(StandardScaler, "fit", forbidden_fit)
    monkeypatch.setattr(StandardScaler, "fit_transform", forbidden_fit)
    monkeypatch.setattr(StandardScaler, "partial_fit", forbidden_fit)
    monkeypatch.setattr(LogisticRegression, "fit", forbidden_fit)
    monkeypatch.setattr(baselines, "select_validation_threshold", forbidden_fit)
    scores, audit = length_inference.score_length_only_authoritative(
        _load(), iter(urls)
    )

    np.testing.assert_array_equal(scores, expected)
    # Lengths 22, 22 and 35 give logits 0.1, 0.1 and 2.7.
    np.testing.assert_allclose(
        scores, [0.52497918747894, 0.52497918747894, 0.9370266439430035]
    )
    assert scores[0] == scores[1]  # Code points, not UTF-8 bytes or canonicalized text.
    assert audit["platform_identity"] == baselines._platform_identity()
    assert audit["max_absolute_probability_difference"] <= 1e-12


@pytest.mark.parametrize(
    "urls", [[], "https://example.test", b"bytes", [None], ["invalid"], 4]
)
def test_scoring_rejects_invalid_complete_inputs(urls):
    from automated_phishing_detection import length_inference

    with pytest.raises(length_inference.LengthInferenceError):
        length_inference.score_length_only_authoritative(_load(), urls)


def test_entire_input_is_checked_before_any_sklearn_scoring(monkeypatch):
    from automated_phishing_detection import length_inference

    def forbidden_score(*args, **kwargs):
        pytest.fail("scoring began before the complete input was validated")

    monkeypatch.setattr(baselines, "_audited_validation_scores", forbidden_score)
    with pytest.raises(length_inference.LengthInferenceError, match=r"raw_urls\[1\]"):
        length_inference.score_length_only_authoritative(
            _load(), ["https://example.test", None]
        )


def test_scoring_requires_loaded_model_and_bound_software(monkeypatch):
    from automated_phishing_detection import length_inference

    with pytest.raises(length_inference.LengthInferenceError, match="loaded"):
        length_inference.score_length_only_authoritative(
            object(), ["https://example.test"]
        )
    model = _load()
    different_versions = deepcopy(baselines._software_versions())
    different_versions["numpy"] = "other"
    monkeypatch.setattr(baselines, "_software_versions", lambda: different_versions)
    with pytest.raises(
        length_inference.LengthInferenceError, match="software versions"
    ):
        length_inference.score_length_only_authoritative(
            model, ["https://example.test"]
        )


def test_scoring_keeps_requested_batch_and_rejects_nonfinite_output(monkeypatch):
    from automated_phishing_detection import length_inference

    observed_batches = []

    def invalid_scores(name, scaled, classifier, *, policy, environment):
        assert name == "length-only"
        assert scaled.flags.f_contiguous
        assert policy == baselines._SCORING_INTEGRITY_POLICY
        observed_batches.append(scaled.shape)
        return np.asarray([float("nan")] * len(scaled)), {}

    monkeypatch.setattr(baselines, "_audited_validation_scores", invalid_scores)
    with pytest.raises(
        length_inference.LengthInferenceError, match="finite probabilities"
    ):
        length_inference.score_length_only_authoritative(
            _load(), ["https://example.test/a", "https://example.test/b"]
        )
    assert observed_batches == [(2, 1)]
