"""One fixed structural follow-up candidate, with no test-time model selection."""

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import baselines, fixed_cascade
from .transport_neutral_features import (
    FEATURE_NAMES,
    REPRESENTATION_ID,
    extract_transport_neutral_features,
)

ARTIFACT_TYPE = "transport-neutral-followup-model-v1"


def _features(urls):
    if isinstance(urls, (str, bytes)):
        raise ValueError("URLs must be a nonempty sequence")
    matrix = np.asfortranarray(
        [extract_transport_neutral_features(url) for url in urls], dtype=np.float64
    )
    if matrix.ndim != 2 or matrix.shape[1] != 24 or not len(matrix):
        raise ValueError("feature matrix must have nonempty 24-feature rows")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("feature matrix must be finite")
    return matrix


def _labels(values, count):
    labels = np.asarray(values)
    if (
        labels.shape != (count,)
        or not np.issubdtype(labels.dtype, np.integer)
        or set(labels.tolist()) != {0, 1}
    ):
        raise ValueError("labels must contain both exact integer classes 0 and 1")
    return labels


def _classifier():
    return LogisticRegression(
        penalty="l1",
        solver="saga",
        C=1.0,
        class_weight="balanced",
        fit_intercept=True,
        max_iter=5000,
        tol=1e-4,
        random_state=42,
    )


def _audit(classifier, scaled):
    return baselines._audited_validation_scores(
        REPRESENTATION_ID,
        scaled,
        classifier,
        policy=baselines._SCORING_INTEGRITY_POLICY,
        environment=baselines._platform_identity(),
    )


def fit_model(train_urls, train_labels, validation_urls, validation_labels):
    """Fit exactly once; input admission and file provenance belong to the caller."""
    train = _features(train_urls)
    validation = _features(validation_urls)
    train_labels = _labels(train_labels, len(train))
    validation_labels = _labels(validation_labels, len(validation))
    scaler = StandardScaler(with_mean=True, with_std=True)
    classifier = _classifier()
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        scaled_train = scaler.fit_transform(train)
        scaled_validation = scaler.transform(validation)
        classifier.fit(scaled_train, train_labels)
    if classifier.n_iter_.shape != (1,) or not 0 < classifier.n_iter_[0] < 5000:
        raise ValueError("candidate did not converge below max_iter")
    arrays = (
        scaler.mean_,
        scaler.scale_,
        scaler.var_,
        classifier.coef_,
        classifier.intercept_,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("fitted state contains nonfinite values")
    scores, audit = _audit(classifier, scaled_validation)
    threshold = baselines.select_validation_threshold(scores, validation_labels)
    return {
        "artifact_type": ARTIFACT_TYPE,
        "representation_id": REPRESENTATION_ID,
        "features": list(FEATURE_NAMES),
        "classes": classifier.classes_.tolist(),
        "classifier_config": classifier.get_params(),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "variance": scaler.var_.tolist(),
            "n_samples_seen": int(scaler.n_samples_seen_),
        },
        "coefficients": classifier.coef_[0].tolist(),
        "intercept": float(classifier.intercept_[0]),
        "n_iter": int(classifier.n_iter_[0]),
        "validation_threshold": threshold,
        "validation_scoring_audit": audit,
        "software_versions": baselines._software_versions(),
    }


def score_urls(artifact, urls):
    """Reconstruct and audit the fixed sklearn procedure without fitting."""
    if (
        artifact.get("artifact_type") != ARTIFACT_TYPE
        or artifact.get("representation_id") != REPRESENTATION_ID
        or artifact.get("features") != list(FEATURE_NAMES)
        or artifact.get("software_versions") != baselines._software_versions()
        or artifact.get("classifier_config") != _classifier().get_params()
        or artifact.get("classes") != [0, 1]
    ):
        raise ValueError("incompatible follow-up model identity or runtime")
    vector = fixed_cascade._numeric_vector
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.mean_ = np.asarray(vector(artifact["scaler"]["mean"], 24, "mean"))
    scaler.scale_ = np.asarray(vector(artifact["scaler"]["scale"], 24, "scale"))
    scaler.var_ = np.asarray(vector(artifact["scaler"]["variance"], 24, "variance"))
    if np.any(scaler.scale_ <= 0) or np.any(scaler.var_ < 0):
        raise ValueError("invalid scaler state")
    scaler.n_features_in_ = 24
    scaler.n_samples_seen_ = fixed_cascade._exact_integer(
        artifact["scaler"]["n_samples_seen"], "training row count", minimum=1
    )
    classifier = _classifier()
    classifier.coef_ = np.asarray(
        [vector(artifact["coefficients"], 24, "coefficients")]
    )
    classifier.intercept_ = np.asarray(
        [fixed_cascade._finite_number(artifact["intercept"], "intercept")]
    )
    classifier.classes_ = np.asarray([0, 1], dtype=np.int64)
    classifier.n_features_in_ = 24
    iterations = fixed_cascade._exact_integer(
        artifact["n_iter"], "iterations", minimum=1
    )
    if iterations >= 5000:
        raise ValueError("invalid convergence count")
    classifier.n_iter_ = np.asarray([iterations], dtype=np.int32)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        scaled = scaler.transform(_features(urls))
    if not np.all(np.isfinite(scaled)):
        raise ValueError("scaler produced nonfinite features")
    scores, audit = _audit(classifier, scaled)
    return tuple(float(score) for score in scores), audit
