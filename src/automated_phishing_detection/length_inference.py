"""Hash-bound loading and no-fit scoring of the frozen length-only baseline."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import baselines, fixed_cascade
from .fixed_cascade import OFFICIAL_BASELINE_CONTRACT_SHA256
from .url_features import FEATURE_NAMES, FeatureExtractionError, extract_url_features

OFFICIAL_LENGTH_ONLY_SHA256 = (
    "b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799"
)
_FEATURE_NAMES = ("raw_url_codepoint_length",)
_LOADED_ARTIFACT_MARKER = object()


class LengthInferenceError(ValueError):
    """Raised when a frozen length artifact or scoring input is invalid."""


@dataclass(frozen=True)
class LoadedLengthOnly:
    """Immutable fitted state; loading does not fit or select a threshold."""

    artifact_sha256: str
    contract_sha256: str
    _mean: tuple[float, ...]
    _scale: tuple[float, ...]
    _variance: tuple[float, ...]
    _coefficients: tuple[float, ...]
    _intercept: float
    _threshold_record_json: str
    _software_versions: tuple[tuple[str, str], ...]
    _n_samples_seen: int
    _n_iter: int
    _artifact_bytes: bytes = field(repr=False)
    _loader_marker: object = field(repr=False, compare=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return _FEATURE_NAMES

    @property
    def validation_threshold_record(self) -> dict[str, object]:
        return json.loads(self._threshold_record_json)


def _load_length_only_artifact_bytes(
    content: bytes,
    *,
    expected_sha256: str = OFFICIAL_LENGTH_ONLY_SHA256,
    expected_contract_sha256: str = OFFICIAL_BASELINE_CONTRACT_SHA256,
) -> LoadedLengthOnly:
    """Load exact bytes; explicit hash overrides are for isolated synthetic fixtures."""
    if type(content) is not bytes:
        raise LengthInferenceError("artifact content must be exact bytes")
    try:
        expected_sha256 = fixed_cascade._lowercase_sha256(
            expected_sha256, "expected artifact hash"
        )
        expected_contract_sha256 = fixed_cascade._lowercase_sha256(
            expected_contract_sha256, "expected contract hash"
        )
        observed_sha256 = sha256(content).hexdigest()
        if observed_sha256 != expected_sha256:
            raise LengthInferenceError(
                "artifact SHA-256 mismatch: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
            parse_constant=fixed_cascade._reject_json_constant,
        )
        mean, scale, variance, coefficients, intercept, threshold, _ = (
            fixed_cascade._validate_artifact(
                value,
                expected_contract_sha256=expected_contract_sha256,
                model_name="length-only",
            )
        )
    except UnicodeDecodeError as exc:
        raise LengthInferenceError("artifact is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise LengthInferenceError("artifact is not valid JSON") from exc
    except fixed_cascade.FixedCascadeError as exc:
        raise LengthInferenceError(str(exc)) from exc
    return LoadedLengthOnly(
        artifact_sha256=observed_sha256,
        contract_sha256=expected_contract_sha256,
        _mean=mean,
        _scale=scale,
        _variance=variance,
        _coefficients=coefficients,
        _intercept=intercept,
        _threshold_record_json=json.dumps(
            threshold,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        _software_versions=tuple(sorted(value["software_versions"].items())),
        _n_samples_seen=value["scaler"]["n_samples_seen"],
        _n_iter=value["classifier"]["n_iter"][0],
        _artifact_bytes=content,
        _loader_marker=_LOADED_ARTIFACT_MARKER,
    )


def load_length_only_artifact(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str = OFFICIAL_LENGTH_ONLY_SHA256,
    expected_contract_sha256: str = OFFICIAL_BASELINE_CONTRACT_SHA256,
) -> LoadedLengthOnly:
    """Read one regular, hash-bound length-only artifact without fitting it."""
    try:
        artifact_path = Path(path)
    except TypeError as exc:
        raise LengthInferenceError("artifact path must be path-like") from exc
    try:
        content = fixed_cascade._read_regular_file(artifact_path)
    except fixed_cascade.FixedCascadeError as exc:
        raise LengthInferenceError(str(exc)) from exc
    return _load_length_only_artifact_bytes(
        content,
        expected_sha256=expected_sha256,
        expected_contract_sha256=expected_contract_sha256,
    )


def score_length_only_authoritative(
    model: LoadedLengthOnly, raw_urls: object
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Restore the original sklearn path for the caller's complete scoring batch."""
    if (
        type(model) is not LoadedLengthOnly
        or model._loader_marker is not _LOADED_ARTIFACT_MARKER
    ):
        raise LengthInferenceError("model must be a loaded LoadedLengthOnly artifact")
    if baselines._software_versions() != dict(model._software_versions):
        raise LengthInferenceError(
            "runtime software versions do not match the frozen baseline artifact"
        )
    if isinstance(raw_urls, (str, bytes)) or not isinstance(raw_urls, Iterable):
        raise LengthInferenceError("raw_urls must be a nonempty iterable")
    rows = []
    for index, raw_url in enumerate(raw_urls):
        try:
            rows.append(extract_url_features(raw_url))
        except FeatureExtractionError as exc:
            raise LengthInferenceError(f"raw_urls[{index}] is invalid: {exc}") from None
    if not rows:
        raise LengthInferenceError("raw_urls must be a nonempty iterable")

    try:
        matrix = np.asarray(rows, dtype=np.float64)
        indices = np.asarray(
            [FEATURE_NAMES.index(name) for name in _FEATURE_NAMES], dtype=np.intp
        )
        # Preserve the fit pipeline's advanced column selection and raw URL length.
        matrix = matrix[:, indices]
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.mean_ = np.asarray(model._mean, dtype=np.float64)
        scaler.scale_ = np.asarray(model._scale, dtype=np.float64)
        scaler.var_ = np.asarray(model._variance, dtype=np.float64)
        scaler.n_features_in_ = 1
        scaler.n_samples_seen_ = model._n_samples_seen
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scaled = scaler.transform(matrix)
        if not np.all(np.isfinite(scaled)):
            raise LengthInferenceError("length-only scaling produced nonfinite values")

        classifier = LogisticRegression(
            **{
                key: value
                for key, value in baselines._CLASSIFIER_CONFIG.items()
                if key != "class"
            }
        )
        classifier.coef_ = np.asarray([model._coefficients], dtype=np.float64)
        classifier.intercept_ = np.asarray([model._intercept], dtype=np.float64)
        classifier.classes_ = np.asarray([0, 1], dtype=np.int64)
        classifier.n_features_in_ = 1
        classifier.n_iter_ = np.asarray([model._n_iter], dtype=np.int32)
        scores, audit = baselines._audited_validation_scores(
            "length-only",
            scaled,
            classifier,
            policy=baselines._SCORING_INTEGRITY_POLICY,
            environment=baselines._platform_identity(),
        )
        scores = fixed_cascade._probability_vector(
            scores, "authoritative length-only scores"
        )
    except (ValueError, Warning, FloatingPointError) as exc:
        raise LengthInferenceError(
            f"authoritative length-only scoring failed: {exc}"
        ) from exc
    return tuple(float(value) for value in scores), audit
