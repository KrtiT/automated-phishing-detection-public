"""Portable Logistic-L1 scoring and validation-only cascade calibration."""

from __future__ import annotations

import json
import math
import os
import stat
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import baselines
from .baselines import clopper_pearson_upper, select_validation_threshold
from .url_features import FEATURE_NAMES, FeatureExtractionError, extract_url_features

OFFICIAL_LOGISTIC_L1_SHA256 = (
    "71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a"
)
OFFICIAL_BASELINE_CONTRACT_SHA256 = (
    "05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba"
)
MAXIMUM_FPR_UPPER_95 = 0.01
RECALL_TOLERANCE = 0.02
_RECALL_TOLERANCE_NUMERATOR = 1
_RECALL_TOLERANCE_DENOMINATOR = 50
_LOADED_ARTIFACT_MARKER = object()

_LOWERCASE_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "analysis_stage",
        "contract_id",
        "contract_sha256",
        "model_name",
        "features",
        "classes",
        "scaler",
        "classifier",
        "validation_scoring_audit",
        "validation_threshold",
        "input_hashes",
        "software_versions",
        "access",
    }
)
_SCALER_CONFIG = {
    "class": "StandardScaler",
    "fit_partition": "train",
    "with_mean": True,
    "with_std": True,
}
_CLASSIFIER_CONFIG = {
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
_THRESHOLD_FIELDS = frozenset(
    {
        "status",
        "threshold",
        "candidate_count",
        "counts",
        "recall",
        "observed_fpr",
        "fpr_upper_95",
    }
)
_COUNT_FIELDS = frozenset(
    {
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "positive",
        "negative",
    }
)


class FixedCascadeError(ValueError):
    """Raised when a frozen artifact or cascade input is invalid."""


def _expect_fields(value: object, expected: frozenset[str], field: str) -> dict:
    if type(value) is not dict or frozenset(value) != expected:
        raise FixedCascadeError(f"{field} fields do not match the frozen schema")
    return value


def _matches_exactly(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _matches_exactly(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _matches_exactly(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def _lowercase_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise FixedCascadeError(f"{field} must be a lowercase SHA-256")
    return value


def _finite_number(value: object, field: str) -> float:
    if type(value) not in {int, float}:
        raise FixedCascadeError(f"{field} must be a finite number")
    try:
        finite = math.isfinite(value)
        converted = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise FixedCascadeError(f"{field} must be a finite number") from exc
    if not finite:
        raise FixedCascadeError(f"{field} must be a finite number")
    return converted


def _exact_integer(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise FixedCascadeError(f"{field} must be an integer >= {minimum}")
    return value


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FixedCascadeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise FixedCascadeError(f"artifact contains nonfinite JSON number: {value}")


def _read_regular_file(path: Path) -> bytes:
    try:
        path_metadata = path.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise FixedCascadeError(
            "artifact does not exist or cannot be inspected"
        ) from exc
    if not stat.S_ISREG(path_metadata.st_mode):
        raise FixedCascadeError("artifact must be a regular file, not an alias")

    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FixedCascadeError("artifact could not be opened safely") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            before.st_dev,
            before.st_ino,
        ) != (path_metadata.st_dev, path_metadata.st_ino):
            raise FixedCascadeError("artifact identity changed before reading")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    before_state = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_state = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_state != after_state or len(content) != before.st_size:
        raise FixedCascadeError("artifact changed while it was being read")
    return content


def _numeric_vector(value: object, width: int, field: str) -> tuple[float, ...]:
    if type(value) is not list or len(value) != width:
        raise FixedCascadeError(f"{field} must contain exactly {width} numbers")
    result = tuple(_finite_number(item, field) for item in value)
    return result


def _validate_threshold_record(value: object, field: str) -> dict[str, object]:
    record = _expect_fields(value, _THRESHOLD_FIELDS, field)
    status = record["status"]
    if type(status) is not str or status not in {"selected", "target_not_met"}:
        raise FixedCascadeError(f"{field}.status is invalid")
    _exact_integer(record["candidate_count"], f"{field}.candidate_count", minimum=1)
    if status == "target_not_met":
        for name in (
            "threshold",
            "counts",
            "recall",
            "observed_fpr",
            "fpr_upper_95",
        ):
            if record[name] is not None:
                raise FixedCascadeError(f"{field}.{name} must be null")
        return record

    threshold = _finite_number(record["threshold"], f"{field}.threshold")
    if not 0.0 <= threshold <= float(np.nextafter(1.0, np.inf)):
        raise FixedCascadeError(f"{field}.threshold is outside the valid range")
    counts = _expect_fields(record["counts"], _COUNT_FIELDS, f"{field}.counts")
    for name in _COUNT_FIELDS:
        _exact_integer(counts[name], f"{field}.counts.{name}")
    if counts["positive"] < 1 or counts["negative"] < 1:
        raise FixedCascadeError(f"{field}.counts must contain both classes")
    if counts["true_positive"] + counts["false_negative"] != counts["positive"]:
        raise FixedCascadeError(f"{field}.positive counts are inconsistent")
    if counts["false_positive"] + counts["true_negative"] != counts["negative"]:
        raise FixedCascadeError(f"{field}.negative counts are inconsistent")

    recall = _finite_number(record["recall"], f"{field}.recall")
    observed_fpr = _finite_number(record["observed_fpr"], f"{field}.observed_fpr")
    upper = _finite_number(record["fpr_upper_95"], f"{field}.fpr_upper_95")
    expected_recall = counts["true_positive"] / counts["positive"]
    expected_fpr = counts["false_positive"] / counts["negative"]
    expected_upper = clopper_pearson_upper(counts["false_positive"], counts["negative"])
    if not math.isclose(recall, expected_recall, rel_tol=1e-15, abs_tol=1e-15):
        raise FixedCascadeError(f"{field}.recall is inconsistent with counts")
    if not math.isclose(observed_fpr, expected_fpr, rel_tol=1e-15, abs_tol=1e-15):
        raise FixedCascadeError(f"{field}.observed_fpr is inconsistent with counts")
    if not math.isclose(upper, expected_upper, rel_tol=1e-15, abs_tol=1e-15):
        raise FixedCascadeError(f"{field}.fpr_upper_95 is inconsistent with counts")
    if upper > MAXIMUM_FPR_UPPER_95:
        raise FixedCascadeError(f"{field} does not satisfy the frozen FPR gate")
    return record


def _validate_stage1_threshold_binding(
    model: PortableLogisticL1,
    scores: np.ndarray,
    labels: np.ndarray,
) -> dict[str, object]:
    record = _validate_threshold_record(
        model.validation_threshold_record, "stage1_model.validation_threshold"
    )
    recomputed = select_validation_threshold(scores, labels)
    comparable_fields = _THRESHOLD_FIELDS - {"threshold"}
    if any(
        not _matches_exactly(record[field], recomputed[field])
        for field in comparable_fields
    ):
        raise FixedCascadeError(
            "stage-one artifact threshold does not match the supplied scores"
        )
    if record["status"] == "target_not_met":
        return record
    threshold_difference = abs(record["threshold"] - recomputed["threshold"])
    if threshold_difference > model.max_absolute_probability_difference:
        raise FixedCascadeError(
            "stage-one threshold difference exceeds the artifact scoring audit"
        )
    observed_counts = _confusion_counts(scores >= record["threshold"], labels)
    if not _matches_exactly(record["counts"], observed_counts):
        raise FixedCascadeError(
            "stage-one artifact threshold counts do not match the supplied scores"
        )
    return record


def _validate_transformer_threshold_binding(
    record: object, scores: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    record = _validate_threshold_record(record, "transformer_threshold_record")
    recomputed = select_validation_threshold(scores, labels)
    if not _matches_exactly(record, recomputed):
        raise FixedCascadeError(
            "transformer threshold record does not exactly match the supplied scores"
        )
    return record


def _validate_artifact(
    value: object, *, expected_contract_sha256: str
) -> tuple[
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[float, ...],
    float,
    dict[str, object],
    float,
]:
    artifact = _expect_fields(value, _TOP_LEVEL_FIELDS, "artifact")
    expected_identity = {
        "schema_version": 2,
        "artifact_type": "rq1-baseline-model",
        "analysis_stage": "development_validation_only",
        "contract_id": "rq1-baselines-v2",
        "contract_sha256": expected_contract_sha256,
        "model_name": "Logistic-L1",
        "features": list(FEATURE_NAMES),
        "classes": [0, 1],
    }
    for field, expected in expected_identity.items():
        if not _matches_exactly(artifact[field], expected):
            raise FixedCascadeError(f"artifact {field} does not match Logistic-L1")

    width = len(FEATURE_NAMES)
    scaler = _expect_fields(
        artifact["scaler"],
        frozenset({"config", "mean", "scale", "variance", "n_samples_seen"}),
        "artifact.scaler",
    )
    if not _matches_exactly(scaler["config"], _SCALER_CONFIG):
        raise FixedCascadeError("artifact scaler configuration has changed")
    mean = _numeric_vector(scaler["mean"], width, "artifact.scaler.mean")
    scale = _numeric_vector(scaler["scale"], width, "artifact.scaler.scale")
    variance = _numeric_vector(scaler["variance"], width, "artifact.scaler.variance")
    if any(item <= 0.0 for item in scale) or any(item < 0.0 for item in variance):
        raise FixedCascadeError("artifact scaler state is invalid")
    _exact_integer(
        scaler["n_samples_seen"], "artifact.scaler.n_samples_seen", minimum=1
    )

    classifier = _expect_fields(
        artifact["classifier"],
        frozenset({"config", "coefficients", "intercept", "n_iter"}),
        "artifact.classifier",
    )
    if not _matches_exactly(classifier["config"], _CLASSIFIER_CONFIG):
        raise FixedCascadeError("artifact classifier configuration has changed")
    coefficients = classifier["coefficients"]
    if type(coefficients) is not list or len(coefficients) != 1:
        raise FixedCascadeError("artifact coefficients must have shape (1, 25)")
    coefficient_row = _numeric_vector(
        coefficients[0], width, "artifact.classifier.coefficients"
    )
    intercept = _numeric_vector(
        classifier["intercept"], 1, "artifact.classifier.intercept"
    )[0]
    iterations = classifier["n_iter"]
    if (
        type(iterations) is not list
        or len(iterations) != 1
        or type(iterations[0]) is not int
        or not 0 < iterations[0] < _CLASSIFIER_CONFIG["max_iter"]
    ):
        raise FixedCascadeError("artifact classifier iteration count is invalid")

    audit = _expect_fields(
        artifact["validation_scoring_audit"],
        frozenset(
            {
                "platform_identity",
                "warning_records",
                "max_absolute_decision_difference",
                "max_absolute_probability_difference",
            }
        ),
        "artifact.validation_scoring_audit",
    )
    platform_identity = _expect_fields(
        audit["platform_identity"],
        frozenset({"sys_platform", "platform_machine", "numpy_blas_name"}),
        "artifact.validation_scoring_audit.platform_identity",
    )
    if any(type(item) is not str or not item for item in platform_identity.values()):
        raise FixedCascadeError("artifact platform identity is invalid")
    warnings_value = audit["warning_records"]
    if type(warnings_value) is not list:
        raise FixedCascadeError("artifact warning records must be a list")
    allowed_messages = {
        "divide by zero encountered in matmul",
        "overflow encountered in matmul",
        "invalid value encountered in matmul",
    }
    for index, warning in enumerate(warnings_value):
        warning = _expect_fields(
            warning,
            frozenset({"stage", "category", "message"}),
            f"artifact warning record {index}",
        )
        if (
            type(warning["stage"]) is not str
            or warning["stage"] not in {"decision_function", "predict_proba"}
            or type(warning["category"]) is not str
            or warning["category"] != "RuntimeWarning"
            or type(warning["message"]) is not str
            or warning["message"] not in allowed_messages
        ):
            raise FixedCascadeError("artifact warning record is not allowed")
    decision_difference = _finite_number(
        audit["max_absolute_decision_difference"],
        "artifact.max_absolute_decision_difference",
    )
    probability_difference = _finite_number(
        audit["max_absolute_probability_difference"],
        "artifact.max_absolute_probability_difference",
    )
    if (
        decision_difference < 0.0
        or probability_difference < 0.0
        or probability_difference > 1.0
    ):
        raise FixedCascadeError("artifact scoring differences are outside valid ranges")

    threshold_record = _validate_threshold_record(
        artifact["validation_threshold"], "artifact.validation_threshold"
    )
    input_hashes = _expect_fields(
        artifact["input_hashes"],
        frozenset({"train", "validation", "preparation_summary", "contract"}),
        "artifact.input_hashes",
    )
    for field, digest in input_hashes.items():
        _lowercase_sha256(digest, f"artifact.input_hashes.{field}")
    if input_hashes["contract"] != expected_contract_sha256:
        raise FixedCascadeError("artifact input contract hash does not match")

    versions = _expect_fields(
        artifact["software_versions"],
        frozenset({"numpy", "scikit-learn", "scipy"}),
        "artifact.software_versions",
    )
    if any(type(item) is not str or not item for item in versions.values()):
        raise FixedCascadeError("artifact software versions are invalid")
    if not _matches_exactly(
        artifact["access"],
        {"group_test_accessed": False, "phishvn_accessed": False},
    ):
        raise FixedCascadeError("artifact access record is invalid")

    return (
        mean,
        scale,
        variance,
        coefficient_row,
        intercept,
        threshold_record,
        probability_difference,
    )


@dataclass(frozen=True)
class PortableLogisticL1:
    """Immutable state required to score the frozen structural baseline."""

    mean: tuple[float, ...]
    scale: tuple[float, ...]
    variance: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    artifact_sha256: str
    contract_sha256: str
    max_absolute_probability_difference: float
    _threshold_record_json: str
    _software_versions: tuple[tuple[str, str], ...]
    _n_samples_seen: int
    _n_iter: int
    _loader_marker: object = dataclass_field(repr=False, compare=False)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    @property
    def validation_threshold_record(self) -> dict[str, object]:
        return json.loads(self._threshold_record_json)

    def score_urls(self, raw_urls: object) -> tuple[float, ...]:
        """Score raw URLs from serialized parameters without fitting an estimator."""
        if isinstance(raw_urls, (str, bytes)) or not isinstance(raw_urls, Iterable):
            raise FixedCascadeError("raw_urls must be a nonempty iterable")
        rows = []
        for index, raw_url in enumerate(raw_urls):
            try:
                rows.append(extract_url_features(raw_url))
            except FeatureExtractionError as exc:
                raise FixedCascadeError(
                    f"raw_urls[{index}] is invalid: {exc}"
                ) from None
        if not rows:
            raise FixedCascadeError("raw_urls must be a nonempty iterable")

        matrix = np.asarray(rows, dtype=np.float64)
        try:
            with warnings.catch_warnings(), np.errstate(all="raise"):
                warnings.simplefilter("error")
                scaled = (matrix - np.asarray(self.mean)) / np.asarray(self.scale)
                decisions = (
                    np.einsum(
                        "ij,j->i",
                        scaled,
                        np.asarray(self.coefficients),
                        optimize=False,
                    )
                    + self.intercept
                )
                probabilities = expit(decisions)
        except (FloatingPointError, ValueError, Warning) as exc:
            raise FixedCascadeError(
                f"portable Logistic-L1 scoring failed: {exc}"
            ) from exc
        if probabilities.shape != (len(rows),) or not np.all(
            np.isfinite(probabilities)
        ):
            raise FixedCascadeError("portable Logistic-L1 scores are invalid")
        return tuple(float(value) for value in probabilities)


def score_logistic_l1_authoritative(
    model: PortableLogisticL1, raw_urls: object
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Reconstruct the hash-loaded baseline's sklearn scoring path without fitting."""
    if (
        type(model) is not PortableLogisticL1
        or model._loader_marker is not _LOADED_ARTIFACT_MARKER
    ):
        raise FixedCascadeError("model must be a loaded PortableLogisticL1 artifact")
    if baselines._software_versions() != dict(model._software_versions):
        raise FixedCascadeError(
            "runtime software versions do not match the frozen baseline artifact"
        )
    if isinstance(raw_urls, (str, bytes)) or not isinstance(raw_urls, Iterable):
        raise FixedCascadeError("raw_urls must be a nonempty iterable")
    rows = []
    for index, raw_url in enumerate(raw_urls):
        try:
            rows.append(extract_url_features(raw_url))
        except FeatureExtractionError as exc:
            raise FixedCascadeError(f"raw_urls[{index}] is invalid: {exc}") from None
    if not rows:
        raise FixedCascadeError("raw_urls must be a nonempty iterable")

    try:
        matrix = np.asarray(rows, dtype=np.float64)
        indices = np.asarray(
            [FEATURE_NAMES.index(name) for name in model.feature_names], dtype=np.intp
        )
        # Match baseline advanced column selection: F-order affects exact score ties.
        matrix = matrix[:, indices]
        scaler = StandardScaler(with_mean=True, with_std=True)
        scaler.mean_ = np.asarray(model.mean, dtype=np.float64)
        scaler.scale_ = np.asarray(model.scale, dtype=np.float64)
        scaler.var_ = np.asarray(model.variance, dtype=np.float64)
        scaler.n_features_in_ = len(model.feature_names)
        scaler.n_samples_seen_ = model._n_samples_seen
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scaled = scaler.transform(matrix)
        if not np.all(np.isfinite(scaled)):
            raise FixedCascadeError("Logistic-L1 scaling produced nonfinite values")

        classifier = LogisticRegression(
            **{
                key: value
                for key, value in _CLASSIFIER_CONFIG.items()
                if key != "class"
            }
        )
        classifier.coef_ = np.asarray([model.coefficients], dtype=np.float64)
        classifier.intercept_ = np.asarray([model.intercept], dtype=np.float64)
        classifier.classes_ = np.asarray([0, 1], dtype=np.int64)
        classifier.n_features_in_ = len(model.feature_names)
        classifier.n_iter_ = np.asarray([model._n_iter], dtype=np.int32)
        scores, audit = baselines._audited_validation_scores(
            "Logistic-L1",
            scaled,
            classifier,
            policy=baselines._SCORING_INTEGRITY_POLICY,
            environment=baselines._platform_identity(),
        )
        scores = _probability_vector(scores, "authoritative Logistic-L1 scores")
    except (ValueError, Warning, FloatingPointError) as exc:
        raise FixedCascadeError(
            f"authoritative Logistic-L1 scoring failed: {exc}"
        ) from exc
    return tuple(float(value) for value in scores), audit


def _load_logistic_l1_artifact_bytes(
    content: bytes,
    *,
    expected_sha256: str = OFFICIAL_LOGISTIC_L1_SHA256,
    expected_contract_sha256: str = OFFICIAL_BASELINE_CONTRACT_SHA256,
) -> PortableLogisticL1:
    """Load exact artifact bytes that are already isolated from their source path."""
    if type(content) is not bytes:
        raise FixedCascadeError("artifact content must be exact bytes")
    expected_sha256 = _lowercase_sha256(expected_sha256, "expected artifact hash")
    expected_contract_sha256 = _lowercase_sha256(
        expected_contract_sha256, "expected contract hash"
    )
    observed_sha256 = sha256(content).hexdigest()
    if observed_sha256 != expected_sha256:
        raise FixedCascadeError(
            "artifact SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except UnicodeDecodeError as exc:
        raise FixedCascadeError("artifact is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise FixedCascadeError("artifact is not valid JSON") from exc
    (
        mean,
        scale,
        variance,
        coefficients,
        intercept,
        threshold,
        probability_difference,
    ) = _validate_artifact(value, expected_contract_sha256=expected_contract_sha256)
    threshold_json = json.dumps(
        threshold,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return PortableLogisticL1(
        mean=mean,
        scale=scale,
        variance=variance,
        coefficients=coefficients,
        intercept=intercept,
        artifact_sha256=observed_sha256,
        contract_sha256=expected_contract_sha256,
        max_absolute_probability_difference=probability_difference,
        _threshold_record_json=threshold_json,
        _software_versions=tuple(sorted(value["software_versions"].items())),
        _n_samples_seen=value["scaler"]["n_samples_seen"],
        _n_iter=value["classifier"]["n_iter"][0],
        _loader_marker=_LOADED_ARTIFACT_MARKER,
    )


def load_logistic_l1_artifact(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str = OFFICIAL_LOGISTIC_L1_SHA256,
    expected_contract_sha256: str = OFFICIAL_BASELINE_CONTRACT_SHA256,
) -> PortableLogisticL1:
    """Load one exact hash-bound Logistic-L1 artifact without fitting it."""
    try:
        artifact_path = Path(path)
    except TypeError as exc:
        raise FixedCascadeError("artifact path must be path-like") from exc
    return _load_logistic_l1_artifact_bytes(
        _read_regular_file(artifact_path),
        expected_sha256=expected_sha256,
        expected_contract_sha256=expected_contract_sha256,
    )


def _probability_vector(value: object, field: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise FixedCascadeError(f"{field} must be a nonempty numeric vector") from exc
    if (
        array.ndim != 1
        or array.size == 0
        or not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise FixedCascadeError(f"{field} must be a nonempty numeric vector")
    array = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise FixedCascadeError(f"{field} must contain finite probabilities")
    return array


def _binary_labels(value: object, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        labels = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise FixedCascadeError(
            "labels must be an equal-length binary integer vector"
        ) from exc
    if (
        labels.ndim != 1
        or labels.shape != expected_shape
        or not np.issubdtype(labels.dtype, np.integer)
        or np.issubdtype(labels.dtype, np.bool_)
        or set(labels.tolist()) != {0, 1}
    ):
        raise FixedCascadeError("labels must be an equal-length binary integer vector")
    return labels.astype(np.int8, copy=False)


def _threshold(value: object, field: str) -> float:
    threshold = _finite_number(value, field)
    if not 0.0 <= threshold <= float(np.nextafter(1.0, np.inf)):
        raise FixedCascadeError(f"{field} is outside the valid range")
    return threshold


@dataclass(frozen=True)
class CascadeScores:
    """Ephemeral per-row outputs from one fixed cascade configuration."""

    probabilities: tuple[float, ...]
    decisions: tuple[int, ...]
    transformer_invoked: tuple[bool, ...]


def score_fixed_cascade(
    stage1_probabilities: object,
    transformer_probabilities: object,
    *,
    stage1_threshold: object,
    transformer_threshold: object,
    half_width: object,
) -> CascadeScores:
    """Apply the inclusive fixed-band routing rule to two score vectors."""
    stage1 = _probability_vector(stage1_probabilities, "stage1_probabilities")
    transformer = _probability_vector(
        transformer_probabilities, "transformer_probabilities"
    )
    if transformer.shape != stage1.shape:
        raise FixedCascadeError(
            "stage-one and transformer vectors must have equal shape"
        )
    stage1_threshold = _threshold(stage1_threshold, "stage1_threshold")
    transformer_threshold = _threshold(transformer_threshold, "transformer_threshold")
    half_width = _finite_number(half_width, "half_width")
    if half_width < 0.0:
        raise FixedCascadeError("half_width must be nonnegative")

    transformer_invoked = np.abs(stage1 - stage1_threshold) <= half_width
    probabilities = np.where(transformer_invoked, transformer, stage1)
    stage1_decisions = stage1 >= stage1_threshold
    transformer_decisions = transformer >= transformer_threshold
    decisions = np.where(
        transformer_invoked, transformer_decisions, stage1_decisions
    ).astype(np.int8)
    return CascadeScores(
        probabilities=tuple(float(value) for value in probabilities),
        decisions=tuple(int(value) for value in decisions),
        transformer_invoked=tuple(bool(value) for value in transformer_invoked),
    )


def cascade_candidate_half_widths(
    stage1_probabilities: object, stage1_threshold: object
) -> tuple[float, ...]:
    """Return the exact sorted unique validation-distance candidates."""
    stage1 = _probability_vector(stage1_probabilities, "stage1_probabilities")
    stage1_threshold = _threshold(stage1_threshold, "stage1_threshold")
    return tuple(float(value) for value in np.unique(np.abs(stage1 - stage1_threshold)))


def _confusion_counts(decisions: Sequence[int], labels: np.ndarray) -> dict[str, int]:
    decision_array = np.asarray(decisions, dtype=np.int8)
    true_positive = int(np.count_nonzero((decision_array == 1) & (labels == 1)))
    false_positive = int(np.count_nonzero((decision_array == 1) & (labels == 0)))
    positive = int(np.count_nonzero(labels == 1))
    negative = int(labels.size - positive)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": negative - false_positive,
        "false_negative": positive - true_positive,
        "positive": positive,
        "negative": negative,
    }


def _choose_candidate(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    if not candidates:
        raise FixedCascadeError("at least one feasible candidate is required")
    try:
        return min(
            candidates,
            key=lambda item: (
                _exact_integer(
                    item["transformer_invocations"], "transformer_invocations"
                ),
                _finite_number(item["half_width"], "half_width"),
            ),
        )
    except KeyError as exc:
        raise FixedCascadeError("candidate fields are incomplete") from exc


def _target_not_met_record(
    *,
    reason: str,
    stage1_status: str,
    transformer_status: str,
    candidate_count: int,
    minimum_recall: float | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "target_not_met",
        "accepted_cascade": False,
        "reason": reason,
        "threshold_statuses": {
            "stage1": stage1_status,
            "transformer": transformer_status,
        },
        "candidate_count": candidate_count,
        "half_width": None,
        "transformer_invocations": None,
        "transformer_invocation_rate": None,
        "counts": None,
        "recall": None,
        "observed_fpr": None,
        "fpr_upper_95": None,
        "minimum_recall": minimum_recall,
        "maximum_fpr_upper_95": MAXIMUM_FPR_UPPER_95,
    }


def calibrate_fixed_cascade(
    stage1_model: PortableLogisticL1,
    stage1_probabilities: object,
    transformer_probabilities: object,
    labels: object,
    *,
    transformer_threshold_record: object,
) -> dict[str, object]:
    """Choose a cascade width using validation probabilities and labels only."""
    if (
        type(stage1_model) is not PortableLogisticL1
        or stage1_model._loader_marker is not _LOADED_ARTIFACT_MARKER
    ):
        raise FixedCascadeError(
            "stage1_model must be a loaded PortableLogisticL1 artifact"
        )
    stage1 = _probability_vector(stage1_probabilities, "stage1_probabilities")
    transformer = _probability_vector(
        transformer_probabilities, "transformer_probabilities"
    )
    if transformer.shape != stage1.shape:
        raise FixedCascadeError(
            "stage-one and transformer vectors must have equal shape"
        )
    labels = _binary_labels(labels, expected_shape=stage1.shape)
    stage1_record = _validate_stage1_threshold_binding(stage1_model, stage1, labels)
    transformer_record = _validate_transformer_threshold_binding(
        transformer_threshold_record, transformer, labels
    )
    stage1_status = stage1_record["status"]
    transformer_status = transformer_record["status"]
    if stage1_status == "target_not_met" or transformer_status == "target_not_met":
        return _target_not_met_record(
            reason="threshold_target_not_met",
            stage1_status=stage1_status,
            transformer_status=transformer_status,
            candidate_count=0,
            minimum_recall=None,
        )

    stage1_threshold = stage1_record["threshold"]
    transformer_threshold = transformer_record["threshold"]
    stage1_decisions = stage1 >= stage1_threshold
    transformer_decisions = transformer >= transformer_threshold
    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = int(labels.size - positive_count)
    transformer_true_positive = int(
        np.count_nonzero(transformer_decisions & (labels == 1))
    )
    transformer_recall = transformer_true_positive / positive_count
    minimum_recall = transformer_recall - RECALL_TOLERANCE
    distances = np.abs(stage1 - stage1_threshold)
    order = np.argsort(distances, kind="stable")
    ordered_distances = distances[order]
    ordered_labels = labels[order]
    ordered_stage1_decisions = stage1_decisions[order]
    ordered_transformer_decisions = transformer_decisions[order]
    half_widths = tuple(float(value) for value in np.unique(ordered_distances))
    current_true_positive = int(np.count_nonzero(stage1_decisions & (labels == 1)))
    current_false_positive = int(np.count_nonzero(stage1_decisions & (labels == 0)))
    feasible = []
    start = 0
    for half_width in half_widths:
        end = start
        while end < labels.size and ordered_distances[end] == half_width:
            end += 1
        group_labels = ordered_labels[start:end]
        decision_changes = ordered_transformer_decisions[start:end].astype(
            np.int8
        ) - ordered_stage1_decisions[start:end].astype(np.int8)
        current_true_positive += int(
            decision_changes[group_labels == 1].sum(dtype=np.int64)
        )
        current_false_positive += int(
            decision_changes[group_labels == 0].sum(dtype=np.int64)
        )
        counts = {
            "true_positive": current_true_positive,
            "false_positive": current_false_positive,
            "true_negative": negative_count - current_false_positive,
            "false_negative": positive_count - current_true_positive,
            "positive": positive_count,
            "negative": negative_count,
        }
        recall = counts["true_positive"] / counts["positive"]
        observed_fpr = counts["false_positive"] / counts["negative"]
        upper = clopper_pearson_upper(counts["false_positive"], counts["negative"])
        candidate = {
            "half_width": half_width,
            "transformer_invocations": end,
            "counts": counts,
            "recall": recall,
            "observed_fpr": observed_fpr,
            "fpr_upper_95": upper,
        }
        recall_gate_met = (
            _RECALL_TOLERANCE_DENOMINATOR
            * (current_true_positive - transformer_true_positive)
            >= -_RECALL_TOLERANCE_NUMERATOR * positive_count
        )
        if recall_gate_met and upper <= MAXIMUM_FPR_UPPER_95:
            feasible.append(candidate)
        start = end

    if not feasible:
        return _target_not_met_record(
            reason="cascade_constraints_not_met",
            stage1_status=stage1_status,
            transformer_status=transformer_status,
            candidate_count=len(half_widths),
            minimum_recall=minimum_recall,
        )

    selected = _choose_candidate(feasible)
    invocation_count = selected["transformer_invocations"]
    return {
        "schema_version": 1,
        "status": "selected",
        "accepted_cascade": True,
        "reason": "constraints_met",
        "threshold_statuses": {
            "stage1": stage1_status,
            "transformer": transformer_status,
        },
        "candidate_count": len(half_widths),
        "half_width": selected["half_width"],
        "transformer_invocations": invocation_count,
        "transformer_invocation_rate": invocation_count / labels.size,
        "counts": selected["counts"],
        "recall": selected["recall"],
        "observed_fpr": selected["observed_fpr"],
        "fpr_upper_95": selected["fpr_upper_95"],
        "minimum_recall": minimum_recall,
        "maximum_fpr_upper_95": MAXIMUM_FPR_UPPER_95,
    }
