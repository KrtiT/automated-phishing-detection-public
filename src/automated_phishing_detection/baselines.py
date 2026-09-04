"""Training-only RQ1 baselines with validation-only threshold selection."""

import ctypes
import errno
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import warnings
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO

import numpy as np
import scipy
import sklearn
from scipy.special import betaincinv
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .url_features import FEATURE_NAMES, FeatureExtractionError, extract_url_features

CONTRACT_ID = "rq1-baselines-v1"
ANALYSIS_STAGE = "development_validation_only"
_OFFICIAL_TRAIN_SHA256 = (
    "575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0"
)
_OFFICIAL_VALIDATION_SHA256 = (
    "970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a"
)
_OFFICIAL_PREPARATION_SUMMARY_SHA256 = (
    "1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e"
)
_OFFICIAL_CONTRACT_SHA256 = (
    "594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4"
)
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_RECORD_FIELDS = frozenset(
    {
        "record_id",
        "raw_url",
        "canonical_url_sha256",
        "registrable_domain",
        "is_phishing",
        "split",
    }
)
_MODEL_FEATURES = {
    "length-only": ("raw_url_codepoint_length",),
    "Logistic-L1": FEATURE_NAMES,
}
_MODEL_FILENAMES = {
    "length-only": "length-only.json",
    "Logistic-L1": "logistic-l1.json",
}
_CLASSIFIER_CONFIG = {
    "class": "LogisticRegression",
    "penalty": "l1",
    "solver": "liblinear",
    "C": 1.0,
    "class_weight": "balanced",
    "fit_intercept": True,
    "intercept_scaling": 1.0,
    "max_iter": 5000,
    "tol": 1e-8,
    "random_state": 42,
}
_SCALER_CONFIG = {
    "class": "StandardScaler",
    "fit_partition": "train",
    "with_mean": True,
    "with_std": True,
}


class BaselineError(ValueError):
    """Raised when fitting cannot satisfy the frozen baseline contract."""


@dataclass(frozen=True)
class _InputHashPolicy:
    train_sha256: str
    validation_sha256: str
    preparation_summary_sha256: str
    contract_sha256: str


_OFFICIAL_INPUT_HASH_POLICY = _InputHashPolicy(
    train_sha256=_OFFICIAL_TRAIN_SHA256,
    validation_sha256=_OFFICIAL_VALIDATION_SHA256,
    preparation_summary_sha256=_OFFICIAL_PREPARATION_SUMMARY_SHA256,
    contract_sha256=_OFFICIAL_CONTRACT_SHA256,
)


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise BaselineError(f"{field} must be an exact integer >= {minimum}")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if type(value) is not str or not value:
        raise BaselineError(f"{field} must be a nonempty string")
    return value


def _lowercase_sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise BaselineError(f"{field} must be a lowercase SHA-256")
    return value


def _expect_keys(value: object, expected: Iterable[str], field: str) -> dict:
    expected_set = frozenset(expected)
    if type(value) is not dict or frozenset(value) != expected_set:
        raise BaselineError(f"{field} fields do not match the frozen schema")
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
            _matches_exactly(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected)
        )
    return actual == expected


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BaselineError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(content: bytes, field: str) -> object:
    try:
        return json.loads(
            content.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except UnicodeDecodeError as exc:
        raise BaselineError(f"{field} is not valid UTF-8") from exc


def _hash_stream(stream: BinaryIO) -> str:
    digest = sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _open_input_streams(
    paths: dict[str, Path], stack: ExitStack
) -> tuple[dict[str, BinaryIO], dict[str, tuple[int, int, int, int]]]:
    streams = {}
    identities = set()
    stability = {}
    for label, path in paths.items():
        try:
            path_metadata = path.lstat()
        except FileNotFoundError as exc:
            raise BaselineError(f"{label} input does not exist") from exc
        if not stat.S_ISREG(path_metadata.st_mode):
            raise BaselineError(f"{label} input must be a regular file, not an alias")

        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise BaselineError(f"{label} input could not be opened safely") from exc
        metadata = os.fstat(descriptor)
        identity = (metadata.st_dev, metadata.st_ino)
        if not stat.S_ISREG(metadata.st_mode) or identity != (
            path_metadata.st_dev,
            path_metadata.st_ino,
        ):
            os.close(descriptor)
            raise BaselineError(f"{label} input changed while it was opened")
        if identity in identities:
            os.close(descriptor)
            raise BaselineError("input files must not alias one another")
        identities.add(identity)
        streams[label] = stack.enter_context(os.fdopen(descriptor, "rb"))
        stability[label] = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
    return streams, stability


def _require_stable_streams(
    streams: dict[str, BinaryIO],
    expected: dict[str, tuple[int, int, int, int]],
) -> None:
    for label, stream in streams.items():
        metadata = os.fstat(stream.fileno())
        observed = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
        )
        if observed != expected[label]:
            raise BaselineError(f"{label} input changed while it was being read")


def _verify_input_hashes(
    streams: dict[str, BinaryIO], policy: _InputHashPolicy
) -> dict[str, str]:
    expected = {
        "train": policy.train_sha256,
        "validation": policy.validation_sha256,
        "preparation_summary": policy.preparation_summary_sha256,
        "contract": policy.contract_sha256,
    }
    for label, digest in expected.items():
        _lowercase_sha256(digest, f"{label} expected hash")

    observed = {label: _hash_stream(stream) for label, stream in streams.items()}
    for label in ("train", "validation", "preparation_summary", "contract"):
        if observed[label] != expected[label]:
            raise BaselineError(
                f"{label} SHA-256 mismatch: expected {expected[label]}, "
                f"observed {observed[label]}"
            )
    return observed


def _validate_contract(contract: object) -> None:
    contract = _expect_keys(
        contract,
        {
            "contract_id",
            "schema_version",
            "protocol_version",
            "input",
            "output",
            "predictor_policy",
            "features",
            "models",
            "partition_use",
            "score",
            "threshold_selection",
        },
        "baseline contract",
    )
    if contract["contract_id"] != CONTRACT_ID:
        raise BaselineError(f"contract_id must be {CONTRACT_ID}")
    if contract["schema_version"] != 1 or type(contract["schema_version"]) is not int:
        raise BaselineError("contract schema_version must be exact integer 1")
    if contract["protocol_version"] != "1.4":
        raise BaselineError("contract protocol_version must be 1.4")
    if not _matches_exactly(
        contract["input"],
        {
            "field": "raw_url",
            "validation": "canonical-url-v1 preparation rules",
            "missing_or_invalid_action": "error",
            "error_message": "raw_url is missing or invalid under canonical-url-v1",
            "imputation": False,
        },
    ):
        raise BaselineError("contract input policy does not match the frozen method")
    if not _matches_exactly(
        contract["output"],
        {
            "container": "ordered_vector",
            "dtype": "float64",
            "feature_count": 25,
        },
    ):
        raise BaselineError("contract output does not match the frozen feature vector")

    predictor_policy = _expect_keys(
        contract["predictor_policy"], {"basis", "forbidden"}, "predictor policy"
    )
    if predictor_policy["basis"] != "raw_url only":
        raise BaselineError("predictor basis must be raw_url only")
    if type(predictor_policy["forbidden"]) is not list or not all(
        type(item) is str and item for item in predictor_policy["forbidden"]
    ):
        raise BaselineError("forbidden predictors must be a list of strings")

    features = contract["features"]
    if type(features) is not list or len(features) != len(FEATURE_NAMES):
        raise BaselineError("contract must contain exactly 25 features")
    for position, (feature, name) in enumerate(zip(features, FEATURE_NAMES), start=1):
        feature = _expect_keys(
            feature,
            {"position", "name", "definition", "dtype", "denominator"},
            f"feature {position}",
        )
        if (
            feature["position"] != position
            or type(feature["position"]) is not int
            or feature["name"] != name
            or feature["dtype"] != "float64"
        ):
            raise BaselineError("contract feature order or type has changed")
        _nonempty_string(feature["definition"], f"feature {position} definition")
        _nonempty_string(feature["denominator"], f"feature {position} denominator")

    models = _expect_keys(
        contract["models"],
        {"common_pipeline", "length-only", "Logistic-L1", "search"},
        "models",
    )
    common = _expect_keys(
        models["common_pipeline"],
        {"scaler", "classifier", "convergence_warning_action"},
        "common pipeline",
    )
    if not _matches_exactly(common["scaler"], _SCALER_CONFIG):
        raise BaselineError("scaler configuration does not match the frozen method")
    if not _matches_exactly(common["classifier"], _CLASSIFIER_CONFIG):
        raise BaselineError("classifier configuration does not match the frozen method")
    if common["convergence_warning_action"] != "error" or models["search"] != "none":
        raise BaselineError("model warning or search policy has changed")
    for model_name, feature_names in _MODEL_FEATURES.items():
        model = _expect_keys(models[model_name], {"features"}, model_name)
        if model["features"] != list(feature_names):
            raise BaselineError(f"{model_name} features do not match the contract")
    if not _matches_exactly(
        contract["partition_use"],
        {
            "fit_scaler_and_classifier": "train only",
            "select_threshold": "validation only",
        },
    ):
        raise BaselineError("partition-use policy has changed")
    if not _matches_exactly(
        contract["score"],
        {
            "value": "P(is_phishing=1)",
            "alert_rule": "score >= threshold",
        },
    ):
        raise BaselineError("score policy has changed")
    if not _matches_exactly(
        contract["threshold_selection"],
        {
            "candidate_thresholds": (
                "unique validation scores plus the finite no-alert threshold "
                "nextafter(maximum validation score, +infinity)"
            ),
            "objective": "maximize validation recall",
            "constraint": (
                "exact one-sided 95% Clopper-Pearson false-positive-rate upper "
                "confidence bound <= 0.01"
            ),
            "tie_breaks": [
                "smaller false-positive-rate upper confidence bound",
                "higher threshold",
            ],
            "no_feasible_candidate": "target_not_met",
        },
    ):
        raise BaselineError("threshold-selection policy has changed")


def _validate_string_list(value: object, field: str) -> None:
    if (
        type(value) is not list
        or not value
        or not all(type(item) is str and item for item in value)
    ):
        raise BaselineError(f"{field} must be a nonempty string list")


def _validate_preparation_summary(summary: object) -> dict:
    summary = _expect_keys(
        summary,
        {
            "schema_version",
            "source_spec_sha256",
            "declared_sources",
            "label_mapping",
            "algorithms",
            "overall_counts",
            "native_label_counts",
            "local_label_counts",
            "splits",
            "quarantine_reason_counts",
            "output_hashes",
        },
        "preparation summary",
    )
    if summary["schema_version"] != 1 or type(summary["schema_version"]) is not int:
        raise BaselineError(
            "preparation summary schema_version must be exact integer 1"
        )
    _lowercase_sha256(summary["source_spec_sha256"], "source_spec_sha256")

    sources = _expect_keys(
        summary["declared_sources"],
        {"contract_id", "schema_version", "phiusiil", "public_suffix_list"},
        "declared_sources",
    )
    if sources["contract_id"] != "phiusiil-development-v1":
        raise BaselineError("development source contract is not recognized")
    if sources["schema_version"] != 2 or type(sources["schema_version"]) is not int:
        raise BaselineError("declared source schema_version must be exact integer 2")
    phiusiil = _expect_keys(
        sources["phiusiil"],
        {
            "uci_dataset_id",
            "paper_doi",
            "archive_url",
            "archive_sha256",
            "csv_filename",
            "csv_sha256",
            "license",
            "page_url",
            "native_label_semantics",
            "publisher_reported_class_sources",
            "publisher_reported_phishing_retrieval_window",
            "publisher_reported_legitimate_collection_window",
            "reference_classification_basis",
            "public_per_row_provenance_available",
        },
        "declared_sources.phiusiil",
    )
    if phiusiil["uci_dataset_id"] != 967 or type(phiusiil["uci_dataset_id"]) is not int:
        raise BaselineError("PhiUSIIL UCI dataset ID must be exact integer 967")
    for key in (
        "paper_doi",
        "archive_url",
        "csv_filename",
        "license",
        "page_url",
        "reference_classification_basis",
    ):
        _nonempty_string(phiusiil[key], f"declared_sources.phiusiil.{key}")
    _lowercase_sha256(phiusiil["archive_sha256"], "PhiUSIIL archive hash")
    source_csv_sha256 = _lowercase_sha256(phiusiil["csv_sha256"], "PhiUSIIL CSV hash")
    if phiusiil["native_label_semantics"] != {
        "0": "phishing",
        "1": "legitimate",
    }:
        raise BaselineError("native label semantics do not match the source record")
    class_sources = _expect_keys(
        phiusiil["publisher_reported_class_sources"],
        {"legitimate", "phishing"},
        "publisher-reported class sources",
    )
    _validate_string_list(class_sources["legitimate"], "legitimate class sources")
    _validate_string_list(class_sources["phishing"], "phishing class sources")
    window = _expect_keys(
        phiusiil["publisher_reported_phishing_retrieval_window"],
        {"start", "end"},
        "phishing retrieval window",
    )
    _nonempty_string(window["start"], "phishing window start")
    _nonempty_string(window["end"], "phishing window end")
    legitimate_window = phiusiil["publisher_reported_legitimate_collection_window"]
    if legitimate_window is not None:
        _expect_keys(legitimate_window, {"start", "end"}, "legitimate window")
    provenance = _expect_keys(
        phiusiil["public_per_row_provenance_available"],
        {"source", "timestamp", "snapshot", "independent_adjudication"},
        "per-row provenance",
    )
    if any(type(value) is not bool for value in provenance.values()):
        raise BaselineError("per-row provenance flags must be exact booleans")

    psl = _expect_keys(
        sources["public_suffix_list"],
        {"url", "upstream_url", "version", "commit", "sha256", "license"},
        "declared_sources.public_suffix_list",
    )
    for key in ("url", "upstream_url", "version", "commit", "license"):
        _nonempty_string(psl[key], f"public_suffix_list.{key}")
    _lowercase_sha256(psl["sha256"], "public suffix list hash")

    if not _matches_exactly(
        summary["label_mapping"],
        {
            "version": "phiusiil-native-label-map-v1",
            "native_label_meanings": {"0": "phishing", "1": "legitimate"},
            "native_to_is_phishing": {"0": 1, "1": 0},
            "is_phishing_meanings": {"0": "legitimate", "1": "phishing"},
        },
    ):
        raise BaselineError("label mapping does not match the preparation contract")

    algorithms = _expect_keys(
        summary["algorithms"],
        {
            "record_identifier_version",
            "canonicalization_version",
            "domain_split_version",
            "seed",
            "allocation_version",
            "allocation_basis",
            "split_percentages",
        },
        "algorithms",
    )
    for key in (
        "record_identifier_version",
        "canonicalization_version",
        "domain_split_version",
        "seed",
        "allocation_version",
        "allocation_basis",
    ):
        _nonempty_string(algorithms[key], f"algorithms.{key}")
    percentages = _expect_keys(
        algorithms["split_percentages"],
        {"train", "validation", "group_test"},
        "split percentages",
    )
    for key, value in percentages.items():
        _exact_int(value, f"split percentage {key}")

    overall = _expect_keys(
        summary["overall_counts"],
        {
            "input_rows",
            "canonicalized_url_groups",
            "retained_rows",
            "retained_domains",
            "quarantined_rows",
        },
        "overall counts",
    )
    for key, value in overall.items():
        _exact_int(value, f"overall_counts.{key}")
    for count_name, expected_keys in (
        ("native_label_counts", {"0", "1", "invalid"}),
        ("local_label_counts", {"0", "1"}),
    ):
        counts = _expect_keys(summary[count_name], expected_keys, count_name)
        for key, value in counts.items():
            _exact_int(value, f"{count_name}.{key}")

    splits = _expect_keys(
        summary["splits"], {"train", "validation", "group_test"}, "splits"
    )
    for split_name, split in splits.items():
        split = _expect_keys(
            split, {"row_count", "domain_count", "class_counts"}, split_name
        )
        _exact_int(split["row_count"], f"{split_name}.row_count", minimum=1)
        _exact_int(split["domain_count"], f"{split_name}.domain_count", minimum=1)
        classes = _expect_keys(
            split["class_counts"], {"0", "1"}, f"{split_name}.class_counts"
        )
        for label, value in classes.items():
            _exact_int(value, f"{split_name}.class_counts.{label}", minimum=1)
        if sum(classes.values()) != split["row_count"]:
            raise BaselineError(
                f"{split_name} declared class counts do not sum to rows"
            )

    quarantine = _expect_keys(
        summary["quarantine_reason_counts"],
        {
            "invalid_or_missing_url",
            "invalid_phiusiil_native_label",
            "canonical_url_conflicting_mapping",
            "canonical_url_duplicate_same_mapping",
        },
        "quarantine reason counts",
    )
    for key, value in quarantine.items():
        _exact_int(value, f"quarantine_reason_counts.{key}")

    output_hashes = _expect_keys(
        summary["output_hashes"],
        {
            "train.jsonl",
            "validation.jsonl",
            "group_test.jsonl",
            "quarantine.jsonl",
            "SHA256SUMS",
        },
        "output hashes",
    )
    for key, value in output_hashes.items():
        _lowercase_sha256(value, f"output_hashes.{key}")
    return {
        "source_csv_sha256": source_csv_sha256,
        "splits": splits,
        "output_hashes": output_hashes,
    }


def _validate_record_id(value: object, source_csv_sha256: str, field: str) -> str:
    prefix = f"phiusiil-row-v1:{source_csv_sha256}:"
    if (
        type(value) is not str
        or not value.startswith(prefix)
        or len(value) != len(prefix) + 16
        or any(character not in _LOWERCASE_HEX for character in value[-16:])
    ):
        raise BaselineError(f"{field} does not match phiusiil-row-v1")
    return value


def _load_partition(
    stream: BinaryIO,
    *,
    split: str,
    declared: dict,
    source_csv_sha256: str,
) -> tuple[np.ndarray, np.ndarray, set[str], set[int], dict[str, int]]:
    row_count = declared["row_count"]
    features = np.empty((row_count, len(FEATURE_NAMES)), dtype=np.float64)
    labels = np.empty(row_count, dtype=np.int8)
    domains: set[str] = set()
    record_ordinals: set[int] = set()
    previous_record_id: str | None = None
    class_counts = {"0": 0, "1": 0}
    observed_rows = 0

    for line_number, raw_line in enumerate(stream, start=1):
        if observed_rows >= row_count:
            raise BaselineError(f"{split} has more rows than declared")
        if not raw_line.endswith(b"\n") or raw_line == b"\n":
            raise BaselineError(f"{split} line {line_number} is not canonical JSONL")
        try:
            record = json.loads(
                raw_line.decode("utf-8"),
                object_pairs_hook=_object_without_duplicate_keys,
            )
        except UnicodeDecodeError as exc:
            raise BaselineError(f"{split} line {line_number} is not UTF-8") from exc
        record = _expect_keys(record, _RECORD_FIELDS, f"{split} record")
        record_id = _validate_record_id(
            record["record_id"], source_csv_sha256, f"{split} record_id"
        )
        if previous_record_id is not None and record_id <= previous_record_id:
            raise BaselineError(f"{split} record_id order must be strictly increasing")
        previous_record_id = record_id
        ordinal = int(record_id[-16:], 16)
        if ordinal in record_ordinals:
            raise BaselineError(f"{split} record identifiers must be unique")
        record_ordinals.add(ordinal)
        if record["split"] != split or type(record["split"]) is not str:
            raise BaselineError(f"{split} record has an incorrect split tag")
        raw_url = _nonempty_string(record["raw_url"], f"{split} raw_url")
        _lowercase_sha256(
            record["canonical_url_sha256"], f"{split} canonical_url_sha256"
        )
        domain = _nonempty_string(
            record["registrable_domain"], f"{split} registrable_domain"
        )
        label = record["is_phishing"]
        if type(label) is not int or label not in (0, 1):
            raise BaselineError(f"{split} is_phishing must be exact integer 0 or 1")
        try:
            vector = extract_url_features(raw_url)
        except FeatureExtractionError as exc:
            raise BaselineError(f"{split} feature extraction failed: {exc}") from exc
        if len(vector) != len(FEATURE_NAMES) or not all(map(math.isfinite, vector)):
            raise BaselineError(f"{split} feature vector must contain 25 finite values")
        features[observed_rows, :] = vector
        labels[observed_rows] = label
        domains.add(domain)
        class_counts[str(label)] += 1
        observed_rows += 1

    if observed_rows != row_count:
        raise BaselineError(
            f"{split} observed {observed_rows} rows but declared {row_count}"
        )
    if class_counts != declared["class_counts"]:
        raise BaselineError(
            f"{split} observed class counts do not match declared counts"
        )
    if set(labels.tolist()) != {0, 1}:
        raise BaselineError(f"{split} must contain both classes")
    if len(domains) != declared["domain_count"]:
        raise BaselineError(f"{split} domain count does not match the declared count")
    return features, labels, domains, record_ordinals, class_counts


def clopper_pearson_upper(false_positives: int, negative_count: int) -> float:
    """Return the exact one-sided 95% binomial upper confidence bound."""
    false_positives = _exact_int(false_positives, "false_positives")
    negative_count = _exact_int(negative_count, "negative_count", minimum=1)
    if false_positives > negative_count:
        raise BaselineError("false_positives must not exceed negative_count")
    if false_positives == negative_count:
        return 1.0
    return float(
        betaincinv(false_positives + 1, negative_count - false_positives, 0.95)
    )


def select_validation_threshold(
    scores: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    """Select the validation threshold fixed by the RQ1 baseline contract."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if scores.ndim != 1 or labels.ndim != 1 or scores.shape != labels.shape:
        raise BaselineError("validation scores and labels must be equal-length vectors")
    if scores.size == 0 or not np.issubdtype(scores.dtype, np.number):
        raise BaselineError("validation scores must be a nonempty numeric vector")
    if not np.all(np.isfinite(scores)):
        raise BaselineError("validation scores must be finite")
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise BaselineError("validation scores must be probabilities in [0, 1]")
    if not np.issubdtype(labels.dtype, np.integer) or set(labels.tolist()) != {0, 1}:
        raise BaselineError("validation labels must contain exact integers 0 and 1")

    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = int(labels.size - positive_count)
    unique_scores = np.unique(scores.astype(np.float64, copy=False))
    no_alert_threshold = float(np.nextafter(unique_scores[-1], np.inf))
    if not math.isfinite(no_alert_threshold):
        raise BaselineError("finite no-alert threshold is unavailable")
    candidates = [
        {
            "threshold": no_alert_threshold,
            "true_positive": 0,
            "false_positive": 0,
        }
    ]

    order = np.argsort(scores, kind="stable")[::-1]
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    true_positive = 0
    false_positive = 0
    index = 0
    while index < len(ordered_scores):
        threshold = float(ordered_scores[index])
        next_index = index
        while (
            next_index < len(ordered_scores) and ordered_scores[next_index] == threshold
        ):
            if ordered_labels[next_index] == 1:
                true_positive += 1
            else:
                false_positive += 1
            next_index += 1
        candidates.append(
            {
                "threshold": threshold,
                "true_positive": true_positive,
                "false_positive": false_positive,
            }
        )
        index = next_index

    feasible = []
    for candidate in candidates:
        upper = clopper_pearson_upper(candidate["false_positive"], negative_count)
        if upper <= 0.01:
            candidate["upper"] = upper
            candidate["recall"] = candidate["true_positive"] / positive_count
            feasible.append(candidate)
    candidate_count = len(candidates)
    if not feasible:
        return {
            "status": "target_not_met",
            "threshold": None,
            "candidate_count": candidate_count,
            "counts": None,
            "recall": None,
            "observed_fpr": None,
            "fpr_upper_95": None,
        }

    selected = max(
        feasible,
        key=lambda item: (item["recall"], -item["upper"], item["threshold"]),
    )
    true_positive = selected["true_positive"]
    false_positive = selected["false_positive"]
    return {
        "status": "selected",
        "threshold": selected["threshold"],
        "candidate_count": candidate_count,
        "counts": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": negative_count - false_positive,
            "false_negative": positive_count - true_positive,
            "positive": positive_count,
            "negative": negative_count,
        },
        "recall": selected["recall"],
        "observed_fpr": false_positive / negative_count,
        "fpr_upper_95": selected["upper"],
    }


def _software_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }


def _fit_model(
    model_name: str,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    input_hashes: dict[str, str],
) -> dict:
    feature_names = _MODEL_FEATURES[model_name]
    feature_indices = np.asarray(
        [FEATURE_NAMES.index(name) for name in feature_names], dtype=np.intp
    )
    train_matrix = train_features[:, feature_indices]
    validation_matrix = validation_features[:, feature_indices]
    scaler_kwargs = {
        key: value
        for key, value in _SCALER_CONFIG.items()
        if key not in {"class", "fit_partition"}
    }
    scaler = StandardScaler(**scaler_kwargs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            scaled_train = scaler.fit_transform(train_matrix)
            scaled_validation = scaler.transform(validation_matrix)
    except (ValueError, Warning, FloatingPointError) as exc:
        raise BaselineError(f"{model_name} scaling failed: {exc}") from exc
    if not np.all(np.isfinite(scaled_train)) or not np.all(
        np.isfinite(scaled_validation)
    ):
        raise BaselineError(f"{model_name} scaling produced nonfinite values")
    classifier_kwargs = {
        key: value for key, value in _CLASSIFIER_CONFIG.items() if key != "class"
    }
    classifier = LogisticRegression(**classifier_kwargs)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            classifier.fit(scaled_train, train_labels)
    except (ValueError, Warning, FloatingPointError) as exc:
        if isinstance(exc, ConvergenceWarning):
            raise BaselineError(f"{model_name} did not converge") from exc
        raise BaselineError(f"{model_name} fitting failed: {exc}") from exc
    if classifier.classes_.tolist() != [0, 1]:
        raise BaselineError(f"{model_name} classifier classes are not [0, 1]")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Accelerate-backed NumPy may emit spurious matmul warnings here even when
        # every input and output is finite. The explicit check below remains fatal.
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
            module=r"sklearn\.utils\.extmath",
        )
        try:
            scores = classifier.predict_proba(scaled_validation)[:, 1]
        except (ValueError, Warning, FloatingPointError) as exc:
            raise BaselineError(f"{model_name} scoring failed: {exc}") from exc
    if not np.all(np.isfinite(scores)):
        raise BaselineError(f"{model_name} produced nonfinite validation scores")
    threshold = select_validation_threshold(scores, validation_labels)
    n_samples_seen = scaler.n_samples_seen_
    if np.ndim(n_samples_seen) != 0:
        raise BaselineError("scaler sample count is unexpectedly feature-specific")
    return {
        "schema_version": 1,
        "artifact_type": "rq1-baseline-model",
        "analysis_stage": ANALYSIS_STAGE,
        "contract_id": CONTRACT_ID,
        "contract_sha256": input_hashes["contract"],
        "model_name": model_name,
        "features": list(feature_names),
        "classes": [int(value) for value in classifier.classes_.tolist()],
        "scaler": {
            "config": _SCALER_CONFIG,
            "mean": [float(value) for value in scaler.mean_.tolist()],
            "scale": [float(value) for value in scaler.scale_.tolist()],
            "variance": [float(value) for value in scaler.var_.tolist()],
            "n_samples_seen": int(n_samples_seen),
        },
        "classifier": {
            "config": _CLASSIFIER_CONFIG,
            "coefficients": [
                [float(value) for value in row] for row in classifier.coef_.tolist()
            ],
            "intercept": [float(value) for value in classifier.intercept_.tolist()],
            "n_iter": [int(value) for value in classifier.n_iter_.tolist()],
        },
        "validation_threshold": threshold,
        "input_hashes": input_hashes,
        "software_versions": _software_versions(),
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
    }


def _json_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
        )
    except ValueError as exc:
        raise BaselineError("artifact contains a nonfinite number") from exc
    return f"{text}\n".encode("ascii")


def _write_file(path: Path, content: bytes, mode: int) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _write_summary_temp(summary_path: Path, content: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{summary_path.name}.tmp-", dir=summary_path.parent
    )
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return Path(temporary_name)


def _publish_path_without_replace(source: Path, destination: Path) -> None:
    if sys.platform == "darwin":
        function = ctypes.CDLL(None, use_errno=True).renameatx_np
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(-2, os.fsencode(source), -2, os.fsencode(destination), 4)
    elif sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            function = library.renameat2
        except AttributeError as exc:
            raise BaselineError("atomic no-replace publication is unavailable") from exc
        function.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        function.restype = ctypes.c_int
        result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    elif os.name == "nt":
        try:
            os.rename(source, destination)
        except FileExistsError as exc:
            raise BaselineError("publication destination already exists") from exc
        return
    else:
        raise BaselineError("atomic no-replace publication is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in (errno.EEXIST, errno.ENOTEMPTY):
            raise BaselineError("publication destination already exists")
        raise OSError(error_number, os.strerror(error_number), os.fspath(destination))


def _build_summary(
    *,
    artifacts: dict[str, dict],
    artifact_hashes: dict[str, str],
    input_hashes: dict[str, str],
    train_counts: dict[str, int],
    validation_counts: dict[str, int],
) -> dict:
    models = {}
    for model_name in ("length-only", "Logistic-L1"):
        artifact = artifacts[model_name]
        filename = _MODEL_FILENAMES[model_name]
        models[model_name] = {
            "artifact": filename,
            "artifact_sha256": artifact_hashes[filename],
            "feature_count": len(artifact["features"]),
            "n_iter": artifact["classifier"]["n_iter"],
            "validation_threshold": artifact["validation_threshold"],
        }
    return {
        "schema_version": 1,
        "analysis_stage": ANALYSIS_STAGE,
        "contract_id": CONTRACT_ID,
        "input_hashes": input_hashes,
        "input_counts": {
            "train": {
                "0": train_counts["0"],
                "1": train_counts["1"],
                "rows": train_counts["0"] + train_counts["1"],
            },
            "validation": {
                "0": validation_counts["0"],
                "1": validation_counts["1"],
                "rows": validation_counts["0"] + validation_counts["1"],
            },
        },
        "pipeline": {
            "scaler": _SCALER_CONFIG,
            "classifier": _CLASSIFIER_CONFIG,
            "convergence_warning_action": "error",
            "score": "P(is_phishing=1)",
            "alert_rule": "score >= threshold",
            "threshold_constraint": (
                "exact one-sided 95% Clopper-Pearson FPR upper bound <= 0.01"
            ),
        },
        "models": models,
        "software_versions": _software_versions(),
        "hypothesis_status": {"H1": "undecided", "H2": "undecided", "H3": "undecided"},
        "access": {"group_test_accessed": False, "phishvn_accessed": False},
    }


def _publish_artifacts(
    *,
    output_dir: Path,
    summary_path: Path,
    artifacts: dict[str, dict],
    input_hashes: dict[str, str],
    train_counts: dict[str, int],
    validation_counts: dict[str, int],
) -> dict:
    temporary_output = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    os.chmod(temporary_output, 0o700)
    temporary_summary: Path | None = None
    try:
        artifact_hashes = {}
        for model_name in ("length-only", "Logistic-L1"):
            filename = _MODEL_FILENAMES[model_name]
            content = _json_bytes(artifacts[model_name])
            _write_file(temporary_output / filename, content, 0o600)
            artifact_hashes[filename] = sha256(content).hexdigest()
        checksum_bytes = "".join(
            f"{artifact_hashes[name]}  {name}\n" for name in sorted(artifact_hashes)
        ).encode("ascii")
        _write_file(temporary_output / "SHA256SUMS", checksum_bytes, 0o600)
        summary = _build_summary(
            artifacts=artifacts,
            artifact_hashes=artifact_hashes,
            input_hashes=input_hashes,
            train_counts=train_counts,
            validation_counts=validation_counts,
        )
        summary_bytes = _json_bytes(summary)
        temporary_summary = _write_summary_temp(summary_path, summary_bytes)

        _publish_path_without_replace(temporary_output, output_dir)
        temporary_output = None
        _publish_path_without_replace(temporary_summary, summary_path)
        temporary_summary = None
        return summary
    finally:
        if temporary_output is not None:
            shutil.rmtree(temporary_output, ignore_errors=True)
        if temporary_summary is not None:
            temporary_summary.unlink(missing_ok=True)


def _validate_paths(
    *,
    input_paths: dict[str, Path],
    output_dir: Path,
    summary_path: Path,
) -> None:
    if os.path.lexists(output_dir):
        raise BaselineError("output directory already exists")
    if os.path.lexists(summary_path):
        raise BaselineError("summary path already exists")
    if not output_dir.parent.is_dir():
        raise BaselineError("output directory parent must already exist")
    if not summary_path.parent.is_dir():
        raise BaselineError("summary parent must already exist")
    resolved_output = output_dir.resolve(strict=False)
    resolved_summary = summary_path.resolve(strict=False)
    if resolved_summary == resolved_output:
        raise BaselineError("summary must not alias the private output directory")
    try:
        resolved_summary.relative_to(resolved_output)
    except ValueError:
        pass
    else:
        raise BaselineError("summary must be outside the private output directory")
    for label, input_path in input_paths.items():
        resolved_input = input_path.resolve(strict=True)
        if resolved_input in (resolved_output, resolved_summary):
            raise BaselineError(f"{label} input aliases an output destination")


def _fit_baselines(
    *,
    train_path: Path,
    validation_path: Path,
    preparation_summary_path: Path,
    contract_path: Path,
    output_dir: Path,
    summary_path: Path,
    _input_hash_policy: _InputHashPolicy,
) -> dict:
    input_paths = {
        "train": Path(train_path),
        "validation": Path(validation_path),
        "preparation_summary": Path(preparation_summary_path),
        "contract": Path(contract_path),
    }
    output_dir = Path(output_dir)
    summary_path = Path(summary_path)
    _validate_paths(
        input_paths=input_paths, output_dir=output_dir, summary_path=summary_path
    )
    with ExitStack() as stack:
        input_streams, input_stability = _open_input_streams(input_paths, stack)
        input_hashes = _verify_input_hashes(input_streams, _input_hash_policy)
        _require_stable_streams(input_streams, input_stability)
        preparation_summary = _validate_preparation_summary(
            _load_json_bytes(
                input_streams["preparation_summary"].read(), "preparation summary"
            )
        )
        _validate_contract(
            _load_json_bytes(input_streams["contract"].read(), "contract")
        )
        if preparation_summary["output_hashes"]["train.jsonl"] != input_hashes["train"]:
            raise BaselineError("train hash does not match the preparation summary")
        if (
            preparation_summary["output_hashes"]["validation.jsonl"]
            != input_hashes["validation"]
        ):
            raise BaselineError(
                "validation hash does not match the preparation summary"
            )

        train_features, train_labels, train_domains, train_ordinals, train_counts = (
            _load_partition(
                input_streams["train"],
                split="train",
                declared=preparation_summary["splits"]["train"],
                source_csv_sha256=preparation_summary["source_csv_sha256"],
            )
        )
        (
            validation_features,
            validation_labels,
            validation_domains,
            validation_ordinals,
            validation_counts,
        ) = _load_partition(
            input_streams["validation"],
            split="validation",
            declared=preparation_summary["splits"]["validation"],
            source_csv_sha256=preparation_summary["source_csv_sha256"],
        )
        _require_stable_streams(input_streams, input_stability)

    if train_domains & validation_domains:
        raise BaselineError("registrable domain crosses train and validation")
    if train_ordinals & validation_ordinals:
        raise BaselineError("record identifier crosses train and validation")
    del train_domains, validation_domains, train_ordinals, validation_ordinals

    artifacts = {
        model_name: _fit_model(
            model_name,
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            input_hashes,
        )
        for model_name in ("length-only", "Logistic-L1")
    }
    return _publish_artifacts(
        output_dir=output_dir,
        summary_path=summary_path,
        artifacts=artifacts,
        input_hashes=input_hashes,
        train_counts=train_counts,
        validation_counts=validation_counts,
    )


def fit_baselines(
    *,
    train_path: Path,
    validation_path: Path,
    preparation_summary_path: Path,
    contract_path: Path,
    output_dir: Path,
    summary_path: Path,
) -> dict:
    """Fit the two frozen baselines without exposing an evaluation-data input."""
    return _fit_baselines(
        train_path=train_path,
        validation_path=validation_path,
        preparation_summary_path=preparation_summary_path,
        contract_path=contract_path,
        output_dir=output_dir,
        summary_path=summary_path,
        _input_hash_policy=_OFFICIAL_INPUT_HASH_POLICY,
    )
