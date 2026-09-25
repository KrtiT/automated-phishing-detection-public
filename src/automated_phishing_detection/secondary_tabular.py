"""Fixed secondary estimators over supplied in-memory development examples.

No paths, data readers, authorization, research execution or primary artifacts
are provided. Callers bind training/validation provenance and execution identity
before using these primitives on research data. Tests use invented URLs only.
The secondary CP operating point is descriptive, including permutation controls;
it is not a primary cutoff or a permutation-test p-value.
"""

from __future__ import annotations

import json
import math
import platform
import warnings
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from . import baselines, fixed_cascade, gmm_monitor, secondary_probes
from .url_features import FEATURE_NAMES, extract_url_features

_VERSIONS = {
    "python": "3.10.19",
    "numpy": "2.2.6",
    "scipy": "1.15.3",
    "scikit-learn": "1.7.2",
    "threadpoolctl": "3.6.0",
}
_KINDS = ("formatting", "permutation", "random_forest")
_FIELDS = {
    "schema_version",
    "contract_id",
    "artifact_type",
    "method_version",
    "analysis_stage",
    "protected_evaluation_authorized",
    "model_kind",
    "features",
    "classes",
    "parameters",
    "scoring",
    "permutation",
    "training_row_count",
    "software_versions",
    "state",
}
_LOGISTIC_SCORING = {
    "input_dtype": "float64",
    "batch_size": 1,
    "probability": "sklearn_predict_proba_class_1",
    "reference": "baseline_float64_einsum_expit_audit",
    "warnings": "fatal",
    "numerical_threads": 1,
}
_RF_SCORING = {
    "input_dtype": "float32",
    "batch_size": 1,
    "tree_order": "fitted_estimator_order",
    "probability": "sequential_float64_sum_of_normalized_leaf_values_divided_by_100",
    "warnings": "fatal",
    "numerical_threads": 1,
}
_RF_SCORING_V2 = {
    **_RF_SCORING,
    "probability": "sequential_float64_sum_of_stored_leaf_values_divided_by_100",
}
_CHECK_IDS = frozenset(
    {
        "input_validation",
        "fit",
        "checkpoint_write",
        "fitted_classes",
        "fitted_state",
        "validation_score",
        "portable_exact_parity",
        "threshold_selection",
    }
)
_NONFINITE_TAGS = {"nan", "positive_infinity", "negative_infinity"}


class SecondaryTabularError(ValueError):
    """A secondary input, fixed runtime or portable model is invalid."""

    def __init__(self, *args, check_id: str | None = None):
        if check_id is not None and (
            type(check_id) is not str or check_id not in _CHECK_IDS
        ):
            raise ValueError("unknown secondary check identifier")
        super().__init__(*args)
        self._check_id = check_id

    @property
    def check_id(self) -> str | None:
        return self._check_id


@contextmanager
def _check_phase(check_id):
    try:
        yield
    except SecondaryTabularError as exc:
        if exc.check_id is None:
            exc._check_id = check_id
        raise
    except Exception as exc:
        raise SecondaryTabularError(
            "secondary fitting failed", check_id=check_id
        ) from exc


@dataclass(frozen=True)
class SecondaryFitResult:
    artifact_bytes: bytes
    validation_scores: tuple[float, ...]
    _threshold_json: str
    _audit_json: str

    @property
    def validation_threshold(self) -> dict:
        return json.loads(self._threshold_json)

    @property
    def scoring_audit(self) -> dict:
        return json.loads(self._audit_json)


@dataclass(frozen=True)
class SecondaryModel:
    artifact_bytes: bytes

    def score_urls(self, raw_urls) -> tuple[float, ...]:
        """Score from strict numeric state only, without invoking any fit method."""
        try:
            with _numerical_runtime():
                artifact = _decode_model(self.artifact_bytes)
                matrix = _features(raw_urls, artifact["model_kind"])
                scores, _ = _score_state(artifact, matrix)
                return tuple(float(value) for value in scores)
        except SecondaryTabularError:
            raise
        except Exception as exc:
            raise SecondaryTabularError("secondary scoring failed") from exc

    def score_urls_singleton_ordered(self, raw_urls) -> tuple[float, ...]:
        """Score supplied URLs in order with one portable model call per URL."""
        try:
            _require(
                isinstance(raw_urls, Sequence)
                and not isinstance(raw_urls, (str, bytes)),
                "URLs must be a nonempty ordered sequence",
            )
            _require(
                len(raw_urls) > 0 and all(type(url) is str for url in raw_urls),
                "URLs must be exact strings",
            )
            urls = tuple(raw_urls)
            with _numerical_runtime():
                artifact = _decode_model(self.artifact_bytes)
                probabilities = []
                for raw_url in urls:
                    matrix = _features((raw_url,), artifact["model_kind"])
                    scores, _ = _score_state(artifact, matrix)
                    _require(
                        type(scores) is np.ndarray
                        and scores.shape == (1,)
                        and np.issubdtype(scores.dtype, np.floating)
                        and math.isfinite(float(scores[0]))
                        and 0 <= scores[0] <= 1,
                        "invalid singleton probability",
                    )
                    probabilities.append(float(scores[0]))
                return tuple(probabilities)
        except SecondaryTabularError:
            raise
        except Exception as exc:
            raise SecondaryTabularError("secondary scoring failed") from exc


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("ascii")


def _require(condition, message):
    if not condition:
        raise SecondaryTabularError(message)


def _keys(value, expected):
    _require(
        type(value) is dict and set(value) == set(expected), "invalid model fields"
    )


@contextmanager
def _numerical_runtime():
    gmm_monitor._require_runtime()
    observed = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "threadpoolctl": threadpoolctl.__version__,
    }
    _require(observed == _VERSIONS, "secondary runtime versions differ")
    with (
        warnings.catch_warnings(),
        np.errstate(all="raise"),
        threadpoolctl.threadpool_limits(limits=1),
    ):
        warnings.simplefilter("error")
        pools = threadpoolctl.threadpool_info()
        _require(
            bool(pools) and all(pool["num_threads"] == 1 for pool in pools),
            "secondary runtime requires one numerical thread",
        )
        yield


def _logistic():
    return LogisticRegression(
        **{
            key: value
            for key, value in baselines._CLASSIFIER_CONFIG.items()
            if key != "class"
        }
    )


def _scaler():
    return StandardScaler(copy=True, with_mean=True, with_std=True)


def _forest():
    return RandomForestClassifier(
        n_estimators=100,
        criterion="gini",
        max_features="sqrt",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
        oob_score=False,
        warm_start=False,
    )


def _parameters(kind):
    if kind == "random_forest":
        return _forest().get_params(deep=False)
    return {
        "classifier": _logistic().get_params(deep=False),
        "scaler": _scaler().get_params(deep=False),
    }


def _feature_names(kind):
    return list(
        secondary_probes.FORMATTING_FEATURE_NAMES
        if kind == "formatting"
        else FEATURE_NAMES
    )


def _features(raw_urls, kind):
    _require(
        isinstance(raw_urls, Sequence) and not isinstance(raw_urls, (str, bytes)),
        "URLs must be a nonempty ordered sequence",
    )
    _require(
        len(raw_urls) > 0 and all(type(url) is str for url in raw_urls),
        "URLs must be exact strings",
    )
    extractor = (
        secondary_probes.extract_formatting_features
        if kind == "formatting"
        else extract_url_features
    )
    matrix = np.ascontiguousarray(
        [extractor(url) for url in raw_urls], dtype=np.float64
    )
    _require(np.all(np.isfinite(matrix)), "secondary features must be finite")
    return matrix


def _labels(labels, count):
    _require(
        isinstance(labels, Sequence) or isinstance(labels, np.ndarray),
        "labels must be an aligned binary sequence",
    )
    _require(
        not isinstance(labels, (str, bytes)) and len(labels) == count,
        "labels must align with URLs",
    )
    _require(
        all(
            type(value) is int
            or isinstance(value, np.integer)
            and not isinstance(value, np.bool_)
            for value in labels
        ),
        "labels must be exact binary integers",
    )
    _require(set(labels) == {0, 1}, "both binary classes are required")
    return np.ascontiguousarray(labels, dtype=np.int64)


def _array(value, shape, *, integer=False):
    _require(type(value) is list, "numeric state must use arrays")
    raw = np.asarray(value, dtype=object)
    predicate = (
        (lambda x: type(x) is int)
        if integer
        else (lambda x: type(x) in (int, float) and math.isfinite(x))
    )
    _require(
        raw.shape == shape and all(predicate(item) for item in raw.flat),
        "invalid numeric state",
    )
    return np.ascontiguousarray(value, dtype=np.int64 if integer else np.float64)


def _validate_logistic_state(state, width, training_rows):
    _keys(state, {"scaler", "coefficients", "intercept", "n_iter"})
    _keys(state["scaler"], {"mean", "scale", "variance", "n_samples_seen"})
    scaler = state["scaler"]
    _require(
        type(scaler["n_samples_seen"]) is int
        and scaler["n_samples_seen"] == training_rows,
        "scaler sample count differs",
    )
    _array(scaler["mean"], (width,))
    scale = _array(scaler["scale"], (width,))
    variance = _array(scaler["variance"], (width,))
    _require(
        np.all(scale > 0)
        and np.all(variance >= 0)
        and np.all(scale[variance == 0] == 1),
        "invalid scaler state",
    )
    _array(state["coefficients"], (1, width))
    _array(state["intercept"], (1,))
    iterations = _array(state["n_iter"], (1,), integer=True)
    _require(0 < iterations[0] < 5000, "secondary logistic did not converge")


def _validate_tree(tree, width):
    _keys(
        tree,
        {
            "children_left",
            "children_right",
            "feature",
            "threshold",
            "value",
            "random_state",
        },
    )
    _require(
        type(tree["children_left"]) is list and bool(tree["children_left"]),
        "tree must contain nodes",
    )
    count = len(tree["children_left"])
    left = _array(tree["children_left"], (count,), integer=True)
    right = _array(tree["children_right"], (count,), integer=True)
    features = _array(tree["feature"], (count,), integer=True)
    thresholds = _array(tree["threshold"], (count,))
    values = _array(tree["value"], (count, 2))
    _require(
        type(tree["random_state"]) is int and 0 <= tree["random_state"] < 2**31,
        "invalid tree seed",
    )
    _require(
        np.all(values >= 0) and np.allclose(values.sum(axis=1), 1, rtol=1e-12, atol=0),
        "invalid tree class distribution",
    )
    parents = np.zeros(count, dtype=np.int64)
    for index in range(count):
        if left[index] == -1:
            _require(
                right[index] == -1
                and features[index] == -2
                and thresholds[index] == -2,
                "invalid leaf node",
            )
        else:
            _require(
                index < left[index] < count
                and index < right[index] < count
                and left[index] != right[index]
                and 0 <= features[index] < width,
                "invalid tree topology",
            )
            parents[left[index]] += 1
            parents[right[index]] += 1
    _require(
        parents[0] == 0 and np.all(parents[1:] == 1), "disconnected or shared tree node"
    )


def _decode_model(content):
    _require(type(content) is bytes, "model must be exact bytes")
    artifact = json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )
    _require(_json_bytes(artifact) == content, "model must use canonical JSON")
    _keys(artifact, _FIELDS)
    rf_v2 = artifact["method_version"] == "secondary-rf-v2"
    fixed = {
        "schema_version": 1,
        "contract_id": "secondary-development-correction-v1"
        if rf_v2
        else "secondary-development-v1",
        "artifact_type": "secondary-tabular-model",
        "method_version": "secondary-rf-v2" if rf_v2 else "secondary-tabular-v1",
        "analysis_stage": "development_validation_only",
        "protected_evaluation_authorized": False,
        "classes": [0, 1],
        "software_versions": _VERSIONS,
    }
    _require(
        all(
            fixed_cascade._matches_exactly(artifact[key], value)
            for key, value in fixed.items()
        ),
        "invalid secondary identity",
    )
    kind = artifact["model_kind"]
    _require(type(kind) is str and kind in _KINDS, "unknown secondary model kind")
    _require(not rf_v2 or kind == "random_forest", "v2 is restricted to Random Forest")
    _require(
        artifact["features"] == _feature_names(kind), "invalid secondary feature order"
    )
    _require(
        fixed_cascade._matches_exactly(artifact["parameters"], _parameters(kind)),
        "secondary parameters differ",
    )
    scoring = (
        (_RF_SCORING_V2 if rf_v2 else _RF_SCORING)
        if kind == "random_forest"
        else _LOGISTIC_SCORING
    )
    _require(
        fixed_cascade._matches_exactly(artifact["scoring"], scoring),
        "secondary scoring convention differs",
    )
    if kind == "permutation":
        permutation = artifact["permutation"]
        _keys(permutation, {"bit_generator", "seed"})
        _require(
            permutation["bit_generator"] == "PCG64"
            and type(permutation["seed"]) is int
            and permutation["seed"] in range(42, 47),
            "invalid permutation identity",
        )
    else:
        _require(artifact["permutation"] is None, "unexpected label permutation")
    count = artifact["training_row_count"]
    _require(type(count) is int and count >= 2, "invalid training row count")
    width = len(artifact["features"])
    if kind == "random_forest":
        _keys(artifact["state"], {"trees"})
        trees = artifact["state"]["trees"]
        _require(
            type(trees) is list and len(trees) == 100, "forest must contain 100 trees"
        )
        for tree in trees:
            _validate_tree(tree, width)
    else:
        _validate_logistic_state(artifact["state"], width, count)
    return artifact


def load_secondary_model_bytes(content: bytes) -> SecondaryModel:
    """Validate strict portable state; no executable estimator deserialization."""
    try:
        _decode_model(content)
        return SecondaryModel(content)
    except SecondaryTabularError:
        raise
    except Exception as exc:
        raise SecondaryTabularError("invalid secondary model") from exc


def _diagnostic_numbers(value):
    if type(value) is float and not math.isfinite(value):
        return {
            "nonfinite": "nan"
            if math.isnan(value)
            else "positive_infinity"
            if value > 0
            else "negative_infinity"
        }
    if type(value) is list:
        return [_diagnostic_numbers(item) for item in value]
    if type(value) is dict:
        return {key: _diagnostic_numbers(item) for key, item in value.items()}
    return value


def _diagnostic_numeric_array(value):
    if type(value) is list:
        for item in value:
            _diagnostic_numeric_array(item)
    elif type(value) is dict:
        _require(
            set(value) == {"nonfinite"}
            and type(value["nonfinite"]) is str
            and value["nonfinite"] in _NONFINITE_TAGS,
            "invalid diagnostic numeric tag",
        )
    else:
        _require(
            type(value) in (int, float) and math.isfinite(value),
            "invalid diagnostic numeric state",
        )


def validate_fit_checkpoint_bytes(
    content: bytes, *, model_bytes: bytes | None = None
) -> dict:
    """Read diagnostic state only; it is never an accepted scoring artifact."""
    try:
        _require(type(content) is bytes, "checkpoint must be exact bytes")
        value = json.loads(
            content,
            object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
            parse_constant=fixed_cascade._reject_json_constant,
        )
        _require(_json_bytes(value) == content, "checkpoint must use canonical JSON")
        _keys(value, _FIELDS | {"seed"})
        _require(
            value["artifact_type"] == "secondary-fit-checkpoint",
            "invalid checkpoint type",
        )
        kind = value["model_kind"]
        _require(type(kind) is str and kind in _KINDS, "invalid checkpoint kind")
        seed = value["seed"]
        _require(
            type(seed) is int
            and (seed in range(42, 47) if kind == "permutation" else seed == 42),
            "invalid checkpoint seed",
        )
        expected = _artifact(
            kind,
            value["training_row_count"],
            {},
            seed if kind == "permutation" else None,
            rf_v2=value["method_version"] == "secondary-rf-v2",
        )
        for key in _FIELDS - {"artifact_type", "state", "classes"}:
            _require(
                fixed_cascade._matches_exactly(value[key], expected[key]),
                "invalid checkpoint metadata",
            )
        _require(
            type(value["training_row_count"]) is int
            and value["training_row_count"] >= 2
            and (
                value["method_version"] != "secondary-rf-v2" or kind == "random_forest"
            ),
            "invalid checkpoint identity",
        )
        _require(type(value["classes"]) is list, "invalid checkpoint classes")
        _diagnostic_numeric_array(value["classes"])
        state = value["state"]
        if kind == "random_forest":
            _keys(state, {"trees"})
            _require(type(state["trees"]) is list, "invalid diagnostic forest")
            for tree in state["trees"]:
                _keys(
                    tree,
                    {
                        "children_left",
                        "children_right",
                        "feature",
                        "threshold",
                        "value",
                        "random_state",
                    },
                )
                for item in tree.values():
                    _diagnostic_numeric_array(item)
        else:
            _keys(state, {"scaler", "coefficients", "intercept", "n_iter"})
            _keys(state["scaler"], {"mean", "scale", "variance", "n_samples_seen"})
            for key in ("coefficients", "intercept", "n_iter"):
                _diagnostic_numeric_array(state[key])
            for item in state["scaler"].values():
                _diagnostic_numeric_array(item)
        if model_bytes is not None:
            model = _decode_model(model_bytes)
            _require(
                all(
                    fixed_cascade._matches_exactly(value[key], model[key])
                    for key in _FIELDS - {"artifact_type"}
                ),
                "checkpoint differs from accepted model",
            )
        return value
    except SecondaryTabularError:
        raise
    except Exception as exc:
        raise SecondaryTabularError("invalid fit checkpoint") from exc


def _logistic_scores(classifier, scaled):
    policy = json.loads(json.dumps(baselines._SCORING_INTEGRITY_POLICY))
    policy["allowed_warning"]["environment"] = {}
    environment = baselines._platform_identity()
    scores = []
    maximum_decision = maximum_probability = 0.0
    for row in scaled:
        values, audit = baselines._audited_validation_scores(
            "secondary_logistic",
            row.reshape(1, -1),
            classifier,
            policy=policy,
            environment=environment,
        )
        _require(not audit["warning_records"], "secondary scoring warning")
        scores.append(float(values[0]))
        maximum_decision = max(
            maximum_decision, audit["max_absolute_decision_difference"]
        )
        maximum_probability = max(
            maximum_probability, audit["max_absolute_probability_difference"]
        )
    return np.asarray(scores, dtype=np.float64), {
        "platform_identity": environment,
        "warning_records": [],
        "max_absolute_decision_difference": maximum_decision,
        "max_absolute_probability_difference": maximum_probability,
        "batch_size": 1,
    }


def _forest_scores(state, matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    _require(np.all(np.isfinite(matrix)), "forest float32 input is nonfinite")
    total = np.zeros((len(matrix), 2), dtype=np.float64)
    for tree in state["trees"]:
        probabilities = np.asarray(tree["value"], dtype=np.float64)
        for row_index, row in enumerate(matrix):
            node = 0
            while tree["children_left"][node] != -1:
                # sklearn promotes its float32 feature to C double for comparison.
                node = (
                    tree["children_left"][node]
                    if float(row[tree["feature"][node]]) <= tree["threshold"][node]
                    else tree["children_right"][node]
                )
            probability = probabilities[node].copy()
            probability /= probability.sum()
            total[row_index] += probability
    total /= len(state["trees"])
    return total[:, 1]


def _forest_scores_v2(state, matrix):
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    _require(np.all(np.isfinite(matrix)), "forest float32 input is nonfinite")
    total = np.zeros((len(matrix), 2), dtype=np.float64)
    for tree in state["trees"]:
        probabilities = np.asarray(tree["value"], dtype=np.float64)
        for row_index, row in enumerate(matrix):
            node = 0
            while tree["children_left"][node] != -1:
                node = (
                    tree["children_left"][node]
                    if float(row[tree["feature"][node]]) <= tree["threshold"][node]
                    else tree["children_right"][node]
                )
            # sklearn 1.7.2 stores class proportions; normalizing again changes bits.
            total[row_index] += probabilities[node]
    total /= len(state["trees"])
    _require(
        np.all(np.isfinite(total)) and np.all((total >= 0) & (total <= 1)),
        "forest probabilities are invalid",
    )
    return total[:, 1]


def _score_state(artifact, matrix):
    state = artifact["state"]
    if artifact["model_kind"] == "random_forest":
        score = (
            _forest_scores_v2
            if artifact["method_version"] == "secondary-rf-v2"
            else _forest_scores
        )
        return score(state, matrix), {"batch_size": 1, "warning_records": []}
    scaler = state["scaler"]
    scaled = matrix.copy(order="C")
    scaled -= np.asarray(scaler["mean"], dtype=np.float64)
    scaled /= np.asarray(scaler["scale"], dtype=np.float64)
    _require(np.all(np.isfinite(scaled)), "scaled features are nonfinite")
    classifier = _logistic()
    classifier.classes_ = np.array([0, 1], dtype=np.int64)
    classifier.coef_ = np.asarray(state["coefficients"], dtype=np.float64)
    classifier.intercept_ = np.asarray(state["intercept"], dtype=np.float64)
    classifier.n_features_in_ = matrix.shape[1]
    classifier.n_iter_ = np.asarray(state["n_iter"], dtype=np.int32)
    return _logistic_scores(classifier, scaled)


def _fit_logistic(train, labels, validation, *, checkpoint=None):
    with _check_phase("fit"):
        scaler = _scaler()
        scaled = np.ascontiguousarray(scaler.fit_transform(train), dtype=np.float64)
        _require(np.all(np.isfinite(scaled)), "training scaler is nonfinite")
        classifier = _logistic().fit(scaled, labels)
    state = {
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "variance": scaler.var_.tolist(),
            "n_samples_seen": scaler.n_samples_seen_.item()
            if isinstance(scaler.n_samples_seen_, np.generic)
            else scaler.n_samples_seen_,
        },
        "coefficients": classifier.coef_.tolist(),
        "intercept": classifier.intercept_.tolist(),
        "n_iter": classifier.n_iter_.tolist(),
    }
    if checkpoint is not None:
        checkpoint(
            state,
            classifier.classes_.tolist(),
            {
                "classifier": classifier.get_params(deep=False),
                "scaler": scaler.get_params(deep=False),
            },
        )
    with _check_phase("fitted_classes"):
        _require(
            classifier.classes_.tolist() == [0, 1], "fitted logistic classes differ"
        )
    with _check_phase("fitted_state"):
        _validate_logistic_state(state, train.shape[1], len(train))
    with _check_phase("validation_score"):
        scores, audit = _logistic_scores(classifier, scaler.transform(validation))
    return state, scores, audit


def _fit_forest(train, labels, validation, *, checkpoint=None):
    with _check_phase("fit"):
        forest = _forest().fit(train, labels)
    state = {
        "trees": [
            {
                "children_left": tree.tree_.children_left.tolist(),
                "children_right": tree.tree_.children_right.tolist(),
                "feature": tree.tree_.feature.tolist(),
                "threshold": tree.tree_.threshold.tolist(),
                "value": tree.tree_.value[:, 0, :].tolist(),
                "random_state": tree.random_state,
            }
            for tree in forest.estimators_
        ]
    }
    if checkpoint is not None:
        checkpoint(state, forest.classes_.tolist(), forest.get_params(deep=False))
    with _check_phase("fitted_classes"):
        _require(forest.classes_.tolist() == [0, 1], "fitted forest classes differ")
    if checkpoint is not None:
        with _check_phase("fitted_state"):
            _require(len(state["trees"]) == 100, "forest must contain 100 trees")
            for tree in state["trees"]:
                _validate_tree(tree, train.shape[1])
    with _check_phase("validation_score"):
        scores = np.array(
            [forest.predict_proba(row.reshape(1, -1))[0, 1] for row in validation],
            dtype=np.float64,
        )
    return state, scores, {"batch_size": 1, "warning_records": []}


def _artifact(kind, training_count, state, seed, *, rf_v2=False):
    return {
        "schema_version": 1,
        "contract_id": "secondary-development-correction-v1"
        if rf_v2
        else "secondary-development-v1",
        "artifact_type": "secondary-tabular-model",
        "method_version": "secondary-rf-v2" if rf_v2 else "secondary-tabular-v1",
        "analysis_stage": "development_validation_only",
        "protected_evaluation_authorized": False,
        "model_kind": kind,
        "features": _feature_names(kind),
        "classes": [0, 1],
        "parameters": _parameters(kind),
        "state": state,
        "scoring": (_RF_SCORING_V2 if rf_v2 else _RF_SCORING)
        if kind == "random_forest"
        else _LOGISTIC_SCORING,
        "permutation": {"bit_generator": "PCG64", "seed": seed}
        if seed is not None
        else None,
        "training_row_count": training_count,
        "software_versions": _VERSIONS,
    }


def _fit(
    kind,
    train_urls,
    train_labels,
    validation_urls,
    validation_labels,
    seed=None,
    *,
    checkpoint=None,
    rf_v2=False,
):
    try:
        with _numerical_runtime():
            with _check_phase("input_validation"):
                _require(
                    checkpoint is None or callable(checkpoint),
                    "invalid checkpoint callback",
                )
                train = _features(train_urls, kind)
                validation = _features(validation_urls, kind)
                train_labels = _labels(train_labels, len(train))
                validation_labels = _labels(validation_labels, len(validation))
                if kind == "permutation":
                    train_labels = np.random.Generator(
                        np.random.PCG64(seed)
                    ).permutation(train_labels)

            def retain(state, classes, parameters):
                try:
                    snapshot = _artifact(kind, len(train), state, seed, rf_v2=rf_v2)
                    snapshot.update(
                        artifact_type="secondary-fit-checkpoint",
                        classes=classes,
                        parameters=parameters,
                        seed=seed if seed is not None else 42,
                    )
                    checkpoint(_json_bytes(_diagnostic_numbers(snapshot)))
                except Exception as exc:
                    raise SecondaryTabularError(
                        "fit checkpoint write failed", check_id="checkpoint_write"
                    ) from exc

            state, scores, audit = (
                _fit_forest if kind == "random_forest" else _fit_logistic
            )(
                train,
                train_labels,
                validation,
                **({"checkpoint": retain} if checkpoint is not None else {}),
            )
            with _check_phase("fitted_state"):
                content = _json_bytes(
                    _artifact(kind, len(train), state, seed, rf_v2=rf_v2)
                )
                checked = _decode_model(content)
            with _check_phase("portable_exact_parity"):
                restored_scores, _ = _score_state(checked, validation)
                _require(
                    np.array_equal(scores, restored_scores),
                    "portable scores differ from fitted estimator",
                )
            with _check_phase("validation_score"):
                _require(
                    np.all(np.isfinite(scores))
                    and np.all((scores >= 0) & (scores <= 1)),
                    "secondary probabilities are invalid",
                )
            with _check_phase("threshold_selection"):
                threshold = baselines.select_validation_threshold(
                    scores, validation_labels
                )
            audit["portable_exact_parity"] = True
            audit["threshold_role"] = "secondary_descriptive_operating_point"
            return SecondaryFitResult(
                content,
                tuple(float(value) for value in scores),
                _json_bytes(threshold).decode("ascii"),
                _json_bytes(audit).decode("ascii"),
            )
    except SecondaryTabularError:
        raise
    except Exception as exc:
        raise SecondaryTabularError("secondary fitting failed") from exc


def fit_formatting(
    train_urls,
    train_labels,
    validation_urls,
    validation_labels,
    *,
    checkpoint: Callable[[bytes], None] | None = None,
) -> SecondaryFitResult:
    """Fit one fixed five-indicator scaler/SAGA model; no tuning or retry."""
    return _fit(
        "formatting",
        train_urls,
        train_labels,
        validation_urls,
        validation_labels,
        checkpoint=checkpoint,
    )


def fit_label_permutation(
    train_urls,
    train_labels,
    validation_urls,
    validation_labels,
    *,
    seed: int,
    checkpoint: Callable[[bytes], None] | None = None,
) -> SecondaryFitResult:
    """Fit one independently permuted negative control; solver seed stays 42."""
    with _check_phase("input_validation"):
        _require(
            type(seed) is int and seed in range(42, 47),
            "permutation seed must be 42 through 46",
        )
    return _fit(
        "permutation",
        train_urls,
        train_labels,
        validation_urls,
        validation_labels,
        seed,
        checkpoint=checkpoint,
    )


def fit_random_forest(
    train_urls,
    train_labels,
    validation_urls,
    validation_labels,
    *,
    checkpoint: Callable[[bytes], None] | None = None,
) -> SecondaryFitResult:
    """Fit the fixed 100-tree unscaled structural benchmark, never a cascade stage."""
    return _fit(
        "random_forest",
        train_urls,
        train_labels,
        validation_urls,
        validation_labels,
        checkpoint=checkpoint,
    )


def fit_random_forest_v2(
    train_urls,
    train_labels,
    validation_urls,
    validation_labels,
    *,
    checkpoint: Callable[[bytes], None],
) -> SecondaryFitResult:
    """Fit the corrected direct-leaf scorer, retaining state before later checks."""
    with _check_phase("input_validation"):
        _require(callable(checkpoint), "v2 requires a fitted-state checkpoint")
    return _fit(
        "random_forest",
        train_urls,
        train_labels,
        validation_urls,
        validation_labels,
        checkpoint=checkpoint,
        rf_v2=True,
    )
