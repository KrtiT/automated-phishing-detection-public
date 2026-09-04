"""Run the frozen RQ1 SAGA convergence diagnostic on training data only."""

from __future__ import annotations

import json
import math
import multiprocessing
import subprocess
import sys
import time
import warnings
from collections.abc import Callable
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from automated_phishing_detection import baselines
from automated_phishing_detection.url_features import FEATURE_NAMES

DIAGNOSTIC_ID = "rq1-saga-convergence-v1"
REPOSITORY = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPOSITORY / "data/processed/phiusiil-v1/train.jsonl"
PREPARATION_SUMMARY_PATH = REPOSITORY / "reports/phiusiil-preparation-summary.json"
CONTRACT_PATH = REPOSITORY / "data/rq1-baseline-contract.json"
UV_LOCK_PATH = REPOSITORY / "uv.lock"
UV_LOCK_SHA256 = "15fadb4ad1f3c702a902b40d587a55294e7a26c33f268e8708ba8d941e6a51f0"

INPUT_PATHS = {
    "train": TRAIN_PATH,
    "preparation_summary": PREPARATION_SUMMARY_PATH,
}
INPUT_HASHES = {
    "contract": "594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4",
    "preparation_summary": (
        "1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e"
    ),
    "train": "575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0",
}
SCALER_CONFIG = {
    "class": "StandardScaler",
    "with_mean": True,
    "with_std": True,
}
CLASSIFIER_CONFIG = {
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
MODEL_FEATURES = {
    "length-only": ("raw_url_codepoint_length",),
    "Logistic-L1": FEATURE_NAMES,
}


class UsageError(RuntimeError):
    """Raised when the no-argument execution contract is violated."""


def _constructor_config(config: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in config.items() if key != "class"}


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(arguments: list[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY,
        capture_output=True,
        check=True,
        text=True,
    )
    return completed.stdout.strip()


def _require_tracked_path(path: Path, description: str) -> None:
    relative_path = path.relative_to(REPOSITORY).as_posix()
    try:
        tracked_path = _git_output(["ls-files", "--error-unmatch", "--", relative_path])
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"{description} must be tracked by Git HEAD") from error
    if tracked_path != relative_path:
        raise RuntimeError(f"{description} must be tracked by Git HEAD")


def _git_identity() -> dict[str, object]:
    head = _git_output(["rev-parse", "HEAD"])
    if len(head) != 40 or any(
        character not in "0123456789abcdef" for character in head
    ):
        raise RuntimeError("Git HEAD is not a lowercase commit identifier")
    tracked_status = _git_output(["status", "--porcelain=v1", "--untracked-files=no"])
    if tracked_status:
        raise RuntimeError("tracked worktree must be clean before the diagnostic")
    _require_tracked_path(Path(__file__).resolve(), "diagnostic executable")
    _require_tracked_path(CONTRACT_PATH, "baseline contract")
    _require_tracked_path(UV_LOCK_PATH, "uv.lock")
    uv_lock_sha256 = _sha256_path(UV_LOCK_PATH)
    if uv_lock_sha256 != UV_LOCK_SHA256:
        raise RuntimeError("uv.lock hash does not match the frozen value")
    return {
        "git_head": head,
        "tracked_worktree_clean": True,
        "uv_lock_sha256": uv_lock_sha256,
    }


def _load_training_partition() -> tuple[np.ndarray, np.ndarray]:
    if _sha256_path(CONTRACT_PATH) != INPUT_HASHES["contract"]:
        raise RuntimeError("contract hash does not match the frozen value")
    with ExitStack() as stack:
        streams, stability = baselines._open_input_streams(INPUT_PATHS, stack)
        observed_train = baselines._hash_stream(streams["train"])
        observed_summary = baselines._hash_stream(streams["preparation_summary"])
        if observed_train != INPUT_HASHES["train"]:
            raise RuntimeError("training input hash does not match the frozen value")
        if observed_summary != INPUT_HASHES["preparation_summary"]:
            raise RuntimeError(
                "preparation summary hash does not match the frozen value"
            )

        summary = baselines._validate_preparation_summary(
            baselines._load_json_bytes(
                streams["preparation_summary"].read(), "preparation summary"
            )
        )
        if summary["output_hashes"]["train.jsonl"] != observed_train:
            raise RuntimeError(
                "training input hash does not match the preparation record"
            )
        features, labels, _, _, _ = baselines._load_partition(
            streams["train"],
            split="train",
            declared=summary["splits"]["train"],
            source_csv_sha256=summary["source_csv_sha256"],
        )
        baselines._require_stable_streams(streams, stability)
    return features, labels


def _normalized_array(value: object, dtype: str) -> dict[str, object]:
    array = np.asarray(value, dtype=np.dtype(dtype))
    if dtype == "<f8" and not np.all(np.isfinite(array)):
        raise RuntimeError("fitted state contains a nonfinite value")
    return {
        "dtype": dtype,
        "hex": array.tobytes(order="C").hex(),
        "shape": list(array.shape),
    }


def _state_sha256(
    model_name: str,
    feature_names: tuple[str, ...],
    scaler: StandardScaler,
    classifier: LogisticRegression,
) -> str:
    ordered_state = [
        {"name": "model_name", "value": model_name},
        {"name": "features", "value": list(feature_names)},
        {"name": "scaler_config", "value": SCALER_CONFIG},
        {"name": "classifier_config", "value": CLASSIFIER_CONFIG},
        {"name": "input_hashes", "value": INPUT_HASHES},
        {"name": "scaler.mean_", "value": _normalized_array(scaler.mean_, "<f8")},
        {
            "name": "scaler.scale_",
            "value": _normalized_array(scaler.scale_, "<f8"),
        },
        {"name": "scaler.var_", "value": _normalized_array(scaler.var_, "<f8")},
        {
            "name": "scaler.n_samples_seen_",
            "value": _normalized_array(scaler.n_samples_seen_, "<i8"),
        },
        {
            "name": "classifier.classes_",
            "value": _normalized_array(classifier.classes_, "<i8"),
        },
        {
            "name": "classifier.coef_",
            "value": _normalized_array(classifier.coef_, "<f8"),
        },
        {
            "name": "classifier.intercept_",
            "value": _normalized_array(classifier.intercept_, "<f8"),
        },
        {
            "name": "classifier.n_iter_",
            "value": _normalized_array(classifier.n_iter_, "<i8"),
        },
    ]
    canonical = json.dumps(
        {"ordered_state": ordered_state, "schema_version": 1},
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(canonical).hexdigest()


def _candidate_summary(
    model_name: str,
    feature_names: tuple[str, ...],
    scaler: StandardScaler,
    classifier: LogisticRegression,
    scaled_features: np.ndarray,
    decision_scores: np.ndarray,
    probabilities: np.ndarray,
    elapsed_seconds: float,
) -> dict[str, object]:
    expected_width = len(feature_names)
    scaled_features = np.asarray(scaled_features)
    decision_scores = np.asarray(decision_scores)
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classifier.classes_)
    coefficients = np.asarray(classifier.coef_)
    intercept = np.asarray(classifier.intercept_)
    iterations = np.asarray(classifier.n_iter_)
    scaler_values = tuple(
        np.asarray(value) for value in (scaler.mean_, scaler.scale_, scaler.var_)
    )

    if scaled_features.ndim != 2 or scaled_features.shape[1] != expected_width:
        raise RuntimeError(f"{model_name} scaled feature shape is incorrect")
    sample_count = scaled_features.shape[0]
    if any(value.shape != (expected_width,) for value in scaler_values):
        raise RuntimeError(f"{model_name} scaler parameter shape is incorrect")
    samples_seen = np.asarray(scaler.n_samples_seen_)
    if samples_seen.ndim != 0 or int(samples_seen) != sample_count:
        raise RuntimeError(f"{model_name} scaler sample count is incorrect")
    if classes.shape != (2,) or classes.tolist() != [0, 1]:
        raise RuntimeError(f"{model_name} classes are not [0, 1]")
    if coefficients.shape != (1, expected_width):
        raise RuntimeError(f"{model_name} coefficient shape is incorrect")
    if intercept.shape != (1,):
        raise RuntimeError(f"{model_name} intercept shape is incorrect")
    if iterations.shape != (1,):
        raise RuntimeError(f"{model_name} iteration-count shape is incorrect")
    n_iter = int(iterations[0])
    if not 0 < n_iter < CLASSIFIER_CONFIG["max_iter"]:
        raise RuntimeError(f"{model_name} did not stop below max_iter")
    if decision_scores.shape != (sample_count,):
        raise RuntimeError(f"{model_name} decision-score shape is incorrect")
    if probabilities.shape != (sample_count, 2):
        raise RuntimeError(f"{model_name} probability shape is incorrect")

    finite_values = (
        scaled_features,
        *scaler_values,
        coefficients,
        intercept,
        decision_scores,
        probabilities,
    )
    if not all(np.all(np.isfinite(value)) for value in finite_values):
        raise RuntimeError(f"{model_name} produced a nonfinite fitted value")
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise RuntimeError(f"{model_name} elapsed time is not finite and nonnegative")

    return {
        "elapsed_seconds": elapsed_seconds,
        "feature_count": expected_width,
        "n_iter": n_iter,
        "nonzero_coefficient_count": int(np.count_nonzero(coefficients)),
        "state_sha256": _state_sha256(model_name, feature_names, scaler, classifier),
    }


def _fit_candidate(
    model_name: str, features: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    feature_names = MODEL_FEATURES[model_name]
    indices = np.asarray(
        [FEATURE_NAMES.index(name) for name in feature_names], dtype=np.intp
    )
    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError("training feature matrix does not match the frozen vector")
    if labels.shape != (features.shape[0],) or set(labels.tolist()) != {0, 1}:
        raise RuntimeError("training labels must be a two-class vector")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scaler = StandardScaler(**_constructor_config(SCALER_CONFIG))
        classifier = LogisticRegression(**_constructor_config(CLASSIFIER_CONFIG))
        started = time.perf_counter()
        scaled_features = scaler.fit_transform(features[:, indices])
        classifier.fit(scaled_features, labels)
        decision_scores = classifier.decision_function(scaled_features)
        probabilities = classifier.predict_proba(scaled_features)
        elapsed_seconds = time.perf_counter() - started

    return _candidate_summary(
        model_name,
        feature_names,
        scaler,
        classifier,
        scaled_features,
        decision_scores,
        probabilities,
        elapsed_seconds,
    )


def _context_record(environment: dict[str, object] | None) -> dict[str, object]:
    return {
        "configuration": {
            "classifier": CLASSIFIER_CONFIG,
            "scaler": SCALER_CONFIG,
        },
        "diagnostic_id": DIAGNOSTIC_ID,
        "environment": environment,
        "input_hashes": INPUT_HASHES,
        "model_features": {
            name: list(feature_names) for name, feature_names in MODEL_FEATURES.items()
        },
        "schema_version": 1,
        "scope": "training_only",
    }


def _planned_record(
    environment: dict[str, object] | None,
    models: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        **_context_record(environment),
        "models": models,
    }


def _failure_record(
    error: Exception,
    *,
    environment: dict[str, object] | None,
    models: dict[str, dict[str, object]],
    model_name: str | None = None,
) -> dict[str, object]:
    failure = {
        "message": str(error),
        "type": type(error).__name__,
    }
    if model_name is not None:
        failure["model_name"] = model_name
    return {
        **_planned_record(environment, models),
        "failure": failure,
        "run_passed": False,
    }


def _execute_single_run() -> dict[str, object]:
    models: dict[str, dict[str, object]] = {}
    try:
        initial_identity = _git_identity()
    except Exception as error:
        return _failure_record(error, environment=None, models=models)

    current_model: str | None = None
    try:
        features, labels = _load_training_partition()
        for model_name in MODEL_FEATURES:
            current_model = model_name
            models[model_name] = _fit_candidate(model_name, features, labels)
        current_model = None
        final_identity = _git_identity()
        if final_identity != initial_identity:
            raise RuntimeError("Git identity changed during the diagnostic")
    except Exception as error:
        return _failure_record(
            error,
            environment=initial_identity,
            models=models,
            model_name=current_model,
        )

    return {
        **_planned_record(initial_identity, models),
        "run_passed": True,
    }


def _child_process_entry(sending_connection) -> None:
    try:
        result = _execute_single_run()
        sending_connection.send(result)
    finally:
        sending_connection.close()


def _run_fresh_process() -> dict[str, object]:
    context = multiprocessing.get_context("spawn")
    receiving_connection, sending_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_process_entry,
        args=(sending_connection,),
    )
    try:
        process.start()
    except Exception:
        receiving_connection.close()
        sending_connection.close()
        raise
    sending_connection.close()
    try:
        try:
            result = receiving_connection.recv()
        except EOFError:
            result = _failure_record(
                RuntimeError("fresh child process exited without a record"),
                environment=None,
                models={},
            )
    finally:
        receiving_connection.close()
        process.join()

    if process.exitcode != 0:
        return _failure_record(
            RuntimeError(f"fresh child process exited with code {process.exitcode}"),
            environment=None,
            models={},
        )
    if type(result) is not dict:
        return _failure_record(
            RuntimeError("fresh child process returned an invalid record"),
            environment=None,
            models={},
        )
    return result


def _overall_result(
    environment: dict[str, object] | None,
    runs: list[dict[str, object]],
    *,
    failure: dict[str, object] | None = None,
) -> dict[str, object]:
    result = {
        **_context_record(environment),
        "runs": runs,
        "status": "failed" if failure is not None else "passed",
    }
    if failure is not None:
        result["failure"] = failure
    return result


def _coordinate_fresh_runs(
    run_fresh_process: Callable[[], dict[str, object]] | None = None,
) -> dict[str, object]:
    runner = _run_fresh_process if run_fresh_process is None else run_fresh_process
    runs: list[dict[str, object]] = []
    for _ in range(2):
        try:
            run = runner()
        except Exception as error:
            run = _failure_record(error, environment=None, models={})
        if type(run) is not dict:
            run = _failure_record(
                RuntimeError("fresh child process returned an invalid record"),
                environment=None,
                models={},
            )
        runs.append(run)

    first_environment = runs[0].get("environment")
    environment = first_environment if type(first_environment) is dict else None
    failed_runs = []
    for run_number, run in enumerate(runs, start=1):
        if run.get("run_passed") is True:
            continue
        failure = run.get("failure")
        if type(failure) is not dict:
            failure = {
                "message": "fresh child process returned an invalid run record",
                "type": "InvalidFreshRunRecord",
            }
        failed_runs.append({"failure": failure, "run_number": run_number})
    if failed_runs:
        return _overall_result(
            environment,
            runs,
            failure={
                "message": "one or more fresh runs failed",
                "runs": failed_runs,
                "type": "FreshRunFailure",
            },
        )

    second_environment = runs[1].get("environment")
    if type(first_environment) is not dict or type(second_environment) is not dict:
        return _overall_result(
            environment,
            runs,
            failure={
                "message": "fresh-run environment identity is missing",
                "type": "FreshRunEnvironmentMismatch",
            },
        )
    if first_environment != second_environment:
        return _overall_result(
            environment,
            runs,
            failure={
                "message": "fresh-run environment identities do not match",
                "run_1": first_environment,
                "run_2": second_environment,
                "type": "FreshRunEnvironmentMismatch",
            },
        )

    mismatches = []
    try:
        first_models = runs[0]["models"]
        second_models = runs[1]["models"]
        for model_name in MODEL_FEATURES:
            for field in ("n_iter", "state_sha256"):
                first_value = first_models[model_name][field]
                second_value = second_models[model_name][field]
                if first_value != second_value:
                    mismatches.append(
                        {
                            "field": field,
                            "model_name": model_name,
                            "run_1": first_value,
                            "run_2": second_value,
                        }
                    )
    except (KeyError, TypeError):
        return _overall_result(
            environment,
            runs,
            failure={
                "message": "fresh child process returned an invalid model aggregate",
                "type": "InvalidFreshRunRecord",
            },
        )
    if mismatches:
        return _overall_result(
            environment,
            runs,
            failure={
                "message": "fresh-run model aggregates do not match",
                "mismatches": mismatches,
                "type": "FreshRunMismatch",
            },
        )
    return _overall_result(environment, runs)


def _execute_diagnostic() -> dict[str, object]:
    return _coordinate_fresh_runs()


def _print_json(value: object) -> None:
    print(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def main() -> int:
    if len(sys.argv) != 1:
        usage_error = UsageError("this diagnostic accepts no arguments")
        result = _overall_result(
            None,
            [],
            failure={
                "message": str(usage_error),
                "type": type(usage_error).__name__,
            },
        )
    else:
        result = _execute_diagnostic()
    _print_json(result)
    if result["status"] == "failed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
