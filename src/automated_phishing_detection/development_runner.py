"""One recorded attempt for the fixed development-only secondary family.

The development profile binds train/validation inputs without opening the
protected evaluation boundary. Completed members are retained immediately; a
later failure never erases them, retries a fit, or creates a partial success.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning

from . import (
    execution_receipt,
    fixed_cascade,
    gmm_monitor,
    secondary_development,
    secondary_metrics,
    secondary_tabular,
)
from .development_execution import (
    STEPS,
    DevelopmentExecutionBinding,
    bind_development_execution,
    recheck_development_binding,
)
from .development_execution import DevelopmentExecutionError as DevelopmentBindingError
from .execution_receipt import publish_completion, record_failure, reserve_attempt
from .source_runner import SourceExecutionError, _json, _read_file_once


class DevelopmentExecutionError(ValueError):
    """A symbolic failure without private inputs, paths, or exception details."""


_ERROR_SYMBOLS = {
    ConvergenceWarning: "ConvergenceWarning",
    FloatingPointError: "FloatingPointError",
    OverflowError: "OverflowError",
    RuntimeWarning: "RuntimeWarning",
    ValueError: "ValueError",
    TypeError: "TypeError",
    RuntimeError: "RuntimeError",
    OSError: "OSError",
    MemoryError: "MemoryError",
    UnicodeError: "UnicodeError",
    UnicodeDecodeError: "UnicodeDecodeError",
    KeyError: "KeyError",
    AssertionError: "AssertionError",
    DevelopmentExecutionError: "DevelopmentExecutionError",
    DevelopmentBindingError: "DevelopmentExecutionError",
    SourceExecutionError: "SourceExecutionError",
    execution_receipt.ExecutionReceiptError: "ExecutionReceiptError",
    fixed_cascade.FixedCascadeError: "FixedCascadeError",
    gmm_monitor.GMMMonitorError: "GMMMonitorError",
    secondary_development.SecondaryDevelopmentError: "SecondaryDevelopmentError",
    secondary_tabular.SecondaryTabularError: "SecondaryTabularError",
    secondary_metrics.SecondaryMetricsError: "SecondaryMetricsError",
}


def _failure_symbol(error: BaseException) -> str:
    """Retain a known causal class without copying names or messages from errors."""
    symbol, seen = "Exception", set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        symbol = _ERROR_SYMBOLS.get(type(error), symbol)
        # Suppressed display context still carries the original numerical failure.
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return symbol


@dataclass(frozen=True)
class DevelopmentRunPaths:
    train: Path
    validation: Path
    suffix_rules: Path
    logistic_l1: Path
    gmm: Path
    attempt: Path
    public_summary: Path


def _output_paths(binding, paths):
    if type(paths) is not DevelopmentRunPaths:
        raise DevelopmentExecutionError("invalid_run_paths")
    for path in (paths.attempt, paths.public_summary):
        absolute = execution_receipt._absolute_path(path)
        if absolute.is_relative_to(binding.base.root):
            raise DevelopmentExecutionError("outputs_must_be_outside_checkout")
        with execution_receipt._directory(absolute.parent) as parent:
            execution_receipt._require_absent(parent, absolute.name)
    if paths.public_summary.absolute().is_relative_to(paths.attempt.absolute()):
        raise DevelopmentExecutionError("public_summary_inside_attempt")


def _identity(binding):
    return {
        "kind": "secondary_development",
        "revision": binding.base.revision,
        "execution_contract_sha256": binding.base.contract_sha256,
        "development_profile_sha256": binding.profile_sha256,
        "methods_contract_sha256": binding.methods_sha256,
        "runtime_sha256": sha256(binding.base.runtime_json.encode()).hexdigest(),
        "pins": asdict(binding.pins),
        "ordered_steps": list(STEPS),
    }


def _read_bound(path, expected):
    content = _read_file_once(path)
    if sha256(content).hexdigest() != expected:
        raise DevelopmentExecutionError("input_hash_mismatch")
    return content


def _drift(binding, paths):
    pins = binding.pins
    suffix = _read_bound(paths.suffix_rules, pins.suffix_rules_sha256)
    logistic_bytes = _read_bound(paths.logistic_l1, pins.logistic_l1_artifact_sha256)
    logistic = fixed_cascade._load_logistic_l1_artifact_bytes(
        logistic_bytes,
        expected_sha256=pins.logistic_l1_artifact_sha256,
        expected_contract_sha256=pins.baseline_contract_sha256,
    )
    gmm = gmm_monitor.load_gmm_artifact_bytes(
        _read_bound(paths.gmm, pins.gmm_artifact_sha256)
    )
    train_content = _read_bound(paths.train, pins.train_sha256)
    reference = secondary_development.build_training_reference(
        train_content,
        preparation_summary=binding.preparation_bytes,
        suffix_rules=suffix,
        logistic_l1=logistic,
        gmm_state=gmm,
        pins=pins,
    )
    # Validation bytes cannot enter either training-reference constructor.
    validation_content = _read_bound(paths.validation, pins.validation_sha256)
    result = secondary_development.evaluate_validation(reference, validation_content)
    if set(result.private_outputs) != {
        "training-reference.json",
        "validation-audit.json",
    }:
        raise DevelopmentExecutionError("unexpected_drift_outputs")
    train = tuple(_json(line) for line in train_content.splitlines())
    validation = tuple(_json(line) for line in validation_content.splitlines())
    return result.private_outputs, result.public_summary, train, validation


def _tabular(member, train, validation):
    arguments = (
        tuple(row["raw_url"] for row in train),
        tuple(row["is_phishing"] for row in train),
        tuple(row["raw_url"] for row in validation),
        tuple(row["is_phishing"] for row in validation),
    )
    seed = 42
    if member == "formatting":
        kind = "formatting"
        fitted = secondary_tabular.fit_formatting(*arguments)
    elif member == "random_forest":
        kind = "random_forest"
        fitted = secondary_tabular.fit_random_forest(*arguments)
    else:
        kind = "permutation"
        seed = int(member.removeprefix("permutation_"))
        fitted = secondary_tabular.fit_label_permutation(*arguments, seed=seed)
    if len(fitted.validation_scores) != len(validation):
        raise DevelopmentExecutionError("prediction_count_mismatch")
    with secondary_tabular._numerical_runtime():
        metrics = {
            "average_precision": float(
                secondary_metrics.average_precision_score(
                    arguments[3], fitted.validation_scores
                )
            ),
            "roc_auc": float(
                secondary_metrics.roc_auc_score(arguments[3], fitted.validation_scores)
            ),
        }
    private = {
        "model.json": fitted.artifact_bytes,
        "validation-predictions.jsonl": b"".join(
            secondary_tabular._json_bytes(
                {
                    "record_id": row["record_id"],
                    "label": row["is_phishing"],
                    "probability": probability,
                }
            )
            for row, probability in zip(validation, fitted.validation_scores)
        ),
        "threshold.json": secondary_tabular._json_bytes(fitted.validation_threshold),
        "scoring-audit.json": secondary_tabular._json_bytes(fitted.scoring_audit),
    }
    public = {
        "model_kind": kind,
        "analysis_role": "descriptive_secondary_not_primary",
        "seed": seed,
        "row_count": len(validation),
        "class_counts": {
            str(label): sum(row["is_phishing"] == label for row in validation)
            for label in (0, 1)
        },
        "validation_threshold": fitted.validation_threshold,
        "scoring_audit": fitted.scoring_audit,
        "score_metrics": metrics,
    }
    return private, public


def _hashes(outputs):
    return {name: sha256(content).hexdigest() for name, content in outputs.items()}


def _run_bound_development(
    binding: DevelopmentExecutionBinding, paths: DevelopmentRunPaths
) -> Path:
    root_attempt = child_attempt = None
    root_publishing = child_publishing = False
    stage = "preflight"
    try:
        recheck_development_binding(binding)
        _output_paths(binding, paths)
        identity = _identity(binding)
        stage = "reservation"
        root_attempt = reserve_attempt(paths.attempt, identity=identity)
        members = []
        for member in STEPS:
            stage = member
            child_publishing = False
            child_attempt = reserve_attempt(
                root_attempt.directory / member,
                identity={
                    "root_reservation_sha256": root_attempt.reservation_sha256,
                    "member": member,
                },
            )
            if member == "drift":
                private, result, train, validation = _drift(binding, paths)
            else:
                private, result = _tabular(member, train, validation)
            summary = {
                "schema_version": 1,
                "status": "development_member_completed",
                "member": member,
                "root_reservation_sha256": root_attempt.reservation_sha256,
                "reservation_sha256": child_attempt.reservation_sha256,
                "private_sha256": _hashes(private),
                "result": result,
            }
            summary_bytes = execution_receipt._json_bytes(summary, "member_summary")
            child_publishing = True
            publish_completion(
                child_attempt,
                private_outputs=private,
                public_summary=summary,
                public_path=root_attempt.directory / f"{member}.json",
            )
            members.append(
                {
                    "member": member,
                    "public_summary_sha256": sha256(summary_bytes).hexdigest(),
                    "summary": summary,
                }
            )
            child_attempt = None
        stage = "final_binding"
        recheck_development_binding(binding)
        bindings = {**identity, "reservation_sha256": root_attempt.reservation_sha256}
        private = {
            "bindings.json": execution_receipt._json_bytes(bindings, "bindings"),
            "members.json": execution_receipt._json_bytes(
                {"schema_version": 1, "members": members}, "members"
            ),
        }
        public = {
            "schema_version": 1,
            "status": "development_evidence_published",
            "protected_evaluation_authorized": False,
            "execution": bindings,
            "members": members,
            "private_sha256": _hashes(private),
        }
        stage = "publication"
        root_publishing = True
        return publish_completion(
            root_attempt,
            private_outputs=private,
            public_summary=public,
            public_path=paths.public_summary,
        )
    except Exception as exc:
        error_symbol = _failure_symbol(exc)
        failed_receipt = False
        for attempt, publishing in (
            (child_attempt, child_publishing),
            (root_attempt, root_publishing),
        ):
            if attempt is not None and not publishing:
                try:
                    record_failure(attempt, stage=stage, error_type=error_symbol)
                except Exception:
                    failed_receipt = True
        reason = "failure_record_incomplete" if failed_receipt else error_symbol
        raise DevelopmentExecutionError(f"{stage}: {reason}") from None


def run_development(
    root: Path,
    *,
    expected_revision: str,
    expected_profile_sha256: str,
    paths: DevelopmentRunPaths,
) -> Path:
    """Bind the development profile before inspecting any supplied file path."""
    binding = bind_development_execution(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _run_bound_development(binding, paths)


def run_development_process(
    root: Path,
    *,
    expected_revision: str,
    expected_profile_sha256: str,
    paths: DevelopmentRunPaths,
) -> dict:
    """Verify retained evidence independently after the fresh worker exits."""
    binding = bind_development_execution(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    from .development_completion import verify_development_completion

    command = [
        sys.executable,
        str(binding.base.root / "scripts/run_secondary_development.py"),
        "--worker",
        "--repo-root",
        str(binding.base.root),
        "--expected-revision",
        expected_revision,
        "--expected-profile-sha256",
        expected_profile_sha256,
    ]
    for name in (
        "train",
        "validation",
        "suffix_rules",
        "logistic_l1",
        "gmm",
        "attempt",
        "public_summary",
    ):
        command.extend(("--" + name.replace("_", "-"), str(getattr(paths, name))))
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError:
        raise DevelopmentExecutionError("worker_launch_failed") from None
    return verify_development_completion(
        binding, paths, producer_exit_code=result.returncode
    )
