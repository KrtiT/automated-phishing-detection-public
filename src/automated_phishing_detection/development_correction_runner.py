"""One supervised retained audit followed by one separately identified RF fit."""

from __future__ import annotations

import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from . import development_completion as completion
from . import execution_receipt as receipt
from . import (
    secondary_development,
    secondary_metrics,
    secondary_tabular,
    source_runner,
)
from .development_correction import (
    CorrectionError,
    bind_correction,
    recheck_correction,
)
from .development_runner import _failure_symbol

STAGES = ("retained_audit", "random_forest")
_RECORDS = {
    "fit-input.json",
    "fit-state.json",
    "failure-details.json",
    "retained_audit-process.json",
    "random_forest-process.json",
}
_CHECKS = {
    "input_validation",
    "fit",
    "checkpoint_write",
    "fitted_classes",
    "fitted_state",
    "validation_score",
    "portable_exact_parity",
    "threshold_selection",
    "retained_audit",
    "random_forest",
    "preflight",
    "publication",
    "verification",
    "reservation",
    "final_binding",
}
_RECEIPTS = {"reservation.json", "finalize.claim", "outcome.json"}
_TABULAR = {
    "model.json",
    "validation-predictions.jsonl",
    "threshold.json",
    "scoring-audit.json",
}


@dataclass(frozen=True)
class CorrectionPaths:
    train: Path
    validation: Path
    suffix_rules: Path
    original_attempt: Path
    attempt: Path
    public_summary: Path


def _require(condition, symbol):
    if not condition:
        raise CorrectionError(symbol)


def _json_bytes(value):
    return receipt._json_bytes(value, "correction_record")


def _record(attempt, name, content):
    _require(name in _RECORDS, "invalid_record_name")
    _require(type(content) is bytes, "invalid_record_bytes")
    with receipt._attempt_directory(attempt) as directory:
        if any(
            receipt._entry(directory, item) is not None
            for item in ("finalize.claim", "outcome.json")
        ):
            raise receipt.ExecutionReceiptError("attempt already finalized")
        receipt._install_record(directory, name, content)


def _identity(binding):
    legacy = binding.development
    return {
        "kind": "secondary_development_correction",
        "revision": legacy.base.revision,
        "profile_sha256": binding.profile_sha256,
        "base_development_profile_sha256": legacy.profile_sha256,
        "base_methods_sha256": legacy.methods_sha256,
        "original_accounting_sha256": sha256(binding.accounting_bytes).hexdigest(),
        "runtime_sha256": sha256(legacy.base.runtime_json.encode()).hexdigest(),
        "pins": asdict(legacy.pins),
        "stages": list(STAGES),
    }


def _root_attempt(binding, path):
    content = source_runner._read_file_once(path / "reservation.json")
    attempt = receipt.Attempt(path.absolute(), sha256(content).hexdigest())
    with receipt._attempt_directory(attempt) as directory:
        _require(
            all(
                receipt._entry(directory, name) is None
                for name in ("finalize.claim", "outcome.json", "evidence")
            ),
            "root_finalized",
        )
        record = source_runner._json(content)
        _require(
            completion._same(record["identity"], _identity(binding)),
            "root_identity_mismatch",
        )
    return attempt


def _outside_outputs(binding, paths):
    _require(type(paths) is CorrectionPaths, "invalid_paths")
    for path in (paths.attempt, paths.public_summary):
        absolute = receipt._absolute_path(path)
        _require(
            not absolute.is_relative_to(binding.development.base.root),
            "outputs_inside_checkout",
        )
        _require(
            not absolute.is_relative_to(paths.original_attempt.absolute()),
            "outputs_inside_original_attempt",
        )
        with receipt._directory(absolute.parent) as parent:
            receipt._require_absent(parent, absolute.name)
    _require(
        not paths.public_summary.absolute().is_relative_to(paths.attempt.absolute()),
        "marker_inside_attempt",
    )


def _bound_input(path, digest):
    content = source_runner._read_file_once(path)
    _require(sha256(content).hexdigest() == digest, "input_hash_mismatch")
    return content


def _inputs(binding, paths):
    legacy = binding.development
    suffix = _bound_input(paths.suffix_rules, legacy.pins.suffix_rules_sha256)
    train = _bound_input(paths.train, legacy.pins.train_sha256)
    validation = _bound_input(paths.validation, legacy.pins.validation_sha256)
    return train, validation, suffix


def _validated_rows(binding, train_bytes, validation_bytes, suffix_bytes):
    legacy = binding.development
    prepared, rules = secondary_development._preparation(
        legacy.preparation_bytes, suffix_bytes, legacy.pins
    )
    train = secondary_development._partition(
        train_bytes, "train", prepared["splits"]["train"], legacy.pins, rules
    )
    validation = secondary_development._partition(
        validation_bytes,
        "validation",
        prepared["splits"]["validation"],
        legacy.pins,
        rules,
    )
    _require(
        set(train.domains).isdisjoint(validation.domains), "development_domains_overlap"
    )
    return tuple(source_runner._json(line) for line in train_bytes.splitlines()), tuple(
        source_runner._json(line) for line in validation_bytes.splitlines()
    )


def _vector_hash(values):
    return sha256(secondary_tabular._json_bytes(list(values))).hexdigest()


def _fit_input(binding, train, validation):
    return {
        "schema_version": 1,
        "model_kind": "random_forest",
        "seed": 42,
        "input_hashes": asdict(binding.development.pins),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "training_record_ids_sha256": _vector_hash(row["record_id"] for row in train),
        "training_labels_sha256": _vector_hash(row["is_phishing"] for row in train),
        "validation_record_ids_sha256": _vector_hash(
            row["record_id"] for row in validation
        ),
        "validation_labels_sha256": _vector_hash(
            row["is_phishing"] for row in validation
        ),
        "label_digest_encoding": "canonical_ASCII_JSON_integer_list_terminal_newline",
    }


def _digest(value):
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_fit_input(value, binding, prepared, ids, labels):
    fixed = {
        "schema_version": 1,
        "model_kind": "random_forest",
        "seed": 42,
        "input_hashes": asdict(binding.development.pins),
        "training_rows": prepared["splits"]["train"]["row_count"],
        "validation_rows": len(ids),
        "validation_record_ids_sha256": _vector_hash(ids),
        "validation_labels_sha256": _vector_hash(labels),
        "label_digest_encoding": "canonical_ASCII_JSON_integer_list_terminal_newline",
    }
    training_digests = {"training_record_ids_sha256", "training_labels_sha256"}
    _require(
        type(value) is dict
        and set(value) == set(fixed) | training_digests
        and completion._same({key: value[key] for key in fixed}, fixed)
        and all(_digest(value[key]) for key in training_digests),
        "fit_input_mismatch",
    )


def _verify_process(value, parent, stage, observed_exit):
    fixed = {
        "schema_version": 1,
        "stage": stage,
        "status": "worker_exited",
        "exit_code": observed_exit,
        "root_reservation_sha256": parent.reservation_sha256,
    }
    digests = {"stdout_sha256", "stderr_sha256"}
    _require(
        type(value) is dict
        and set(value) == set(fixed) | digests
        and completion._same({key: value[key] for key in fixed}, fixed)
        and all(_digest(value[key]) for key in digests),
        "process_observation_mismatch",
    )


def _rf(binding, attempt, inputs):
    with secondary_tabular._numerical_runtime():
        train, validation = _validated_rows(binding, *inputs)
    _record(
        attempt,
        "fit-input.json",
        secondary_tabular._json_bytes(_fit_input(binding, train, validation)),
    )
    result = secondary_tabular.fit_random_forest_v2(
        tuple(row["raw_url"] for row in train),
        tuple(row["is_phishing"] for row in train),
        tuple(row["raw_url"] for row in validation),
        tuple(row["is_phishing"] for row in validation),
        checkpoint=lambda content: _record(attempt, "fit-state.json", content),
    )
    labels = [row["is_phishing"] for row in validation]
    with secondary_tabular._numerical_runtime():
        metrics = {
            "average_precision": float(
                secondary_metrics.average_precision_score(
                    labels, result.validation_scores
                )
            ),
            "roc_auc": float(
                secondary_metrics.roc_auc_score(labels, result.validation_scores)
            ),
        }
    public = {
        "model_kind": "random_forest",
        "analysis_role": "descriptive_secondary_not_primary",
        "seed": 42,
        "row_count": len(validation),
        "class_counts": {str(label): labels.count(label) for label in (0, 1)},
        "validation_threshold": result.validation_threshold,
        "scoring_audit": result.scoring_audit,
        "score_metrics": metrics,
    }
    private = {
        "model.json": result.artifact_bytes,
        "validation-predictions.jsonl": b"".join(
            secondary_tabular._json_bytes(
                {
                    "record_id": row["record_id"],
                    "label": row["is_phishing"],
                    "probability": score,
                }
            )
            for row, score in zip(validation, result.validation_scores)
        ),
        "threshold.json": secondary_tabular._json_bytes(result.validation_threshold),
        "scoring-audit.json": secondary_tabular._json_bytes(result.scoring_audit),
    }
    return private, public


def _failure(attempt, stage, error):
    check = getattr(error, "check_id", None)
    check = check if type(check) is str and check in _CHECKS else stage
    check = check if check in _CHECKS else "verification"
    symbol = _failure_symbol(error)
    _record(
        attempt,
        "failure-details.json",
        _json_bytes(
            {
                "schema_version": 1,
                "stage": stage,
                "check_id": check,
                "error_type": symbol,
                "reservation_sha256": attempt.reservation_sha256,
            }
        ),
    )
    receipt.record_failure(attempt, stage=stage, error_type=symbol)


def run_correction_worker(
    root, *, expected_revision, expected_profile_sha256, paths, stage
):
    binding = bind_correction(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _run_worker(binding, paths, stage)


def _run_worker(binding, paths, stage):
    _require(stage in STAGES, "invalid_stage")
    recheck_correction(binding)
    parent = _root_attempt(binding, paths.attempt)
    # A direct RF worker invocation still requires the saved successful audit.
    if stage == "random_forest":
        process = source_runner._json(
            source_runner._read_file_once(
                parent.directory / "retained_audit-process.json"
            )
        )
        _require(
            type(process.get("exit_code")) is int and process["exit_code"] == 0,
            "audit_worker_not_successful",
        )
        _verify_stage(binding, paths, parent, "retained_audit", observed_exit=0)
    attempt = receipt.reserve_attempt(
        parent.directory / stage,
        identity={"root_reservation_sha256": parent.reservation_sha256, "stage": stage},
    )
    publishing = False
    try:
        inputs = _inputs(binding, paths)
        if stage == "retained_audit":
            from .development_audit import audit_retained

            train, validation, suffix = inputs
            result = audit_retained(
                binding.development,
                accounting_bytes=binding.accounting_bytes,
                original_attempt=paths.original_attempt,
                train_bytes=train,
                validation_bytes=validation,
                suffix_bytes=suffix,
            )
            private = {"audit.json": secondary_tabular._json_bytes(result)}
            auxiliary = {}
        else:
            private, result = _rf(binding, attempt, inputs)
            auxiliary = {
                name: sha256(
                    source_runner._read_file_once(attempt.directory / name)
                ).hexdigest()
                for name in ("fit-input.json", "fit-state.json")
            }
        recheck_correction(binding)
        summary = {
            "schema_version": 1,
            "status": "development_correction_stage_completed",
            "stage": stage,
            "root_reservation_sha256": parent.reservation_sha256,
            "reservation_sha256": attempt.reservation_sha256,
            "result": result,
            "private_sha256": {
                name: sha256(content).hexdigest() for name, content in private.items()
            },
            "auxiliary_sha256": auxiliary,
        }
        publishing = True
        return receipt.publish_completion(
            attempt,
            private_outputs=private,
            public_summary=summary,
            public_path=parent.directory / f"{stage}.json",
        )
    except Exception as error:
        if not publishing:
            _failure(attempt, stage, error)
        raise CorrectionError("worker_failed") from None


def _launch(parent, stage, command):
    _require(stage in STAGES, "invalid_stage")
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError:
        _record(
            parent,
            f"{stage}-process.json",
            _json_bytes(
                {
                    "schema_version": 1,
                    "stage": stage,
                    "status": "worker_launch_failed",
                    "exit_code": None,
                    "root_reservation_sha256": parent.reservation_sha256,
                }
            ),
        )
        raise CorrectionError("worker_launch_failed") from None
    _record(
        parent,
        f"{stage}-process.json",
        _json_bytes(
            {
                "schema_version": 1,
                "stage": stage,
                "status": "worker_exited",
                "exit_code": result.returncode,
                "root_reservation_sha256": parent.reservation_sha256,
                "stdout_sha256": sha256(result.stdout).hexdigest(),
                "stderr_sha256": sha256(result.stderr).hexdigest(),
            }
        ),
    )
    return result.returncode


@contextmanager
def _stage_snapshot(parent, stage):
    path = parent.directory / stage
    private_names = {"audit.json"} if stage == "retained_audit" else _TABULAR
    auxiliary = (
        set() if stage == "retained_audit" else {"fit-input.json", "fit-state.json"}
    )
    with ExitStack() as stack:
        directories = [
            (
                stack.enter_context(receipt._directory(path)),
                _RECEIPTS | {"evidence"} | auxiliary,
            ),
            (stack.enter_context(receipt._directory(path / "evidence")), private_names),
        ]
        marker_parent = stack.enter_context(receipt._directory(parent.directory))
        files = [
            (marker_parent, f"{stage}.json"),
            (marker_parent, f"{stage}-process.json"),
        ]
        for directory, names in directories:
            completion._directory_contents(directory, names)
            files.extend((directory, name) for name in sorted(names - {"evidence"}))
        states = []
        contents = {}
        for directory, name in files:
            metadata = receipt._entry(directory, name)
            _require(metadata is not None, "missing_stage_output")
            state = source_runner._file_state(metadata)
            contents[directory.path / name] = source_runner._read_file_once(
                directory.path / name, expected_state=state
            )
            states.append((directory, name, state))
        yield contents
        for directory, names in directories:
            completion._directory_contents(directory, names)
        marker_parent.check()
        for directory, name, state in states:
            metadata = receipt._entry(directory, name)
            _require(
                metadata is not None and source_runner._file_state(metadata) == state,
                "stage_output_changed",
            )


def _verify_stage(binding, paths, parent, stage, *, observed_exit):
    _require(
        type(observed_exit) is int and observed_exit == 0, "worker_exit_not_successful"
    )
    path = parent.directory / stage
    private_names = {"audit.json"} if stage == "retained_audit" else _TABULAR
    with _stage_snapshot(parent, stage) as contents:
        process = source_runner._json(
            contents[parent.directory / f"{stage}-process.json"]
        )
        _verify_process(process, parent, stage, observed_exit)
        reservation_hash, summary, hashes = completion._receipt(
            contents,
            path,
            parent.directory / f"{stage}.json",
            {"root_reservation_sha256": parent.reservation_sha256, "stage": stage},
            private_names,
        )
        expected_fields = {
            "schema_version",
            "status",
            "stage",
            "root_reservation_sha256",
            "reservation_sha256",
            "result",
            "private_sha256",
            "auxiliary_sha256",
        }
        _require(
            type(summary) is dict
            and set(summary) == expected_fields
            and type(summary["schema_version"]) is int
            and summary["schema_version"] == 1
            and summary["status"] == "development_correction_stage_completed"
            and summary["stage"] == stage
            and summary["root_reservation_sha256"] == parent.reservation_sha256
            and summary["reservation_sha256"] == reservation_hash
            and summary["private_sha256"] == hashes,
            "stage_summary_mismatch",
        )
        if stage == "retained_audit":
            from .development_audit import validate_audit_summary

            _require(summary["auxiliary_sha256"] == {}, "unexpected_audit_auxiliary")
            _require(
                contents[path / "evidence/audit.json"]
                == secondary_tabular._json_bytes(summary["result"]),
                "audit_payload_mismatch",
            )
            validate_audit_summary(
                summary["result"],
                binding=binding.development,
                accounting_bytes=binding.accounting_bytes,
                original_attempt=paths.original_attempt,
            )
        else:
            auxiliary = {
                name: sha256(contents[path / name]).hexdigest()
                for name in ("fit-input.json", "fit-state.json")
            }
            _require(
                summary["auxiliary_sha256"] == auxiliary, "checkpoint_hash_mismatch"
            )
            model = contents[path / "evidence/model.json"]
            secondary_tabular.validate_fit_checkpoint_bytes(
                contents[path / "fit-state.json"], model_bytes=model
            )
            artifact = source_runner._json(model)
            _require(
                artifact["contract_id"] == "secondary-development-correction-v1"
                and artifact["method_version"] == "secondary-rf-v2",
                "wrong_rf_method",
            )
            prepared = source_runner._json(binding.development.preparation_bytes)
            with secondary_tabular._numerical_runtime():
                ids, labels = completion._tabular(
                    contents,
                    path / "evidence",
                    "random_forest",
                    summary["result"],
                    binding.development,
                    prepared,
                    expected_method_version="secondary-rf-v2",
                )
            fit_input = completion._private_json(contents[path / "fit-input.json"])
            _verify_fit_input(fit_input, binding, prepared, ids, labels)
        recheck_correction(binding)
        return summary


def _command(binding, paths, stage):
    command = [
        sys.executable,
        str(binding.development.base.root / "scripts/run_secondary_correction.py"),
        "--repo-root",
        str(binding.development.base.root),
        "--expected-revision",
        binding.development.base.revision,
        "--expected-profile-sha256",
        binding.profile_sha256,
        "--worker",
        stage,
    ]
    for name in (
        "train",
        "validation",
        "suffix_rules",
        "original_attempt",
        "attempt",
        "public_summary",
    ):
        command.extend(("--" + name.replace("_", "-"), str(getattr(paths, name))))
    return command


def run_correction(root, *, expected_revision, expected_profile_sha256, paths):
    binding = bind_correction(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _supervise(binding, paths)


def _supervise(binding, paths):
    recheck_correction(binding)
    _outside_outputs(binding, paths)
    parent = receipt.reserve_attempt(paths.attempt, identity=_identity(binding))
    stage, publishing = "reservation", False
    accepted = []
    try:
        for stage in STAGES:
            code = _launch(parent, stage, _command(binding, paths, stage))
            accepted.append(
                _verify_stage(binding, paths, parent, stage, observed_exit=code)
            )
        stage = "final_binding"
        recheck_correction(binding)
        with ExitStack() as snapshots:
            for member in STAGES:
                snapshots.enter_context(_stage_snapshot(parent, member))
            for member, previous in zip(STAGES, accepted):
                current = _verify_stage(binding, paths, parent, member, observed_exit=0)
                _require(completion._same(current, previous), "accepted_stage_changed")
        summary = {
            "schema_version": 1,
            "status": "completed_secondary_development_correction",
            "analysis_stage": "development_validation_only",
            "protected_evaluation_authorized": False,
            "original_aggregate_accepted": False,
            "new_fits": 1,
            "retries": 0,
            "execution": {
                **_identity(binding),
                "reservation_sha256": parent.reservation_sha256,
            },
            "worker_exit_codes": {name: 0 for name in STAGES},
            "retained_audit": accepted[0],
            "corrected_random_forest": accepted[1],
            "verification_scope": "observed_workers_receipts_hashes_exact_saved_model_rescoring_in_audit_and_saved_RF_metrics_checkpoint_validation_no_independent_RF_refit",
        }
        stage = "publication"
        publishing = True
        receipt.publish_completion(
            parent,
            private_outputs={
                "stage-summaries.json": secondary_tabular._json_bytes(accepted)
            },
            public_summary=summary,
            public_path=paths.public_summary,
        )
        return summary
    except Exception as error:
        if not publishing:
            _failure(parent, stage, error)
        raise CorrectionError("correction_stopped") from None
