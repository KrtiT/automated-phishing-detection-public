"""Read back new diagnostic evidence; never fit, predict, or certify old fits.

Internal consistency is not a signature or proof of producer exit. The caller
must independently observe successful execution and own access authorization.
"""

from dataclasses import asdict
from hashlib import sha256

import numpy as np

from . import audited_permutation_member as member
from . import (
    baselines,
    execution_receipt,
    secondary_development,
    secondary_metrics,
    secondary_tabular,
)
from .source_runner import _read_file_once

_PRIVATE = frozenset(
    {
        "model.json",
        "validation-predictions.jsonl",
        "threshold.json",
        "scoring-audit.json",
        "fit-linkage.json",
    }
)
_RECEIPTS = frozenset(
    {"fit-call-prepared.json", "fit-call-returned.json", "fit-state.json"}
)


def _require(condition):
    if not condition:
        raise member.DiagnosticError("diagnostic_evidence_mismatch")


def _require_equal(actual, expected):
    _require(
        secondary_tabular._json_bytes(actual) == secondary_tabular._json_bytes(expected)
    )


def _decode(content, *, trailing_newline=True):
    value = secondary_development._json(content)
    encoder = (
        secondary_tabular._json_bytes
        if trailing_newline
        else lambda value: execution_receipt._json_bytes(value, "evidence")
    )
    _require(encoder(value) == content)
    return value


def _digest(array, dtype):
    return sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def _matrix(matrix):
    return {
        "shape": list(matrix.shape),
        "dtype": "<f8",
        "order": "C",
        "sha256": _digest(matrix, "<f8"),
    }


def _verify_prepared(content, checkpoint, train, seed):
    receipt = _decode(content)
    labels = np.asarray([row["is_phishing"] for row in train], dtype=np.int64)
    indices = np.random.Generator(np.random.PCG64(seed)).permutation(
        np.arange(len(labels), dtype=np.int64)
    )
    expected_labels = labels[indices]
    features = secondary_tabular._features(
        tuple(row["raw_url"] for row in train), "permutation"
    )
    scaled = features.copy(order="C")
    scaled -= np.asarray(checkpoint["state"]["scaler"]["mean"], dtype=np.float64)
    scaled /= np.asarray(checkpoint["state"]["scaler"]["scale"], dtype=np.float64)
    expected = {
        "schema_version": 1,
        "analysis_stage": "post_review_new_fit_diagnostic",
        "event": "fit_call_prepared",
        "ordered_record_ids": [row["record_id"] for row in train],
        "consumed_labels": expected_labels.tolist(),
        "consumed_labels_sha256": _digest(expected_labels, "<i8"),
        "original_labels_sha256": _digest(labels, "<i8"),
        "label_class_counts": {
            str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)
        },
        "permutation_seed": seed,
        "bit_generator": "PCG64",
        "permutation_indices_sha256": _digest(indices, "<i8"),
        "source_features": _matrix(features),
        "scaled_features": _matrix(scaled),
        "parameters": secondary_tabular._parameters("permutation")["classifier"],
    }
    _require(content == secondary_tabular._json_bytes(expected))
    _require_equal(receipt["parameters"], checkpoint["parameters"]["classifier"])
    return int(np.count_nonzero(expected_labels != labels))


def _verify_predictions(private, validation, summary):
    predictions = [
        _decode(line)
        for line in private["validation-predictions.jsonl"].splitlines(keepends=True)
    ]
    _require(len(predictions) == len(validation))
    for prediction, row in zip(predictions, validation):
        _require(set(prediction) == {"record_id", "label", "probability"})
        _require(prediction["record_id"] == row["record_id"])
        _require(
            type(prediction["label"]) is int
            and prediction["label"] == row["is_phishing"]
        )
        _require(
            type(prediction["probability"]) is float
            and 0 <= prediction["probability"] <= 1
        )
    probabilities = np.asarray(
        [row["probability"] for row in predictions], dtype=np.float64
    )
    labels = np.asarray([row["is_phishing"] for row in validation], dtype=np.int64)
    threshold = baselines.select_validation_threshold(probabilities, labels)
    audit = _decode(private["scoring-audit.json"])
    _require(audit.get("portable_exact_parity") is True)
    _require(audit.get("threshold_role") == "secondary_descriptive_operating_point")
    expected = {
        "model_kind": "permutation",
        "analysis_role": "descriptive_secondary_not_primary",
        "seed": summary["seed"],
        "row_count": len(validation),
        "class_counts": {
            str(label): int(np.count_nonzero(labels == label)) for label in (0, 1)
        },
        "validation_threshold": threshold,
        "scoring_audit": audit,
        "score_metrics": {
            "average_precision": float(
                secondary_metrics.average_precision_score(labels, probabilities)
            ),
            "roc_auc": float(secondary_metrics.roc_auc_score(labels, probabilities)),
        },
        "source_identity_verified": True,
    }
    _require(private["threshold.json"] == secondary_tabular._json_bytes(threshold))
    _require(
        secondary_tabular._json_bytes(summary)
        == secondary_tabular._json_bytes(expected)
    )


def verify_completed_member(
    *,
    seed,
    attempt,
    public_path,
    pins,
    preparation_summary,
    suffix_rules,
    train_content,
    validation_content,
):
    try:
        _require(type(seed) is int and seed in range(42, 47))
        with execution_receipt._attempt_directory(attempt) as directory:
            _require(not public_path.absolute().is_relative_to(directory.path))
            public = _read_file_once(public_path)
            summary = _decode(public, trailing_newline=False)
            outcome = _decode(
                _read_file_once(directory.path / "outcome.json"), trailing_newline=False
            )
            claim = _decode(
                _read_file_once(directory.path / "finalize.claim"),
                trailing_newline=False,
            )
            started = _decode(_read_file_once(directory.path / "member-started.json"))
            receipts = {
                name: _read_file_once(directory.path / name) for name in _RECEIPTS
            }
            private = {
                name: _read_file_once(directory.path / "evidence" / name)
                for name in _PRIVATE
            }
            directory.check()
        _require_equal(
            claim,
            {
                "schema_version": 1,
                "reservation_sha256": attempt.reservation_sha256,
                "operation": "completion",
            },
        )
        _require_equal(started, {"seed": seed, "input_pins": asdict(pins)})
        private_hashes = {
            name: sha256(content).hexdigest() for name, content in private.items()
        }
        receipt_hashes = {
            name: sha256(content).hexdigest() for name, content in receipts.items()
        }
        _require_equal(
            outcome,
            {
                "schema_version": 1,
                "status": "completion_prepared",
                "reservation_sha256": attempt.reservation_sha256,
                "public_summary_sha256": sha256(public).hexdigest(),
                "private_sha256": private_hashes,
            },
        )
        linkage = {
            "analysis_stage": "post_review_new_fit_diagnostic",
            "historical_fit_verified": False,
            "seed": seed,
            "partition_sha256": {
                "train": pins.train_sha256,
                "validation": pins.validation_sha256,
            },
            "preparation_summary_sha256": pins.preparation_summary_sha256,
            "suffix_rules_sha256": pins.suffix_rules_sha256,
            "receipt_sha256": receipt_hashes,
            "model_sha256": sha256(private["model.json"]).hexdigest(),
        }
        _require(private["fit-linkage.json"] == secondary_tabular._json_bytes(linkage))
        _require_equal(
            summary,
            {
                "schema_version": 1,
                "status": "audited_permutation_member_completed",
                "analysis_stage": "post_review_new_fit_diagnostic",
                "historical_fit_verified": False,
                "protected_evaluation_authorized": False,
                "reservation_sha256": attempt.reservation_sha256,
                "result": summary["result"],
                "private_sha256": private_hashes,
                "receipt_sha256": receipt_hashes,
                "input_pins": asdict(pins),
            },
        )
        _require_equal(
            _decode(receipts["fit-call-returned.json"]),
            {
                "schema_version": 1,
                "event": "fit_call_returned",
                "prepared_receipt_sha256": receipt_hashes["fit-call-prepared.json"],
            },
        )
        with secondary_tabular._numerical_runtime():
            train, validation = member._validated_rows(
                train_content,
                validation_content,
                preparation_summary,
                suffix_rules,
                pins,
            )
            checkpoint = secondary_tabular.validate_fit_checkpoint_bytes(
                receipts["fit-state.json"], model_bytes=private["model.json"]
            )
            _require(
                checkpoint["seed"] == seed and checkpoint["model_kind"] == "permutation"
            )
            _require(checkpoint["training_row_count"] == len(train))
            changes = _verify_prepared(
                receipts["fit-call-prepared.json"], checkpoint, train, seed
            )
            _require(summary["result"]["seed"] == seed)
            _verify_predictions(private, validation, summary["result"])
        return {
            "status": "evidence_consistent",
            "seed": seed,
            "training_rows": len(train),
            "validation_rows": len(validation),
            "permuted_label_changes": changes,
            "model_sha256": linkage["model_sha256"],
            "public_summary_sha256": sha256(public).hexdigest(),
            "historical_fit_verified": False,
        }
    except Exception as error:
        raise member.DiagnosticError(
            "diagnostic_evidence_verification_failed"
        ) from error
