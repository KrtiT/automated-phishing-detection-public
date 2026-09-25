"""Independent acceptance of an observed successful internal producer exit.

This verifier authenticates publication/source links and reconstructs scientific
outputs from retained bytes, including monitor and routing replay. An output
marker is never evidence of a successful subprocess exit. The
caller must obtain that exit status from the actual producer it supervised.
"""

from __future__ import annotations

import math
import os
import types
from dataclasses import asdict, fields, is_dataclass
from hashlib import sha256
from typing import get_args, get_origin, get_type_hints

from . import (
    execution_receipt,
    fixed_cascade,
    hypothesis_evaluation,
    secondary_metrics,
    source_runner,
)
from .bound_secondary import SecondaryInferenceCounts
from .evaluation_producer import ManifestOutcome, _json_bytes, _manifest_summary
from .saved_evidence import reconstruct_internal_evidence
from .saved_metrics import DetectionMetrics
from .selective_inference import InferenceCounts
from .source_checkpoint_verification import verify_source_checkpoints
from .source_checkpoints import CHECKPOINT_NAMES

_PRIVATE_NAMES = frozenset(
    {
        "predictions.jsonl",
        "manifests.json",
        "bindings.json",
        "secondary.json",
        "routing.json",
    }
)
_ATTEMPT_NAMES = frozenset(
    {"reservation.json", "finalize.claim", "outcome.json", "evidence", "checkpoints"}
)
_PUBLIC_NAMES = frozenset(
    {
        "schema_version",
        "status",
        "protected_evaluation_authorized",
        "source_binding",
        "row_count",
        "domain_count",
        "class_counts",
        "offline_inference_counts",
        "offline_secondary_inference_counts",
        "manifests",
        "primary",
        "private_sha256",
        "execution",
        "secondary",
        "source_reconstruction",
        "checkpoint_sha256",
    }
)
_MODELS = frozenset({"length_only", "logistic_l1", "transformer", "cascade"})
_SECONDARY_MODELS = frozenset(
    {
        *_MODELS,
        "tabular.formatting",
        "tabular.permutation_42",
        "tabular.permutation_43",
        "tabular.permutation_44",
        "tabular.permutation_45",
        "tabular.permutation_46",
        "tabular.random_forest",
        "seed_42.transformer",
        "seed_42.cascade",
        "seed_43.transformer",
        "seed_43.cascade",
        "seed_44.transformer",
        "seed_44.cascade",
        "seed_45.transformer",
        "seed_45.cascade",
        "seed_46.transformer",
        "seed_46.cascade",
    }
)


class CompletionVerificationError(ValueError):
    """Symbolic completion rejection, without private values or exception text."""


def _require(condition, symbol):
    if not condition:
        raise CompletionVerificationError(symbol)


def _keys(value, expected):
    return type(value) is dict and set(value) == set(expected)


def _count(value):
    return type(value) is int and value >= 0


def _number(value):
    return type(value) in (float, int) and math.isfinite(value)


def _digest(value):
    return type(value) is str and execution_receipt._SHA256.fullmatch(value) is not None


def _shape(value, annotation):
    """Check existing dataclass JSON shapes without constructing trusted objects."""
    origin, arguments = get_origin(annotation), get_args(annotation)
    if origin is types.UnionType:
        return any(_shape(value, item) for item in arguments)
    if annotation is type(None):
        return value is None
    if annotation is float:
        return _number(value)
    if annotation is int:
        return _count(value)
    if annotation in (str, bool):
        return type(value) is annotation
    if is_dataclass(annotation):
        if not _keys(value, (field.name for field in fields(annotation))):
            return False
        return all(
            _shape(value[name], member)
            for name, member in get_type_hints(annotation).items()
        )
    if origin is dict:
        return type(value) is dict and all(
            _shape(key, arguments[0]) and _shape(item, arguments[1])
            for key, item in value.items()
        )
    if origin is tuple:
        if type(value) is not list:
            return False
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return all(_shape(item, arguments[0]) for item in value)
        return len(value) == len(arguments) and all(
            _shape(item, member) for item, member in zip(value, arguments, strict=True)
        )
    return False


def _canonical(value):
    return execution_receipt._json_bytes(value, "completion_record")


def _same(first, second):
    return _canonical(first) == _canonical(second)


def _expected_identity(binding, source):
    pins = dict(binding.source_hashes)
    return {
        "kind": "internal_evaluation",
        "source_interface": "original_csv_reconstruction_v1",
        "revision": binding.revision,
        "execution_contract_sha256": binding.contract_sha256,
        "source_spec_sha256": pins["data/sources.json"],
        "preparation_summary_sha256": pins["reports/phiusiil-preparation-summary.json"],
        "runtime_sha256": sha256(binding.runtime_json.encode()).hexdigest(),
        "partition_sha256": source["expected_sha256"],
        "source_csv_sha256": source["source_csv_sha256"],
        "suffix_rules_sha256": source["suffix_rules_sha256"],
    }


def _detection_counts(value, classes):
    _require(_shape(value, DetectionMetrics), "invalid_detection_schema")
    tp, fn, fp, tn = (
        value[name]
        for name in (
            "true_positives",
            "false_negatives",
            "false_positives",
            "true_negatives",
        )
    )
    _require(all(_count(item) for item in (tp, fn, fp, tn)), "invalid_detection_counts")
    _require(
        tp + fn == classes["1"] and fp + tn == classes["0"], "class_count_mismatch"
    )
    for rate, numerator, denominator in (
        (value["recall"], tp, classes["1"]),
        (value["fpr"], fp, classes["0"]),
    ):
        _require(
            rate["numerator"] == numerator and rate["denominator"] == denominator,
            "rate_count_mismatch",
        )
        if denominator:
            _require(
                rate["status"] == "estimated"
                and rate["estimate"] == numerator / denominator
                and _number(rate["upper_95"])
                and 0 <= rate["upper_95"] <= 1,
                "invalid_rate_estimate",
            )
        else:
            _require(
                rate["status"] == "not_estimable"
                and rate["estimate"] is None
                and rate["upper_95"] is None,
                "invalid_unavailable_rate",
            )


def _primary(value, classes):
    _require(
        _shape(value, hypothesis_evaluation.PrimaryEvaluation), "invalid_primary_schema"
    )
    _require(
        _keys(value["metrics"], (f"internal.{model}" for model in _MODELS)),
        "invalid_primary_models",
    )
    contrasts = {
        f"{role}.{candidate}_minus_{reference}"
        for role, candidate, reference in hypothesis_evaluation.CONTRASTS
    }
    _require(_keys(value["contrasts"], contrasts), "invalid_primary_contrasts")
    _require(
        _keys(value["hypotheses"], {"H1", "H2", "H3"}), "invalid_primary_hypotheses"
    )
    for name, contrast in value["contrasts"].items():
        _require(
            (contrast is None) == name.startswith("gold."),
            "invalid_contrast_availability",
        )
    for metric in value["metrics"].values():
        _detection_counts(metric, classes)


def _default_role(value, schema, field="analysis_role"):
    _require(
        value[field] == schema.__dataclass_fields__[field].default,
        "secondary_role_mismatch",
    )


def _secondary(value, public):
    _require(
        _keys(value, {"schema_version", "metrics", "mcnemar", "holm"}),
        "invalid_secondary_schema",
    )
    _require(
        type(value["schema_version"]) is int and value["schema_version"] == 2,
        "invalid_secondary_version",
    )
    _require(_keys(value["metrics"], _SECONDARY_MODELS), "invalid_secondary_models")
    family = secondary_metrics.ABLATION_FAMILY
    _require(_keys(value["mcnemar"], family), "invalid_mcnemar_family")
    for model, metric in value["metrics"].items():
        _require(
            _shape(metric, secondary_metrics.SecondaryMetrics),
            "invalid_secondary_metric_schema",
        )
        _default_role(metric, secondary_metrics.SecondaryMetrics)
        _default_role(metric["recall_at_fpr"], secondary_metrics.RecallAtFPR)
        _require(
            metric["row_count"] == public["row_count"]
            and metric["domain_count"] == public["domain_count"],
            "secondary_population_mismatch",
        )
        if model in _MODELS:
            _require(
                _same(
                    metric["counts"],
                    public["primary"]["metrics"][f"internal.{model}"],
                ),
                "secondary_primary_counts_mismatch",
            )
        else:
            _detection_counts(metric["counts"], public["class_counts"])
        bins = metric["calibration_bins"]
        _require(
            len(bins) == 10
            and [item["index"] for item in bins] == list(range(10))
            and all(
                _count(item["count"])
                and _count(item["positive_count"])
                and item["positive_count"] <= item["count"]
                for item in bins
            )
            and sum(item["count"] for item in bins) == public["row_count"]
            and sum(item["positive_count"] for item in bins)
            == public["class_counts"]["1"],
            "invalid_calibration_bin_counts",
        )
        _require(
            [item["prevalence"] for item in metric["prevalence_projections"]]
            == [0.001, 0.01, 0.05],
            "invalid_projection_family",
        )
        for projection in metric["prevalence_projections"]:
            _default_role(
                projection, secondary_metrics.PrevalenceProjection, "assumption"
            )
            _require(
                projection["reference_requests"] == 10000,
                "invalid_projection_reference_count",
            )
    for name in family:
        cell = value["mcnemar"][name]
        if name.startswith("external_"):
            _require(cell is None, "unexpected_external_mcnemar")
        else:
            _require(
                _shape(cell, secondary_metrics.McNemarResult), "invalid_mcnemar_schema"
            )
            _default_role(cell, secondary_metrics.McNemarResult)
            _require(
                cell["row_count"]
                == cell["positive_count"]
                == public["class_counts"]["1"]
                and cell["negative_count"] == 0,
                "invalid_mcnemar_population",
            )
            counts = [
                cell[key]
                for key in (
                    "both_correct",
                    "candidate_only_correct",
                    "reference_only_correct",
                    "both_incorrect",
                )
            ]
            _require(
                all(_count(item) for item in counts)
                and sum(counts) == cell["row_count"],
                "invalid_mcnemar_counts",
            )
    holm = value["holm"]
    _require(_shape(holm, secondary_metrics.HolmFamily), "invalid_holm_schema")
    _default_role(holm, secondary_metrics.HolmFamily)
    _require(
        holm["family_size"] == 4
        and holm["complete"] is False
        and [cell["test_id"] for cell in holm["cells"]] == list(family),
        "invalid_holm_family",
    )
    for cell in holm["cells"]:
        if cell["test_id"].startswith("external_"):
            unavailable = {"value": None, "reason": "missing_evidence"}
            _require(
                _same(cell["raw_pvalue"], unavailable)
                and _same(cell["adjusted_pvalue"], unavailable),
                "unexpected_external_holm_evidence",
            )


def _manifests(value):
    _require(_keys(value, {"10", "100", "500"}), "invalid_manifest_family")
    for prevalence, outcome in value.items():
        _require(type(outcome) is dict, "invalid_manifest_schema")
        if outcome.get("status") == "prepared":
            _require(
                _keys(
                    outcome,
                    {
                        "status",
                        "sha256",
                        "measured_count",
                        "warmup_count",
                        "prevalence_basis_points",
                    },
                )
                and _digest(outcome["sha256"])
                and all(
                    _count(outcome[key])
                    for key in (
                        "measured_count",
                        "warmup_count",
                        "prevalence_basis_points",
                    )
                )
                and outcome["prevalence_basis_points"] == int(prevalence),
                "invalid_prepared_manifest",
            )
            _require(
                outcome["measured_count"] == 10000 and outcome["warmup_count"] == 1000,
                "invalid_manifest_counts",
            )
        else:
            _require(
                _shape(outcome, ManifestOutcome)
                and outcome["status"] == "insufficient_capacity"
                and outcome["manifest"] is None
                and type(outcome["insufficient_label"]) is int
                and outcome["insufficient_label"] in (0, 1)
                and _count(outcome["required"])
                and _count(outcome["available"])
                and outcome["available"] < outcome["required"],
                "invalid_unavailable_manifest",
            )


def _public(public, binding, source, identity, reservation_hash):
    _require(_keys(public, _PUBLIC_NAMES), "invalid_public_schema")
    _require(
        type(public["schema_version"]) is int
        and public["schema_version"] == 4
        and public["status"] == "internal_evidence_published"
        and public["source_binding"] == "authenticated_public_preparation"
        and type(public["protected_evaluation_authorized"]) is bool
        and public["protected_evaluation_authorized"]
        is binding.protected_evaluation_ready,
        "invalid_public_status",
    )
    _require(
        _same(
            public["execution"], {**identity, "reservation_sha256": reservation_hash}
        ),
        "public_execution_identity_mismatch",
    )
    classes = public["class_counts"]
    _require(
        _count(public["row_count"])
        and _count(public["domain_count"])
        and public["row_count"] == source["expected_row_count"]
        and public["domain_count"] == source["expected_domain_count"]
        and public["domain_count"] <= public["row_count"]
        and _keys(classes, {"0", "1"})
        and all(_count(item) for item in classes.values())
        and sum(classes.values()) == public["row_count"]
        and _same(classes, source["expected_class_counts"]),
        "invalid_public_population",
    )
    counts = public["offline_inference_counts"]
    _require(
        _shape(counts, InferenceCounts)
        and counts["failed_requests"] == 0
        and all(
            value == public["row_count"]
            for key, value in counts.items()
            if key != "failed_requests"
        ),
        "invalid_offline_inference_counts",
    )
    secondary_counts = public["offline_secondary_inference_counts"]
    _require(
        _shape(secondary_counts, SecondaryInferenceCounts)
        and secondary_counts["tabular_singleton_calls"]
        == [
            [name, public["row_count"]]
            for name in (
                "formatting",
                "permutation_42",
                "permutation_43",
                "permutation_44",
                "permutation_45",
                "permutation_46",
                "random_forest",
            )
        ]
        and secondary_counts["transformer_singleton_calls"]
        == [
            [42, 0],
            *[[seed, public["row_count"]] for seed in (43, 44, 45, 46)],
        ]
        and secondary_counts["reused_primary_transformer_scores"]
        == public["row_count"],
        "invalid_secondary_inference_counts",
    )
    _require(
        _keys(public["private_sha256"], _PRIVATE_NAMES)
        and all(_digest(value) for value in public["private_sha256"].values()),
        "invalid_private_hash_schema",
    )
    _manifests(public["manifests"])
    _primary(public["primary"], classes)
    _secondary(public["secondary"], public)


def _private_bindings(content, identity):
    value = source_runner._json(content)
    source_fields = {"partition_sha256", "source_csv_sha256", "suffix_rules_sha256"}
    _require(
        _keys(
            value,
            source_fields
            | {
                "schema_version",
                "artifact_hashes",
                "thresholds",
                "secondary",
                "gmm_audit",
                "replay_artifacts",
            },
        )
        and type(value["schema_version"]) is int
        and value["schema_version"] == 3,
        "invalid_private_bindings_schema",
    )
    _require(
        all(value[name] == identity[name] for name in source_fields),
        "private_source_binding_mismatch",
    )
    hashes = value["artifact_hashes"]
    _require(
        type(hashes) is dict
        and bool(hashes)
        and all(
            type(name) is str
            and execution_receipt._FILENAME.fullmatch(name)
            and _digest(digest)
            for name, digest in hashes.items()
        ),
        "invalid_artifact_hashes",
    )
    thresholds = value["thresholds"]
    _require(
        _keys(
            thresholds,
            {
                "length_only",
                "logistic_l1",
                "transformer",
                "half_width",
                "monitor_boundary",
            },
        )
        and all(_number(item) for item in thresholds.values())
        and thresholds["half_width"] >= 0,
        "invalid_threshold_schema",
    )
    for model in ("length_only", "logistic_l1", "transformer"):
        fixed_cascade._threshold(thresholds[model], "completion_threshold")


def _directory_contents(directory, expected):
    directory.check()
    _require(
        set(os.listdir(directory.descriptor)) == expected, "unexpected_output_files"
    )


def _output_snapshot(attempt, evidence, checkpoints, public_parent, public_name):
    records = [
        (attempt, name) for name in sorted(_ATTEMPT_NAMES - {"evidence", "checkpoints"})
    ]
    records += [(evidence, name) for name in sorted(_PRIVATE_NAMES)]
    records += [(checkpoints, name) for name in sorted(CHECKPOINT_NAMES)]
    records.append((public_parent, public_name))
    result = []
    for directory, name in records:
        state = execution_receipt._entry(directory, name)
        _require(state is not None, "missing_output_file")
        result.append((directory, name, source_runner._file_state(state)))
    return result


def _verify_outputs(binding, paths, source, source_buffers, identity):
    attempt_path = execution_receipt._absolute_path(paths.attempt)
    public_path = execution_receipt._absolute_path(paths.public_summary)
    _require(
        not attempt_path.is_relative_to(binding.root)
        and not public_path.is_relative_to(binding.root)
        and not public_path.is_relative_to(attempt_path),
        "invalid_completion_output_paths",
    )
    with (
        execution_receipt._directory(attempt_path) as attempt,
        execution_receipt._directory(attempt_path / "evidence") as evidence,
        execution_receipt._directory(attempt_path / "checkpoints") as checkpoints,
        execution_receipt._directory(public_path.parent) as public_parent,
    ):
        _directory_contents(attempt, _ATTEMPT_NAMES)
        _directory_contents(evidence, _PRIVATE_NAMES)
        _directory_contents(checkpoints, CHECKPOINT_NAMES)
        snapshot = _output_snapshot(
            attempt, evidence, checkpoints, public_parent, public_path.name
        )
        contents = {
            (directory.path / name): source_runner._read_file_once(
                directory.path / name, expected_state=initial
            )
            for directory, name, initial in snapshot
        }
        reservation = contents[attempt_path / "reservation.json"]
        _require(
            reservation
            == _canonical(
                {
                    "schema_version": 1,
                    "status": "reserved",
                    "directory": str(attempt_path),
                    "identity": identity,
                }
            ),
            "reservation_identity_mismatch",
        )
        reservation_hash = sha256(reservation).hexdigest()
        _require(
            contents[attempt_path / "finalize.claim"]
            == _canonical(
                {
                    "schema_version": 1,
                    "reservation_sha256": reservation_hash,
                    "operation": "completion",
                }
            ),
            "invalid_completion_claim",
        )
        public_bytes = contents[public_path]
        public = source_runner._json(public_bytes)
        _require(public_bytes == _canonical(public), "noncanonical_public_summary")
        _public(public, binding, source, identity, reservation_hash)
        hashes = {
            name: sha256(contents[evidence.path / name]).hexdigest()
            for name in sorted(_PRIVATE_NAMES)
        }
        _require(
            _same(hashes, public["private_sha256"]), "private_output_hash_mismatch"
        )
        _require(
            contents[attempt_path / "outcome.json"]
            == _canonical(
                {
                    "schema_version": 1,
                    "status": "completion_prepared",
                    "reservation_sha256": reservation_hash,
                    "public_summary_sha256": sha256(public_bytes).hexdigest(),
                    "private_sha256": hashes,
                }
            ),
            "invalid_completion_outcome",
        )
        _require(
            _same(
                source_runner._json(contents[evidence.path / "secondary.json"]),
                public["secondary"],
            ),
            "private_secondary_mismatch",
        )
        _private_bindings(contents[evidence.path / "bindings.json"], identity)
        verify_source_checkpoints(
            {name: contents[checkpoints.path / name] for name in CHECKPOINT_NAMES},
            public,
            source,
            source_runner._json(source_buffers[source_runner._PREPARATION]),
            identity,
            reservation_hash,
            contents[evidence.path / "predictions.jsonl"],
        )
        reconstructed = reconstruct_internal_evidence(
            contents[evidence.path / "predictions.jsonl"],
            contents[evidence.path / "manifests.json"],
            contents[evidence.path / "bindings.json"],
            contents[evidence.path / "routing.json"],
        )
        reconstructed_public = source_runner._json(
            _json_bytes(
                {
                    "row_count": reconstructed.row_count,
                    "domain_count": reconstructed.domain_count,
                    "class_counts": reconstructed.class_counts,
                    "offline_inference_counts": asdict(reconstructed.inference_counts),
                    "offline_secondary_inference_counts": asdict(
                        reconstructed.secondary_inference_counts
                    ),
                    "manifests": {
                        str(prevalence): _manifest_summary(outcome)
                        for prevalence, outcome in reconstructed.manifests.items()
                    },
                    "primary": asdict(reconstructed.primary),
                    "secondary": reconstructed.secondary,
                }
            )
        )
        _require(
            _same(
                reconstructed_public,
                {
                    "row_count": public["row_count"],
                    "domain_count": public["domain_count"],
                    "class_counts": public["class_counts"],
                    "offline_inference_counts": public["offline_inference_counts"],
                    "offline_secondary_inference_counts": public[
                        "offline_secondary_inference_counts"
                    ],
                    "manifests": public["manifests"],
                    "primary": public["primary"],
                    "secondary": public["secondary"],
                },
            ),
            "reconstructed_evidence_mismatch",
        )
        source_runner.recheck_binding(binding)
        _directory_contents(attempt, _ATTEMPT_NAMES)
        _directory_contents(evidence, _PRIVATE_NAMES)
        _directory_contents(checkpoints, CHECKPOINT_NAMES)
        public_parent.check()
        for directory, name, initial in snapshot:
            current = execution_receipt._entry(directory, name)
            _require(
                current is not None and source_runner._file_state(current) == initial,
                "output_changed_during_verification",
            )
        return public


def verify_internal_completion(binding, paths, *, producer_exit_code: int) -> dict:
    """Accept only linked, intact evidence after an externally observed zero exit.

    No producer input or model file is opened. Private evidence and checkpoints are read
    exactly once through the descriptor-checked reader, then retained in memory.
    Prediction identities, features, decisions, metrics, gates, and manifests are
    reconstructed from retained bytes without reopening source or model inputs.
    """
    _require(
        type(producer_exit_code) is int and producer_exit_code == 0,
        "producer_exit_not_successful",
    )
    try:
        source_runner.recheck_binding(binding)
        _require(type(paths) is source_runner.InternalRunPaths, "invalid_run_paths")
        source, source_buffers = source_runner._public_sources(binding)
        identity = _expected_identity(binding, source)
        return _verify_outputs(binding, paths, source, source_buffers, identity)
    except CompletionVerificationError:
        raise
    except Exception:
        raise CompletionVerificationError("completion_verification_failed") from None
