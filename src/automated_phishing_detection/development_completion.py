"""Verify a completed development attempt from retained output bytes only.

The caller supplies the actual supervised producer exit status. Receipts alone
are not evidence of a successful process. This verifies saved model structure,
member identities, predictions, operating points and drift trace arithmetic; it
does not refit models, rescore URLs or independently replicate the experiment.
"""

from __future__ import annotations

import math
import os
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256

import numpy as np

from . import (
    baselines,
    development_execution,
    development_runner,
    execution_receipt,
    gmm_monitor,
    protocol_preflight,
    secondary_development,
    secondary_drift,
    secondary_metrics,
    secondary_tabular,
    source_runner,
)

_STEPS = (
    "drift",
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)
_RECEIPTS = {"reservation.json", "finalize.claim", "outcome.json"}
_ROOT_FILES = {"bindings.json", "members.json"}
_DRIFT_FILES = {"training-reference.json", "validation-audit.json"}
_TABULAR_FILES = {
    "model.json",
    "validation-predictions.jsonl",
    "threshold.json",
    "scoring-audit.json",
}


class DevelopmentCompletionError(ValueError):
    """Symbolic rejection without private paths, values or exception messages."""


def _require(condition, symbol):
    if not condition:
        raise DevelopmentCompletionError(symbol)


def _keys(value, expected):
    return type(value) is dict and set(value) == set(expected)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value):
    return type(value) is int and value >= 0


def _canonical(value):
    return execution_receipt._json_bytes(value, "completion_record")


def _same(first, second):
    return _canonical(first) == _canonical(second)


def _private_json(content):
    value = source_runner._json(content)
    _require(
        content == secondary_tabular._json_bytes(value), "noncanonical_private_json"
    )
    return value


def _identity(binding):
    return {
        "kind": "secondary_development",
        "revision": binding.base.revision,
        "execution_contract_sha256": binding.base.contract_sha256,
        "development_profile_sha256": binding.profile_sha256,
        "methods_contract_sha256": binding.methods_sha256,
        "runtime_sha256": sha256(binding.base.runtime_json.encode()).hexdigest(),
        "pins": asdict(binding.pins),
        "ordered_steps": list(_STEPS),
    }


def _receipt(contents, path, marker, identity, private_names):
    reservation = contents[path / "reservation.json"]
    _require(
        reservation
        == _canonical(
            {
                "schema_version": 1,
                "status": "reserved",
                "directory": str(path),
                "identity": identity,
            }
        ),
        "reservation_identity_mismatch",
    )
    reservation_hash = sha256(reservation).hexdigest()
    _require(
        contents[path / "finalize.claim"]
        == _canonical(
            {
                "schema_version": 1,
                "reservation_sha256": reservation_hash,
                "operation": "completion",
            }
        ),
        "invalid_completion_claim",
    )
    summary_bytes = contents[marker]
    summary = source_runner._json(summary_bytes)
    _require(summary_bytes == _canonical(summary), "noncanonical_public_summary")
    hashes = {
        name: sha256(contents[path / "evidence" / name]).hexdigest()
        for name in sorted(private_names)
    }
    _require(
        contents[path / "outcome.json"]
        == _canonical(
            {
                "schema_version": 1,
                "status": "completion_prepared",
                "reservation_sha256": reservation_hash,
                "public_summary_sha256": sha256(summary_bytes).hexdigest(),
                "private_sha256": hashes,
            }
        ),
        "invalid_completion_outcome",
    )
    return reservation_hash, summary, hashes


def _ids(values, count, source_hash):
    _require(type(values) is list and len(values) == count, "invalid_record_count")
    for identity in values:
        baselines._validate_record_id(identity, source_hash, "retained_record_id")
    _require(all(a < b for a, b in zip(values, values[1:])), "invalid_record_order")


def _domains(values, count):
    _require(type(values) is list and len(values) == count, "invalid_domain_count")
    for domain in values:
        _require(
            type(domain) is str and protocol_preflight._ascii_domain(domain) == domain,
            "invalid_retained_domain",
        )


def _vector(values, count):
    return (
        type(values) is list
        and len(values) == count
        and all(_number(value) for value in values)
    )


def _tabular(
    contents,
    path,
    member,
    result,
    binding,
    prepared,
    *,
    expected_method_version="secondary-tabular-v1",
):
    _require(
        _keys(
            result,
            {
                "model_kind",
                "seed",
                "row_count",
                "class_counts",
                "validation_threshold",
                "scoring_audit",
                "analysis_role",
                "score_metrics",
            },
        ),
        "invalid_tabular_summary",
    )
    kind = "permutation" if member.startswith("permutation_") else member
    seed = int(member[-2:]) if kind == "permutation" else 42
    declaration = prepared["splits"]["validation"]
    _require(
        _same(
            {
                key: result[key]
                for key in ("model_kind", "seed", "row_count", "class_counts")
            },
            {
                "model_kind": kind,
                "seed": seed,
                "row_count": declaration["row_count"],
                "class_counts": declaration["class_counts"],
            },
        ),
        "invalid_tabular_identity_or_counts",
    )
    model_bytes = contents[path / "model.json"]
    secondary_tabular.load_secondary_model_bytes(model_bytes)
    model = source_runner._json(model_bytes)
    expected_contract = {
        "secondary-tabular-v1": "secondary-development-v1",
        "secondary-rf-v2": "secondary-development-correction-v1",
    }.get(expected_method_version)
    _require(
        expected_contract is not None
        and (expected_method_version != "secondary-rf-v2" or kind == "random_forest")
        and model["method_version"] == expected_method_version
        and model["contract_id"] == expected_contract,
        "model_method_mismatch",
    )
    _require(
        model["model_kind"] == kind
        and model["training_row_count"] == prepared["splits"]["train"]["row_count"],
        "model_member_mismatch",
    )
    _require(
        model["permutation"]
        == (
            {"bit_generator": "PCG64", "seed": seed} if kind == "permutation" else None
        ),
        "model_seed_mismatch",
    )
    predictions = []
    for line in contents[path / "validation-predictions.jsonl"].splitlines(
        keepends=True
    ):
        row = _private_json(line)
        _require(
            _keys(row, {"record_id", "label", "probability"}),
            "invalid_prediction_schema",
        )
        _require(
            type(row["label"]) is int and row["label"] in (0, 1),
            "invalid_prediction_label",
        )
        _require(
            _number(row["probability"]) and 0 <= row["probability"] <= 1,
            "invalid_prediction_probability",
        )
        predictions.append(row)
    identities = [row["record_id"] for row in predictions]
    _ids(identities, declaration["row_count"], binding.pins.source_csv_sha256)
    labels = [row["label"] for row in predictions]
    _require(
        {"0": labels.count(0), "1": labels.count(1)} == declaration["class_counts"],
        "prediction_class_count_mismatch",
    )
    threshold = _private_json(contents[path / "threshold.json"])
    _require(
        result["analysis_role"] == "descriptive_secondary_not_primary",
        "invalid_tabular_analysis_role",
    )
    scores = [row["probability"] for row in predictions]
    metrics = {
        "average_precision": float(
            secondary_metrics.average_precision_score(labels, scores)
        ),
        "roc_auc": float(secondary_metrics.roc_auc_score(labels, scores)),
    }
    _require(_same(result["score_metrics"], metrics), "score_metrics_mismatch")
    expected = baselines.select_validation_threshold(scores, labels)
    _require(
        _same(threshold, expected) and _same(result["validation_threshold"], expected),
        "validation_threshold_mismatch",
    )
    audit = _private_json(contents[path / "scoring-audit.json"])
    expected_fields = {
        "batch_size",
        "warning_records",
        "portable_exact_parity",
        "threshold_role",
    }
    if kind != "random_forest":
        expected_fields |= {
            "platform_identity",
            "max_absolute_decision_difference",
            "max_absolute_probability_difference",
        }
    _require(_keys(audit, expected_fields), "invalid_scoring_audit_schema")
    _require(
        _same(
            {
                key: audit[key]
                for key in (
                    "batch_size",
                    "warning_records",
                    "portable_exact_parity",
                    "threshold_role",
                )
            },
            {
                "batch_size": 1,
                "warning_records": [],
                "portable_exact_parity": True,
                "threshold_role": "secondary_descriptive_operating_point",
            },
        ),
        "invalid_scoring_audit",
    )
    if kind != "random_forest":
        _require(
            _same(audit["platform_identity"], baselines._platform_identity()),
            "scoring_platform_mismatch",
        )
        _require(
            _number(audit["max_absolute_decision_difference"])
            and audit["max_absolute_decision_difference"] >= 0,
            "invalid_decision_audit",
        )
        _require(
            _number(audit["max_absolute_probability_difference"])
            and 0 <= audit["max_absolute_probability_difference"] <= 2e-12,
            "invalid_probability_audit",
        )
    _require(_same(result["scoring_audit"], audit), "scoring_audit_mismatch")
    return identities, labels


def _training_reference(reference, binding, prepared):
    _require(
        _keys(
            reference,
            {
                "schema_version",
                "contract_id",
                "input_hashes",
                "training_record_ids",
                "training_domains",
                "scaler",
                "portable_state_sha256",
                "validation_declaration_sha256",
                "mmd",
                "psi",
            },
        ),
        "invalid_training_reference_schema",
    )
    _require(
        type(reference["schema_version"]) is int
        and reference["schema_version"] == 1
        and reference["contract_id"] == "secondary-development-v1",
        "invalid_training_reference_identity",
    )
    _require(
        _same(reference["input_hashes"], asdict(binding.pins)),
        "training_reference_pin_mismatch",
    )
    train = prepared["splits"]["train"]
    ids, domains = reference["training_record_ids"], reference["training_domains"]
    _ids(ids, train["row_count"], binding.pins.source_csv_sha256)
    _domains(domains, train["row_count"])
    _require(
        len(set(domains)) == train["domain_count"], "training_domain_count_mismatch"
    )
    scaler = reference["scaler"]
    _require(
        _keys(scaler, {"mean", "scale"})
        and _vector(scaler["mean"], 26)
        and _vector(scaler["scale"], 26)
        and all(value > 0 for value in scaler["scale"]),
        "invalid_saved_scaler",
    )
    _require(
        type(reference["portable_state_sha256"]) is str
        and execution_receipt._SHA256.fullmatch(reference["portable_state_sha256"])
        is not None,
        "invalid_portable_state_digest",
    )
    _require(
        reference["validation_declaration_sha256"]
        == sha256(
            gmm_monitor._canonical_json_bytes(prepared["splits"]["validation"])
        ).hexdigest(),
        "validation_declaration_mismatch",
    )
    _mmd_reference(reference["mmd"], ids, domains)
    _psi_reference(reference["psi"], train["row_count"])


def _mmd_reference(value, training_ids, training_domains):
    _require(
        _keys(
            value, {"values", "domains", "stable_ids", "bandwidth_squared", "reason"}
        ),
        "invalid_mmd_reference",
    )
    first = {}
    for identity, domain in zip(training_ids, training_domains, strict=True):
        first.setdefault(domain, identity)
    selected = sorted(
        first,
        key=lambda domain: (
            sha256(
                (secondary_drift._REFERENCE_PREFIX + domain).encode("ascii")
            ).digest(),
            domain,
        ),
    )[:256]
    _require(
        value["domains"] == selected
        and value["stable_ids"] == [first[domain] for domain in selected],
        "mmd_reference_membership_mismatch",
    )
    matrix = value["values"]
    _require(
        type(matrix) is list
        and len(matrix) == len(selected)
        and all(_vector(row, 26) for row in matrix),
        "invalid_mmd_reference_values",
    )
    if len(selected) < 256:
        expected_reason, bandwidth = "fewer_than_256_training_domains", None
    else:
        array = np.asarray(matrix, dtype=np.float64)
        distances = secondary_drift._squared_distances(array, array)[
            np.triu_indices(256, k=1)
        ]
        positive = distances[distances > 0]
        expected_reason = None if len(positive) else "no_positive_reference_distance"
        bandwidth = (
            float(np.quantile(positive, 0.5, method="linear"))
            if len(positive)
            else None
        )
    _require(
        _same(
            {"reason": value["reason"], "bandwidth": value["bandwidth_squared"]},
            {"reason": expected_reason, "bandwidth": bandwidth},
        ),
        "mmd_bandwidth_mismatch",
    )


def _psi_reference(value, training_count):
    _require(
        _keys(value, {"features", "training_row_count", "reason"})
        and type(value["training_row_count"]) is int
        and value["training_row_count"] == training_count
        and value["reason"] is None,
        "invalid_psi_reference",
    )
    features = value["features"]
    _require(type(features) is list and len(features) == 26, "invalid_psi_features")
    for feature in features:
        _require(
            _keys(
                feature,
                {
                    "internal_edges",
                    "constant",
                    "training_counts",
                    "training_proportions",
                },
            ),
            "invalid_psi_feature",
        )
        edges, constant, counts, proportions = (
            feature[name]
            for name in (
                "internal_edges",
                "constant",
                "training_counts",
                "training_proportions",
            )
        )
        _require(
            type(edges) is list
            and all(_number(edge) for edge in edges)
            and all(a < b for a, b in zip(edges, edges[1:])),
            "invalid_psi_edges",
        )
        _require(
            (constant is None and 1 <= len(edges) <= 9)
            or (_number(constant) and not edges),
            "invalid_psi_constant",
        )
        size = len(edges) + 1 if constant is None else 3
        _require(
            type(counts) is list
            and len(counts) == size
            and all(_count(count) for count in counts)
            and sum(counts) == training_count,
            "invalid_psi_training_counts",
        )
        if constant is not None:
            _require(counts == [0, training_count, 0], "invalid_psi_constant_counts")
        expected = secondary_drift._proportions(counts, training_count).tolist()
        _require(
            _vector(proportions, size) and proportions == expected,
            "invalid_psi_proportions",
        )


def _trace(value, ends, method, reference_reason):
    _require(
        _keys(value, {"window_end_positions", "scores", "feature_scores", "reason"}),
        "invalid_drift_trace",
    )
    reason = reference_reason if ends else "no_complete_256_row_window"
    _require(
        value["window_end_positions"] == ends
        and all(type(end) is int for end in value["window_end_positions"])
        and value["reason"] == reason,
        "invalid_drift_trace_windows",
    )
    size = 0 if reason else len(ends)
    _require(_vector(value["scores"], size), "invalid_drift_scores")
    features = value["feature_scores"]
    if method == "mmd" or reason:
        _require(features == [], "unexpected_feature_scores")
    else:
        _require(
            type(features) is list
            and len(features) == size
            and all(_vector(row, 26) for row in features)
            and all(score >= 0 for row in features for score in row)
            and value["scores"] == [max(row) for row in features],
            "invalid_psi_feature_scores",
        )
    return secondary_drift.DriftWindowScores(
        tuple(ends),
        tuple(value["scores"]),
        tuple(tuple(row) for row in features),
        reason,
    )


def _drift(contents, path, public, hashes, binding, prepared):
    _require(
        _keys(
            public,
            {
                "schema_version",
                "contract_id",
                "status",
                "analysis_stage",
                "analysis_role",
                "protected_evaluation_authorized",
                "input_binding",
                "input_hashes",
                "input_counts",
                "methods",
                "private_sha256",
            },
        ),
        "invalid_drift_public_schema",
    )
    reference_bytes = contents[path / "training-reference.json"]
    reference = _private_json(reference_bytes)
    _training_reference(reference, binding, prepared)
    audit = _private_json(contents[path / "validation-audit.json"])
    _require(
        _keys(
            audit,
            {
                "schema_version",
                "contract_id",
                "training_reference_sha256",
                "input_hashes",
                "window_length",
                "window_stride",
                "streams",
                "results",
            },
        ),
        "invalid_drift_audit_schema",
    )
    _require(
        _same(
            {
                name: audit[name]
                for name in (
                    "schema_version",
                    "contract_id",
                    "training_reference_sha256",
                    "input_hashes",
                    "window_length",
                    "window_stride",
                )
            },
            {
                "schema_version": 1,
                "contract_id": "secondary-development-v1",
                "training_reference_sha256": sha256(reference_bytes).hexdigest(),
                "input_hashes": asdict(binding.pins),
                "window_length": 256,
                "window_stride": 64,
            },
        ),
        "invalid_drift_audit_identity",
    )
    _require(
        _keys(audit["streams"], {"calibration", "audit"})
        and _keys(audit["results"], {"mmd", "psi"}),
        "invalid_drift_methods_or_streams",
    )
    validation = prepared["splits"]["validation"]
    ids, domains, seen = (
        [None] * validation["row_count"],
        [None] * validation["row_count"],
        set(),
    )
    traces, counts = {}, {}
    for name, stream in audit["streams"].items():
        _require(
            _keys(
                stream,
                {
                    "input_row_positions",
                    "record_ids",
                    "domains",
                    "window_end_positions",
                    "mmd",
                    "psi",
                },
            ),
            "invalid_drift_stream",
        )
        positions = stream["input_row_positions"]
        _require(
            type(positions) is list
            and all(type(i) is int and 0 <= i < len(ids) for i in positions)
            and all(a < b for a, b in zip(positions, positions[1:])),
            "invalid_stream_positions",
        )
        _ids(stream["record_ids"], len(positions), binding.pins.source_csv_sha256)
        _domains(stream["domains"], len(positions))
        for position, identity, domain in zip(
            positions, stream["record_ids"], stream["domains"], strict=True
        ):
            _require(position not in seen, "stream_overlap")
            ids[position], domains[position] = identity, domain
            seen.add(position)
        ends = list(range(256, len(positions) + 1, 64))
        _require(
            stream["window_end_positions"] == ends
            and all(type(end) is int for end in stream["window_end_positions"]),
            "invalid_stream_window_ends",
        )
        traces[name] = {
            method: _trace(stream[method], ends, method, reference[method]["reason"])
            for method in ("mmd", "psi")
        }
        counts[f"{name}_rows"] = len(positions)
    _require(len(seen) == len(ids), "missing_validation_positions")
    _ids(ids, len(ids), binding.pins.source_csv_sha256)
    _require(
        not set(ids).intersection(reference["training_record_ids"])
        and not set(domains).intersection(reference["training_domains"]),
        "training_validation_overlap",
    )
    _require(
        len(set(domains)) == validation["domain_count"],
        "validation_domain_count_mismatch",
    )
    allocation = gmm_monitor.allocate_validation_domains(domains)
    _require(
        all(
            list(indices) == audit["streams"][name]["input_row_positions"]
            for name, indices in allocation.items()
        ),
        "frozen_allocation_mismatch",
    )
    methods = {}
    for method in ("mmd", "psi"):
        # Recompute boundaries and strict alerts from saved scores, not from URLs.
        methods[method], expected = secondary_development._method_result(
            traces["calibration"][method],
            traces["audit"][method],
            reference[method]["reason"],
        )
        expected = source_runner._json(secondary_tabular._json_bytes(expected))
        _require(
            _same(audit["results"][method], expected),
            "drift_boundary_or_alert_mismatch",
        )
    available = sum(value["status"] == "estimated" for value in methods.values())
    expected_public = {
        "schema_version": 1,
        "contract_id": "secondary-development-v1",
        "status": "completed_development_validation"
        if available == 2
        else "partially_estimable"
        if available
        else "not_estimable",
        "analysis_stage": "development_validation_only",
        "analysis_role": "secondary_descriptive_only",
        "protected_evaluation_authorized": False,
        "input_binding": "caller_supplied_expected_pins_not_authorization",
        "input_hashes": asdict(binding.pins),
        "input_counts": {
            "training_rows": prepared["splits"]["train"]["row_count"],
            "training_domain_count": prepared["splits"]["train"]["domain_count"],
            "validation_rows": validation["row_count"],
            "validation_domain_count": validation["domain_count"],
            **counts,
        },
        "methods": methods,
        "private_sha256": hashes,
    }
    _require(_same(public, expected_public), "drift_public_summary_mismatch")
    return ids


def _verify_records(binding, paths, contents):
    identity = _identity(binding)
    root_hash, public, root_hashes = _receipt(
        contents, paths.attempt, paths.public_summary, identity, _ROOT_FILES
    )
    prepared = baselines._validate_preparation_summary(
        source_runner._json(binding.preparation_bytes)
    )
    members, previous, drift_ids = [], None, None
    for member in _STEPS:
        path, marker = paths.attempt / member, paths.attempt / f"{member}.json"
        reservation_hash, summary, hashes = _receipt(
            contents,
            path,
            marker,
            {"root_reservation_sha256": root_hash, "member": member},
            _DRIFT_FILES if member == "drift" else _TABULAR_FILES,
        )
        _require(
            _keys(
                summary,
                {
                    "schema_version",
                    "status",
                    "member",
                    "root_reservation_sha256",
                    "reservation_sha256",
                    "private_sha256",
                    "result",
                },
            ),
            "invalid_member_summary_schema",
        )
        _require(
            _same(
                {name: value for name, value in summary.items() if name != "result"},
                {
                    "schema_version": 1,
                    "status": "development_member_completed",
                    "member": member,
                    "root_reservation_sha256": root_hash,
                    "reservation_sha256": reservation_hash,
                    "private_sha256": hashes,
                },
            ),
            "member_summary_identity_mismatch",
        )
        if member == "drift":
            drift_ids = _drift(
                contents,
                path / "evidence",
                summary["result"],
                hashes,
                binding,
                prepared,
            )
        else:
            predictions = _tabular(
                contents,
                path / "evidence",
                member,
                summary["result"],
                binding,
                prepared,
            )
            _require(
                predictions[0] == drift_ids
                and (previous is None or predictions == previous),
                "validation_membership_or_label_mismatch",
            )
            previous = predictions
        members.append(
            {
                "member": member,
                "public_summary_sha256": sha256(contents[marker]).hexdigest(),
                "summary": summary,
            }
        )
    execution = {**identity, "reservation_sha256": root_hash}
    _require(
        contents[paths.attempt / "evidence/bindings.json"] == _canonical(execution),
        "root_bindings_mismatch",
    )
    _require(
        contents[paths.attempt / "evidence/members.json"]
        == _canonical({"schema_version": 1, "members": members}),
        "root_members_mismatch",
    )
    _require(
        _same(
            public,
            {
                "schema_version": 1,
                "status": "development_evidence_published",
                "protected_evaluation_authorized": False,
                "execution": execution,
                "members": members,
                "private_sha256": root_hashes,
            },
        ),
        "root_summary_mismatch",
    )
    return public


def _directory_contents(directory, expected):
    directory.check()
    _require(
        set(os.listdir(directory.descriptor)) == expected, "unexpected_output_files"
    )


def _verify_outputs(binding, paths):
    attempt = execution_receipt._absolute_path(paths.attempt)
    public = execution_receipt._absolute_path(paths.public_summary)
    _require(
        not attempt.is_relative_to(binding.base.root)
        and not public.is_relative_to(binding.base.root)
        and not public.is_relative_to(attempt),
        "invalid_completion_output_paths",
    )
    paths = development_runner.DevelopmentRunPaths(
        paths.train,
        paths.validation,
        paths.suffix_rules,
        paths.logistic_l1,
        paths.gmm,
        attempt,
        public,
    )
    directories = [
        (
            attempt,
            _RECEIPTS
            | {"evidence"}
            | set(_STEPS)
            | {f"{step}.json" for step in _STEPS},
        ),
        (attempt / "evidence", _ROOT_FILES),
    ]
    for member in _STEPS:
        directories.extend(
            (
                (attempt / member, _RECEIPTS | {"evidence"}),
                (
                    attempt / member / "evidence",
                    _DRIFT_FILES if member == "drift" else _TABULAR_FILES,
                ),
            )
        )
    with ExitStack() as stack:
        pinned = [
            (stack.enter_context(execution_receipt._directory(path)), expected)
            for path, expected in directories
        ]
        public_parent = stack.enter_context(execution_receipt._directory(public.parent))
        files = []
        for directory, expected in pinned:
            _directory_contents(directory, expected)
            for name in sorted(
                expected
                - {"evidence"}
                - (set(_STEPS) if directory.path == attempt else set())
            ):
                files.append((directory, name))
        files.append((public_parent, public.name))
        snapshot = []
        for directory, name in files:
            metadata = execution_receipt._entry(directory, name)
            _require(metadata is not None, "missing_output_file")
            snapshot.append((directory, name, source_runner._file_state(metadata)))
        contents = {
            directory.path / name: source_runner._read_file_once(
                directory.path / name, expected_state=state
            )
            for directory, name, state in snapshot
        }
        with secondary_tabular._numerical_runtime():
            result = _verify_records(binding, paths, contents)
        development_execution.recheck_development_binding(binding)
        for directory, expected in pinned:
            _directory_contents(directory, expected)
        public_parent.check()
        for directory, name, state in snapshot:
            current = execution_receipt._entry(directory, name)
            _require(
                current is not None and source_runner._file_state(current) == state,
                "output_changed_during_verification",
            )
        return result


def verify_development_completion(binding, paths, *, producer_exit_code: int) -> dict:
    """Require observed success and independently check all eight retained members.

    Every retained file is read once through a checked no-follow descriptor. All
    directory identities and file states stay pinned until the final source and
    runtime recheck. Original input partitions and accepted models are not opened.
    """
    _require(
        type(producer_exit_code) is int and producer_exit_code == 0,
        "producer_exit_not_successful",
    )
    try:
        development_execution.recheck_development_binding(binding)
        _require(
            type(paths) is development_runner.DevelopmentRunPaths, "invalid_run_paths"
        )
        return _verify_outputs(binding, paths)
    except DevelopmentCompletionError:
        raise
    except Exception:
        raise DevelopmentCompletionError("completion_verification_failed") from None
