"""In-memory development composition of the frozen secondary drift comparators.

Pins are expected identities supplied by a future authenticated caller, not
self-authentication or authorization to acquire records. No file is opened, no
scaler/GMM is fitted, and no primary gate is computed here. Only already-loaded
portable Logistic-L1 state and the accepted GMM's original scaler are used.
"""

from __future__ import annotations

import io
import json
import platform
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np
import threadpoolctl

from . import baselines, fixed_cascade, gmm_monitor, phiusiil, protocol_preflight
from . import secondary_drift as drift

CONTRACT_ID = "secondary-development-v1"
_REFERENCE_MARKER = object()


class SecondaryDevelopmentError(ValueError):
    """Supplied development evidence violates its declared binding or method."""


@dataclass(frozen=True)
class DevelopmentPins:
    train_sha256: str
    validation_sha256: str
    source_csv_sha256: str
    preparation_summary_sha256: str
    suffix_rules_sha256: str
    logistic_l1_artifact_sha256: str
    baseline_contract_sha256: str
    gmm_artifact_sha256: str
    gmm_contract_sha256: str


@dataclass(frozen=True)
class TrainingReference:
    pins: DevelopmentPins
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    training_record_ids: tuple[str, ...]
    training_domains: tuple[str, ...]
    mmd: drift.MMDReference
    psi: drift.PSIReference
    portable_model: fixed_cascade.PortableLogisticL1 = field(repr=False)
    suffix_rules: bytes = field(repr=False)
    validation_declaration: bytes = field(repr=False)
    private_payload: bytes = field(repr=False)
    _marker: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class DevelopmentResult:
    private_outputs: dict[str, bytes]
    public_summary: dict


@dataclass(frozen=True)
class _Partition:
    features: np.ndarray
    raw_urls: tuple[str, ...]
    record_ids: tuple[str, ...]
    domains: tuple[str, ...]


def _require(condition, reason):
    if not condition:
        raise SecondaryDevelopmentError(reason)


def _json_bytes(value):
    return gmm_monitor._canonical_json_bytes(value)


def _json(content):
    return json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )


def _bound_bytes(content, digest, role):
    _require(type(content) is bytes, f"invalid_{role}_bytes")
    _require(sha256(content).hexdigest() == digest, f"{role}_hash_mismatch")


def _input_hashes(pins):
    return {
        "train": pins.train_sha256,
        "validation": pins.validation_sha256,
        "preparation_summary": pins.preparation_summary_sha256,
        "baseline_contract": pins.baseline_contract_sha256,
        "logistic_l1_artifact": pins.logistic_l1_artifact_sha256,
        "gmm_contract": pins.gmm_contract_sha256,
    }


@contextmanager
def _numerical_context():
    gmm_monitor._require_runtime()
    _require(platform.python_version() == "3.10.19", "unsupported_python_version")
    with (
        warnings.catch_warnings(),
        np.errstate(all="raise"),
        threadpoolctl.threadpool_limits(limits=1),
    ):
        warnings.simplefilter("error")
        _require(
            all(pool["num_threads"] == 1 for pool in threadpoolctl.threadpool_info()),
            "numerical_thread_limit_not_applied",
        )
        yield


def _preparation(preparation_summary, suffix_rules, pins):
    _bound_bytes(preparation_summary, pins.preparation_summary_sha256, "preparation")
    _bound_bytes(suffix_rules, pins.suffix_rules_sha256, "suffix_rules")
    summary = _json(preparation_summary)
    prepared = baselines._validate_preparation_summary(summary)
    _require(
        prepared["source_csv_sha256"] == pins.source_csv_sha256, "source_csv_mismatch"
    )
    _require(
        summary["declared_sources"]["public_suffix_list"]["sha256"]
        == pins.suffix_rules_sha256,
        "preparation_suffix_rules_mismatch",
    )
    for split in ("train", "validation"):
        _require(
            prepared["output_hashes"][f"{split}.jsonl"]
            == getattr(pins, f"{split}_sha256"),
            "preparation_partition_mismatch",
        )
    return prepared, protocol_preflight.parse_suffix_rules(suffix_rules.decode("utf-8"))


def _partition(content, split, declaration, pins, rules):
    _bound_bytes(content, getattr(pins, f"{split}_sha256"), split)
    features, _, _, _, _ = baselines._load_partition(
        io.BytesIO(content),
        split=split,
        declared=declaration,
        source_csv_sha256=pins.source_csv_sha256,
    )
    rows = []
    canonical_hashes = set()
    for line in content.splitlines(keepends=True):
        row = _json(line)
        canonical_line = (
            json.dumps(
                row,
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        _require(line == canonical_line, "noncanonical_partition_record")
        canonical = phiusiil.canonicalize_url(row["raw_url"])
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        _require(digest == row["canonical_url_sha256"], "canonical_url_hash_mismatch")
        _require(digest not in canonical_hashes, "duplicate_canonical_url")
        canonical_hashes.add(digest)
        domain = protocol_preflight.registrable_domain_for_url(row["raw_url"], rules)
        _require(domain == row["registrable_domain"], "registrable_domain_mismatch")
        rows.append(row)
    return _Partition(
        features,
        tuple(row["raw_url"] for row in rows),
        tuple(row["record_id"] for row in rows),
        tuple(row["registrable_domain"] for row in rows),
    )


def _accepted_states(logistic_l1, gmm_state, pins, training_count):
    _require(
        type(logistic_l1) is fixed_cascade.PortableLogisticL1
        and logistic_l1._loader_marker is fixed_cascade._LOADED_ARTIFACT_MARKER,
        "logistic_state_not_loaded",
    )
    _require(
        logistic_l1.artifact_sha256 == pins.logistic_l1_artifact_sha256
        and logistic_l1.contract_sha256 == pins.baseline_contract_sha256
        and logistic_l1._n_samples_seen == training_count,
        "logistic_state_binding_mismatch",
    )
    content = _json_bytes(gmm_state)
    _bound_bytes(content, pins.gmm_artifact_sha256, "gmm_artifact")
    gmm = gmm_monitor.load_gmm_artifact_bytes(content)
    _require(gmm["input_hashes"] == _input_hashes(pins), "gmm_input_binding_mismatch")
    _require(
        gmm["scaler"]["n_samples_seen"] == training_count, "gmm_training_count_mismatch"
    )
    return tuple(gmm["scaler"]["mean"]), tuple(gmm["scaler"]["scale"])


def _standardized_features(partition, logistic_l1, mean, scale):
    probabilities = gmm_monitor._finite_array(
        logistic_l1.score_urls(partition.raw_urls),
        (len(partition.raw_urls),),
        "portable monitor probabilities",
    )
    _require(
        np.all((probabilities >= 0) & (probabilities <= 1)),
        "invalid_monitor_probability",
    )
    matrix = np.column_stack((partition.features, probabilities))
    # Preserve the original GMM subtract-then-divide rounding, not reciprocal multiplication.
    standardized = (matrix - np.asarray(mean, dtype=np.float64)) / np.asarray(
        scale, dtype=np.float64
    )
    _require(np.all(np.isfinite(standardized)), "nonfinite_standardized_features")
    return np.ascontiguousarray(standardized, dtype=np.float64)


def _portable_snapshot(model):
    return {
        "mean": model.mean,
        "scale": model.scale,
        "variance": model.variance,
        "coefficients": model.coefficients,
        "intercept": model.intercept,
        "artifact_sha256": model.artifact_sha256,
        "contract_sha256": model.contract_sha256,
    }


def _reference_payload(reference):
    return _json_bytes(
        {
            "schema_version": 1,
            "contract_id": CONTRACT_ID,
            "input_hashes": asdict(reference.pins),
            "training_record_ids": reference.training_record_ids,
            "training_domains": reference.training_domains,
            "scaler": {"mean": reference.scaler_mean, "scale": reference.scaler_scale},
            "portable_state_sha256": sha256(
                _json_bytes(_portable_snapshot(reference.portable_model))
            ).hexdigest(),
            "validation_declaration_sha256": sha256(
                reference.validation_declaration
            ).hexdigest(),
            "mmd": asdict(reference.mmd),
            "psi": asdict(reference.psi),
        }
    )


def build_training_reference(
    train_content: bytes,
    *,
    preparation_summary: bytes,
    suffix_rules: bytes,
    logistic_l1: fixed_cascade.PortableLogisticL1,
    gmm_state: dict,
    pins: DevelopmentPins,
) -> TrainingReference:
    """Bind training bytes and construct immutable references before validation.

    The preparation metadata declares later validation counts and hashes, but no
    validation rows can be supplied to this constructor. Expected pins themselves
    must still be authenticated by the future development file/process runner.
    """
    try:
        with _numerical_context():
            _require(type(pins) is DevelopmentPins, "invalid_development_pins")
            for name, digest in asdict(pins).items():
                baselines._lowercase_sha256(digest, name)
            prepared, rules = _preparation(preparation_summary, suffix_rules, pins)
            training = _partition(
                train_content, "train", prepared["splits"]["train"], pins, rules
            )
            mean, scale = _accepted_states(
                logistic_l1, gmm_state, pins, len(training.raw_urls)
            )
            matrix = _standardized_features(training, logistic_l1, mean, scale)
            reference = TrainingReference(
                pins,
                mean,
                scale,
                training.record_ids,
                training.domains,
                drift.fit_mmd_reference(matrix, training.domains, training.record_ids),
                drift.fit_psi_reference(matrix),
                logistic_l1,
                suffix_rules,
                _json_bytes(prepared["splits"]["validation"]),
                b"",
                _REFERENCE_MARKER,
            )
            return TrainingReference(
                **{**vars(reference), "private_payload": _reference_payload(reference)}
            )
    except SecondaryDevelopmentError:
        raise
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OverflowError,
        Warning,
        FloatingPointError,
        UnicodeError,
    ):
        raise SecondaryDevelopmentError("training_reference_failed") from None


def _check_reference(reference):
    _require(
        type(reference) is TrainingReference and reference._marker is _REFERENCE_MARKER,
        "invalid_training_reference",
    )
    _require(
        _reference_payload(reference) == reference.private_payload,
        "training_reference_changed",
    )
    model = reference.portable_model
    _require(
        type(model) is fixed_cascade.PortableLogisticL1
        and model._loader_marker is fixed_cascade._LOADED_ARTIFACT_MARKER
        and model._n_samples_seen == len(reference.training_record_ids)
        and model.artifact_sha256 == reference.pins.logistic_l1_artifact_sha256
        and model.contract_sha256 == reference.pins.baseline_contract_sha256,
        "reference_portable_state_changed",
    )
    _bound_bytes(
        reference.suffix_rules, reference.pins.suffix_rules_sha256, "suffix_rules"
    )


def _method_result(calibration, audit, reference_reason):
    calibration_count = len(calibration.window_end_positions)
    audit_count = len(audit.window_end_positions)
    calibration_reason = reference_reason or calibration.reason
    boundary = (
        drift.DriftCalibration(None, calibration_count, calibration_reason)
        if calibration_reason
        else drift.calibrate_window_scores(calibration.scores)
    )
    reason = boundary.reason or audit.reason
    audited = (
        drift.DriftAudit(
            boundary.threshold, calibration_count, audit_count, (), None, None, reason
        )
        if reason
        else drift.audit_window_scores(audit.scores, boundary)
    )
    summary = {
        "status": "not_estimable" if reason else "estimated",
        "reason": reason,
        "threshold": boundary.threshold,
        "calibration_window_count": boundary.calibration_window_count,
        "audit_window_count": audited.window_count,
        "audit_alert_count": audited.alert_count,
        "audit_alert_fraction": audited.alert_fraction,
    }
    return summary, {"calibration": asdict(boundary), "audit": asdict(audited)}


def _validation_result(reference, validation, allocation, traces):
    methods, results = {}, {}
    for method in ("mmd", "psi"):
        methods[method], results[method] = _method_result(
            traces["calibration"][method],
            traces["audit"][method],
            getattr(reference, method).reason,
        )
    streams = {}
    for name, indices in allocation.items():
        streams[name] = {
            "input_row_positions": indices,
            "record_ids": tuple(validation.record_ids[i] for i in indices),
            "domains": tuple(validation.domains[i] for i in indices),
            "window_end_positions": traces[name]["psi"].window_end_positions,
            "mmd": asdict(traces[name]["mmd"]),
            "psi": asdict(traces[name]["psi"]),
        }
    private = {
        "training-reference.json": reference.private_payload,
        "validation-audit.json": _json_bytes(
            {
                "schema_version": 1,
                "contract_id": CONTRACT_ID,
                "training_reference_sha256": sha256(
                    reference.private_payload
                ).hexdigest(),
                "input_hashes": asdict(reference.pins),
                "window_length": drift.WINDOW_SIZE,
                "window_stride": drift.WINDOW_STRIDE,
                "streams": streams,
                "results": results,
            }
        ),
    }
    available = sum(value["status"] == "estimated" for value in methods.values())
    status = (
        "completed_development_validation"
        if available == 2
        else "partially_estimable"
        if available
        else "not_estimable"
    )
    public = {
        "schema_version": 1,
        "contract_id": CONTRACT_ID,
        "status": status,
        "analysis_stage": "development_validation_only",
        "analysis_role": "secondary_descriptive_only",
        "protected_evaluation_authorized": False,
        "input_binding": "caller_supplied_expected_pins_not_authorization",
        "input_hashes": asdict(reference.pins),
        "input_counts": {
            "training_rows": len(reference.training_record_ids),
            "training_domain_count": len(set(reference.training_domains)),
            "validation_rows": len(validation.record_ids),
            "validation_domain_count": len(set(validation.domains)),
            **{f"{name}_rows": len(indices) for name, indices in allocation.items()},
        },
        "methods": methods,
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in private.items()
        },
    }
    return DevelopmentResult(private, public)


def evaluate_validation(
    reference: TrainingReference, validation_content: bytes
) -> DevelopmentResult:
    """Apply fixed training references to the original GMM calibration/audit split."""
    try:
        with _numerical_context():
            _check_reference(reference)
            rules = protocol_preflight.parse_suffix_rules(
                reference.suffix_rules.decode("utf-8")
            )
            validation = _partition(
                validation_content,
                "validation",
                _json(reference.validation_declaration),
                reference.pins,
                rules,
            )
            _require(
                not set(reference.training_record_ids).intersection(
                    validation.record_ids
                ),
                "training_validation_identity_overlap",
            )
            _require(
                not set(reference.training_domains).intersection(validation.domains),
                "training_validation_domain_overlap",
            )
            matrix = _standardized_features(
                validation,
                reference.portable_model,
                reference.scaler_mean,
                reference.scaler_scale,
            )
            allocation = gmm_monitor.allocate_validation_domains(validation.domains)
            traces = {
                name: {
                    "mmd": drift.mmd_window_scores(
                        reference.mmd, matrix[list(indices)]
                    ),
                    "psi": drift.psi_window_scores(
                        reference.psi, matrix[list(indices)]
                    ),
                }
                for name, indices in allocation.items()
            }
            return _validation_result(reference, validation, allocation, traces)
    except SecondaryDevelopmentError:
        raise
    except (
        ValueError,
        TypeError,
        RuntimeError,
        OverflowError,
        Warning,
        FloatingPointError,
        UnicodeError,
    ):
        raise SecondaryDevelopmentError("validation_evaluation_failed") from None
