"""Bind accepted drift snapshots before inference ownership, without fitting.

This loader does not authorize protected access. The caller remains responsible
for a reviewed execution binding, reservation and single-read source boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path

from . import (
    baselines,
    execution_preflight,
    fixed_cascade,
    phiusiil,
    retained_drift,
    secondary_development,
    seed_probe_execution,
)
from .bound_models import BoundModels
from .bound_secondary import PUBLIC_REPORTS
from .execution_preflight import ExecutionBinding
from .secondary_development import DevelopmentPins
from .source_runner import _read_file_once

_REPORT, _REPORT_SHA256 = PUBLIC_REPORTS["tabular"]
_PREPARATION = "reports/phiusiil-preparation-summary.json"
_SOURCE = "data/sources.json"


class BoundDriftError(ValueError):
    """An accepted public chain or retained private snapshot does not match."""


@dataclass(frozen=True)
class DriftArtifactPaths:
    training_reference: Path
    validation_audit: Path


@dataclass(frozen=True)
class BoundDrift:
    reference: retained_drift.RetainedDriftReference = field(repr=False)
    training_reference_bytes: bytes = field(repr=False)
    validation_audit_bytes: bytes = field(repr=False)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise BoundDriftError(reason)


def _public_inputs(binding: ExecutionBinding) -> dict[str, bytes]:
    hashes = dict(binding.source_hashes)
    _require(hashes.get(_REPORT) == _REPORT_SHA256, "public_report_not_accepted")
    contents = {}
    try:
        for relative in (_REPORT, _PREPARATION, _SOURCE):
            content = execution_preflight._read_regular(binding.root, relative)
            _require(
                sha256(content).hexdigest() == hashes.get(relative),
                "public_input_hash_mismatch",
            )
            contents[relative] = content
    except (OSError, ValueError):
        raise BoundDriftError("public_input_unavailable_or_mismatched") from None
    return contents


def _pins(
    report: dict, binding: ExecutionBinding, models: BoundModels
) -> DevelopmentPins:
    pins = DevelopmentPins(**report["completion"]["execution"]["pins"])
    for value in asdict(pins).values():
        baselines._lowercase_sha256(value, "development_pin")
    hashes, artifacts = dict(binding.source_hashes), dict(models.artifact_hashes)
    _require(
        pins.preparation_summary_sha256 == hashes[_PREPARATION]
        and pins.baseline_contract_sha256
        == hashes["data/rq1-baseline-contract-v2.json"]
        and pins.gmm_contract_sha256
        == hashes["data/rq2-gmm-development-contract-v1.json"]
        and pins.logistic_l1_artifact_sha256 == artifacts["logistic-l1.json"]
        and pins.gmm_artifact_sha256 == artifacts["gmm.json"],
        "accepted_drift_binding_mismatch",
    )
    return pins


def _preparation(
    source_bytes: bytes,
    preparation_bytes: bytes,
    source_hash: str,
    pins: DevelopmentPins,
) -> int:
    source = phiusiil._load_source_spec(source_bytes)
    summary = secondary_development._json(preparation_bytes)
    prepared = baselines._validate_preparation_summary(summary)
    _require(
        summary["source_spec_sha256"] == source_hash
        and fixed_cascade._matches_exactly(summary["declared_sources"], source)
        and prepared["source_csv_sha256"] == pins.source_csv_sha256
        and source["public_suffix_list"]["sha256"] == pins.suffix_rules_sha256,
        "source_preparation_chain_mismatch",
    )
    for split in ("train", "validation"):
        _require(
            prepared["output_hashes"][f"{split}.jsonl"]
            == getattr(pins, f"{split}_sha256"),
            "preparation_partition_mismatch",
        )
    return prepared["splits"]["train"]["row_count"]


def _primary_state(
    reference: retained_drift.RetainedDriftReference,
    models: BoundModels,
    scaler: tuple[tuple[float, ...], tuple[float, ...]],
    training_count: int,
) -> None:
    snapshot = secondary_development._portable_snapshot(models.cascade.stage1_model)
    digest = sha256(secondary_development._json_bytes(snapshot)).hexdigest()
    _require(
        reference.scaler_mean == scaler[0]
        and reference.scaler_scale == scaler[1]
        and reference.portable_state_sha256 == digest
        and reference.psi.training_row_count == training_count,
        "primary_drift_state_mismatch",
    )


def _load(
    binding: ExecutionBinding, paths: DriftArtifactPaths, models: BoundModels
) -> BoundDrift:
    contents = _public_inputs(binding)
    report = secondary_development._json(contents[_REPORT])
    pins = _pins(report, binding, models)
    summary, reference_hash, audit_hash = seed_probe_execution._retained_drift(
        report, pins
    )
    hashes = dict(binding.source_hashes)
    count = _preparation(
        contents[_SOURCE], contents[_PREPARATION], hashes[_SOURCE], pins
    )
    scaler = secondary_development._accepted_states(
        models.cascade.stage1_model, models.gmm, pins, count
    )
    reference_bytes = _read_file_once(paths.training_reference)
    audit_bytes = _read_file_once(paths.validation_audit)
    reference = retained_drift.load_retained_drift_reference(
        reference_bytes,
        audit_bytes,
        expected_reference_sha256=reference_hash,
        expected_audit_sha256=audit_hash,
        pins=pins,
        preparation_summary=contents[_PREPARATION],
        expected_drift_summary=secondary_development._json(summary),
    )
    _primary_state(reference, models, scaler, count)
    return BoundDrift(reference, reference_bytes, audit_bytes)


def load_bound_drift(
    binding: ExecutionBinding, paths: DriftArtifactPaths, models: BoundModels
) -> BoundDrift:
    """Read each snapshot once after public and loaded-primary state checks."""
    try:
        return _load(binding, paths, models)
    except BoundDriftError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OSError, RecursionError):
        raise BoundDriftError("invalid_bound_drift_evidence") from None
