"""Load retained drift snapshots without raw data, model reads or fitting.

Expected digests, pins and the public drift summary are caller-supplied claims,
not independent research authorization. A future authenticated caller must bind
them to accepted evidence, then check the loaded primary model and GMM scaler
against this result before replay. This module never creates a fitted
``TrainingReference`` or its constructor marker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import PurePosixPath

from . import baselines, development_completion, secondary_development, secondary_drift


class RetainedDriftError(ValueError):
    """Symbolic rejection of supplied snapshot evidence, without private values."""


@dataclass(frozen=True)
class RetainedDriftReference:
    pins: secondary_development.DevelopmentPins
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    portable_state_sha256: str
    mmd: secondary_drift.MMDReference = field(repr=False)
    psi: secondary_drift.PSIReference = field(repr=False)
    mmd_calibration: secondary_drift.DriftCalibration
    psi_calibration: secondary_drift.DriftCalibration
    validation_record_ids: tuple[str, ...] = field(repr=False)
    audit_validation_positions: tuple[int, ...] = field(repr=False)
    audit_record_ids: tuple[str, ...] = field(repr=False)
    audit_domains: tuple[str, ...] = field(repr=False)
    training_reference_sha256: str
    validation_audit_sha256: str
    preparation_summary_sha256: str


@dataclass(frozen=True)
class _PinsBinding:
    pins: secondary_development.DevelopmentPins


def _require(condition, reason):
    if not condition:
        raise RetainedDriftError(reason)


def _bound_bytes(content, digest, role):
    baselines._lowercase_sha256(digest, f"{role}_sha256")
    _require(type(content) is bytes, f"invalid_{role}_bytes")
    _require(sha256(content).hexdigest() == digest, f"{role}_hash_mismatch")


def _preparation(content, pins):
    summary = secondary_development._json(content)
    prepared = baselines._validate_preparation_summary(summary)
    _require(
        prepared["source_csv_sha256"] == pins.source_csv_sha256,
        "preparation_source_mismatch",
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
    return prepared


def _restore(reference, audit, validation_ids, pins, reference_hash, audit_hash):
    mmd, psi = reference["mmd"], reference["psi"]
    audit_stream = audit["streams"]["audit"]
    return RetainedDriftReference(
        pins=pins,
        scaler_mean=tuple(reference["scaler"]["mean"]),
        scaler_scale=tuple(reference["scaler"]["scale"]),
        portable_state_sha256=reference["portable_state_sha256"],
        mmd=secondary_drift.MMDReference(
            tuple(tuple(row) for row in mmd["values"]),
            tuple(mmd["domains"]),
            tuple(mmd["stable_ids"]),
            mmd["bandwidth_squared"],
            mmd["reason"],
        ),
        psi=secondary_drift.PSIReference(
            tuple(
                secondary_drift.PSIFeatureReference(
                    tuple(feature["internal_edges"]),
                    feature["constant"],
                    tuple(feature["training_counts"]),
                    tuple(feature["training_proportions"]),
                )
                for feature in psi["features"]
            ),
            psi["training_row_count"],
            psi["reason"],
        ),
        mmd_calibration=secondary_drift.DriftCalibration(
            **audit["results"]["mmd"]["calibration"]
        ),
        psi_calibration=secondary_drift.DriftCalibration(
            **audit["results"]["psi"]["calibration"]
        ),
        validation_record_ids=tuple(validation_ids),
        audit_validation_positions=tuple(audit_stream["input_row_positions"]),
        audit_record_ids=tuple(audit_stream["record_ids"]),
        audit_domains=tuple(audit_stream["domains"]),
        training_reference_sha256=reference_hash,
        validation_audit_sha256=audit_hash,
        preparation_summary_sha256=pins.preparation_summary_sha256,
    )


def load_retained_drift_reference(
    training_reference: bytes,
    validation_audit: bytes,
    *,
    expected_reference_sha256: str,
    expected_audit_sha256: str,
    pins: secondary_development.DevelopmentPins,
    preparation_summary: bytes,
    expected_drift_summary: dict,
) -> RetainedDriftReference:
    """Check supplied snapshots and restore their immutable replay state.

    The preparation report is exact hash-bound public JSON. Retained drift
    artifacts must be canonical private JSON, and every byte input must match
    its explicit digest. The shared completion checker verifies membership,
    saved reference arrays, stream allocation and boundary arithmetic, without
    fitting or rescoring. Returned thresholds come from the saved audit, never
    from a replacement calibration. No files are read; no protected-evaluation
    access is authorized.
    """
    try:
        _require(
            type(pins) is secondary_development.DevelopmentPins,
            "invalid_development_pins",
        )
        for name, digest in asdict(pins).items():
            baselines._lowercase_sha256(digest, name)
        _bound_bytes(training_reference, expected_reference_sha256, "reference")
        _bound_bytes(validation_audit, expected_audit_sha256, "audit")
        _bound_bytes(
            preparation_summary, pins.preparation_summary_sha256, "preparation"
        )
        prepared = _preparation(preparation_summary, pins)
        # These are dictionary keys only, not filesystem paths or read authority.
        path = PurePosixPath("retained-drift")
        contents = {
            path / "training-reference.json": training_reference,
            path / "validation-audit.json": validation_audit,
        }
        validation_ids = development_completion._drift(
            contents,
            path,
            expected_drift_summary,
            {
                "training-reference.json": expected_reference_sha256,
                "validation-audit.json": expected_audit_sha256,
            },
            _PinsBinding(pins),
            prepared,
        )
        return _restore(
            development_completion._private_json(training_reference),
            development_completion._private_json(validation_audit),
            validation_ids,
            pins,
            expected_reference_sha256,
            expected_audit_sha256,
        )
    except RetainedDriftError:
        raise
    except development_completion.DevelopmentCompletionError as exc:
        raise RetainedDriftError(str(exc)) from None
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise RetainedDriftError("invalid_retained_drift_evidence") from None
