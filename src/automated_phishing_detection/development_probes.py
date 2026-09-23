"""Connect retained drift state to the original development audit rows.

This byte-only adapter checks supplied identities, not the caller's permission to
read research files. Official execution still needs the reviewed process runner
and primary-artifact binding. No reference, boundary or model is fitted here.
"""

from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256

from . import gmm_monitor, probe_replay, secondary_development
from .retained_drift import RetainedDriftReference


class DevelopmentProbeError(ValueError):
    """Supplied validation rows and retained replay state do not agree."""


def _require(condition, symbol):
    if not condition:
        raise DevelopmentProbeError(symbol)


def prepare_audit_probe_rows(
    *,
    validation_bytes: bytes,
    suffix_rules_bytes: bytes,
    preparation_summary: bytes,
    retained: RetainedDriftReference,
) -> tuple[probe_replay.AuditInput, ...]:
    """Verify the complete validation partition, then retain its fixed audit order.

    Labels remain in the supplied source bytes only. They do not select rows or
    enter the returned probe inputs. Recomputing the original label-blind domain
    allocation checks the saved positions; it never chooses another split.
    """
    _require(type(retained) is RetainedDriftReference, "invalid_retained_reference")
    try:
        with secondary_development._numerical_context():
            prepared, rules = secondary_development._preparation(
                preparation_summary, suffix_rules_bytes, retained.pins
            )
            _require(
                retained.preparation_summary_sha256
                == retained.pins.preparation_summary_sha256,
                "preparation_identity_mismatch",
            )
            validation = secondary_development._partition(
                validation_bytes,
                "validation",
                prepared["splits"]["validation"],
                retained.pins,
                rules,
            )
            positions = gmm_monitor.allocate_validation_domains(validation.domains)[
                "audit"
            ]
            _require(
                validation.record_ids == retained.validation_record_ids
                and positions == retained.audit_validation_positions
                and tuple(validation.record_ids[i] for i in positions)
                == retained.audit_record_ids
                and tuple(validation.domains[i] for i in positions)
                == retained.audit_domains,
                "retained_audit_alignment_mismatch",
            )
            return tuple(
                probe_replay.AuditInput(
                    validation.record_ids[i], i, validation.raw_urls[i]
                )
                for i in positions
            )
    except DevelopmentProbeError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise DevelopmentProbeError("invalid_development_probe_inputs") from None


def replay_development_probes(
    *,
    validation_bytes: bytes,
    suffix_rules_bytes: bytes,
    preparation_summary: bytes,
    retained: RetainedDriftReference,
    primary_scorer,
    stage1_model,
    gmm_artifact: dict,
    operating_points: probe_replay.OperatingPoints,
    row_callback: Callable[[str, probe_replay.ProbeRow], None] | None = None,
    stream_callback: Callable[[probe_replay.ProbeStream], None] | None = None,
) -> probe_replay.ProbeReplay:
    """Replay four aligned streams using saved references and unchanged model state.

    The caller must bind the primary scorer and operating points separately.
    Supplied artifact/reference identities are checked before its first call.
    Optional synchronous callbacks pass through unchanged. Row snapshots precede
    policy replay, so their routing fields are provisional; completed-stream
    snapshots have authoritative routing and windows. Callback failures stop work.
    """
    rows = prepare_audit_probe_rows(
        validation_bytes=validation_bytes,
        suffix_rules_bytes=suffix_rules_bytes,
        preparation_summary=preparation_summary,
        retained=retained,
    )
    try:
        with secondary_development._numerical_context():
            mean, scale = secondary_development._accepted_states(
                stage1_model,
                gmm_artifact,
                retained.pins,
                retained.psi.training_row_count,
            )
            portable_hash = sha256(
                secondary_development._json_bytes(
                    secondary_development._portable_snapshot(stage1_model)
                )
            ).hexdigest()
            _require(
                mean == retained.scaler_mean
                and scale == retained.scaler_scale
                and portable_hash == retained.portable_state_sha256,
                "retained_monitor_state_mismatch",
            )
            return probe_replay.replay_probes(
                rows,
                primary_scorer=primary_scorer,
                stage1_model=stage1_model,
                gmm_artifact=gmm_artifact,
                operating_points=operating_points,
                mmd_reference=retained.mmd,
                mmd_calibration=retained.mmd_calibration,
                psi_reference=retained.psi,
                psi_calibration=retained.psi_calibration,
                row_callback=row_callback,
                stream_callback=stream_callback,
            )
    except DevelopmentProbeError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        raise DevelopmentProbeError("development_probe_replay_failed") from None
