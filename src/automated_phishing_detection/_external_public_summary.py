"""Aggregate-only external summaries with a closed scientific reason vocabulary."""

from dataclasses import asdict
from hashlib import sha256

from . import hypothesis_evaluation, probe_replay
from .bound_secondary import SecondaryInferenceCounts
from .external_replay import ReplayedExternal
from .hypothesis_evaluation import WindowCounts
from .phishvn import PreparedExternal
from .selective_inference import InferenceCounts

_REASONS = {
    None,
    "no_complete_256_row_window",
    "fewer_than_256_training_domains",
    "no_positive_reference_distance",
    "no_training_rows",
    "no_calibration_windows",
}


def _monitor_summary(monitor: probe_replay.MonitorReplay) -> dict:
    if monitor.reason not in _REASONS or any(
        window.reason not in _REASONS for window in monitor.windows
    ):
        raise ValueError("unknown_external_monitor_reason")
    return probe_replay._monitor_summary(monitor)


def _primary_summary(replay: ReplayedExternal) -> dict:
    evidence = replay.evidence
    primary = hypothesis_evaluation.evaluate_primary(
        populations=evidence.populations,
        controls=evidence.controls,
        external_windows=evidence.external_windows,
        audit_windows=WindowCounts(28, 252),
    )
    return asdict(primary)


def public_summary(
    prepared: PreparedExternal,
    replay: ReplayedExternal,
    counts: InferenceCounts,
    secondary_counts: SecondaryInferenceCounts,
    outputs: dict[str, bytes],
) -> dict:
    return {
        "schema_version": 1,
        "status": "external_evidence_composed",
        "protected_evaluation_authorized": False,
        "source_binding": "caller_supplied_preparation_only",
        "row_count": len(replay.rows),
        "domain_count": len({row.record.registrable_domain for row in replay.rows}),
        "preparation": prepared.public_summary,
        "role_counts": replay.evidence.role_counts,
        "offline_inference_counts": asdict(counts),
        "offline_secondary_inference_counts": asdict(secondary_counts),
        "primary": _primary_summary(replay),
        "monitors": {
            monitor.name: _monitor_summary(monitor) for monitor in replay.monitors
        },
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in outputs.items()
        },
    }
