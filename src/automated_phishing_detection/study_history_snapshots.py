"""Authenticate accepted snapshot bytes without eligibility or execution authority.

Scientific payloads remain opaque. The stopped cell and physical evidence require
separate verification; this result cannot authorize replay or any new process.
"""

from dataclasses import dataclass, field

from . import _study_history_snapshot_records as records
from . import _study_run_schema as schema
from ._study_history_snapshot_cells import accepted_cells
from ._study_history_snapshot_sources import external_snapshot, internal_snapshot
from .stopped_study_authorization import (
    StoppedStudyAuthorization,
    verify_stopped_study_authorization,
)


@dataclass(frozen=True)
class StudyHistorySnapshots:
    authorization: StoppedStudyAuthorization = field(repr=False)
    internal_sha256: tuple[tuple[str, str], ...] = field(repr=False)
    external_sha256: tuple[tuple[str, str], ...] = field(repr=False)
    cells_sha256: tuple[tuple[int, tuple[tuple[str, str], ...]], ...] = field(
        repr=False
    )
    scope: str = field(
        default="accepted_sources_and_complete_prefix_opaque_bytes_only", init=False
    )


def _snapshots(authorization, payloads, expected):
    metadata = schema.load(authorization.source_results_bytes)["accepted_inputs"]
    profile = schema.load(authorization.profile_bytes)
    internal = internal_snapshot(
        payloads[0], expected[0], metadata["internal"], profile
    )
    external = external_snapshot(
        payloads[1],
        expected[1],
        metadata["external"],
        metadata["internal"],
        internal,
        profile,
    )
    cells = accepted_cells(payloads[2], expected[2], authorization)
    return StudyHistorySnapshots(
        authorization,
        tuple(sorted(records.hashes(internal).items())),
        tuple(sorted(records.hashes(external).items())),
        cells,
    )


def verify_study_history_snapshots(
    root_snapshot,
    *,
    expected_profile_sha256,
    expected_envelope_sha256,
    expected_root_snapshot_sha256,
    internal_payloads,
    external_payloads,
    cell_snapshots,
    expected_internal_sha256,
    expected_external_sha256,
    expected_cells_sha256,
):
    """Reauthenticate the original root and every accepted snapshot from byte pins."""
    try:
        authorization = verify_stopped_study_authorization(
            root_snapshot,
            expected_profile_sha256=expected_profile_sha256,
            expected_envelope_sha256=expected_envelope_sha256,
            expected_snapshot_sha256=expected_root_snapshot_sha256,
        )
        return _snapshots(
            authorization,
            (internal_payloads, external_payloads, cell_snapshots),
            (expected_internal_sha256, expected_external_sha256, expected_cells_sha256),
        )
    except Exception:
        raise schema.StudyRunRecordError("invalid_study_history_snapshots") from None
