"""Authenticate supplied cancellation history without granting execution authority.

Caller-supplied inventories cannot establish filesystem absence or custody.
Record consistency establishes neither physical cause nor scientific eligibility.
"""

from dataclasses import dataclass, field

from . import _study_run_schema as schema
from ._stopped_study_cell_inputs import authenticate_inputs
from ._stopped_study_cell_process import authenticate_process
from ._stopped_study_cell_records import (
    ATTEMPT_NAMES,
    INPUT_NAMES,
    authenticate_inventory,
    authenticate_projection,
    stopped_admission,
)
from .stopped_study_authorization import verify_stopped_study_authorization


@dataclass(frozen=True)
class StoppedStudyCellHistory:
    """Immutable historical bytes and identities, never a live capability."""

    root_reservation_sha256: str
    cell_reservation_sha256: str
    cell_ordinal: int
    parent_pid: int
    service_pid: int
    service_command_sha256: str
    cell_binding_sha256: str
    root_snapshot_sha256: tuple[tuple[str, str], ...]
    attempt_snapshot_sha256: tuple[tuple[str, str], ...]
    input_snapshot_sha256: tuple[tuple[str, str], ...]
    root_payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    attempt_payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    input_payloads: tuple[tuple[str, bytes], ...] = field(repr=False)


def verify_stopped_study_cell_history(
    root_snapshot,
    attempt_payloads,
    input_payloads,
    *,
    expected_profile_sha256,
    expected_envelope_sha256,
    expected_root_snapshot_sha256,
    expected_attempt_snapshot_sha256,
    expected_input_snapshot_sha256,
):
    """Recheck the original root and all supplied stopped-cell bytes and pins."""
    try:
        root = verify_stopped_study_authorization(
            root_snapshot,
            expected_profile_sha256=expected_profile_sha256,
            expected_envelope_sha256=expected_envelope_sha256,
            expected_snapshot_sha256=expected_root_snapshot_sha256,
        )
        return _history(
            root,
            expected_attempt_snapshot_sha256,
            expected_input_snapshot_sha256,
            (root_snapshot.payloads, attempt_payloads, input_payloads),
        )
    except Exception:
        raise schema.StudyRunRecordError("invalid_stopped_study_cell_history") from None


def _history(root, expected_attempt, expected_input, payloads):
    attempt, attempt_hashes = authenticate_inventory(
        payloads[1], expected_attempt, ATTEMPT_NAMES
    )
    inputs, input_hashes = authenticate_inventory(
        payloads[2], expected_input, INPUT_NAMES
    )
    entry, frame, projection = stopped_admission(root)
    cell, metadata, reservation = authenticate_inputs(root, attempt, inputs, frame)
    authenticate_projection(projection, reservation, attempt)
    authenticate_process(attempt, reservation, cell, metadata, entry, frame)
    return _result(
        root, reservation, frame, entry, (attempt_hashes, input_hashes), payloads
    )


def _result(root, reservation, frame, entry, hashes, payloads):
    return StoppedStudyCellHistory(
        root.reservation_sha256,
        reservation,
        root.stopped_ordinal,
        root.parent_pid,
        entry["launched_pid"],
        entry["command_sha256"],
        frame.cell_binding_sha256,
        root.snapshot_sha256,
        hashes[0],
        hashes[1],
        *payloads,
    )
