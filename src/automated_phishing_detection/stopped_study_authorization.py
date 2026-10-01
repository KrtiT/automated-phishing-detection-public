"""Authenticate stopped-prefix metadata without granting execution authority.

Consistency proves neither numerical correctness, physical eligibility, original
source access rights, nor authority to launch any new child process.
"""

from dataclasses import dataclass, field

from . import _adopted_study_records as records
from . import _study_run_schema as schema
from ._checkpoint_codec import canonical_bytes
from ._stopped_study_ledger import validate_ledger
from ._stopped_study_root import authenticate_root, authenticate_snapshot


@dataclass(frozen=True)
class StoppedStudyAuthorization:
    """Immutable historical metadata; never a live authorization capability."""

    reservation_sha256: str
    profile_sha256: str
    envelope_sha256: str
    parent_pid: int
    accepted_ordinals: tuple[int, ...]
    stopped_ordinal: int
    snapshot_sha256: tuple[tuple[str, str], ...]
    execution_bytes: bytes = field(repr=False)
    profile_bytes: bytes = field(repr=False)
    source_results_bytes: bytes = field(repr=False)
    accounting_bytes: bytes = field(repr=False)


def verify_stopped_study_authorization(
    snapshot,
    *,
    expected_profile_sha256,
    expected_envelope_sha256,
    expected_snapshot_sha256,
):
    """Verify the entire accepted prefix against independently supplied pins."""
    try:
        payloads, hashes = authenticate_snapshot(snapshot, expected_snapshot_sha256)
        execution, contents, accounting, scientific, ordinals = authenticate_root(
            payloads,
            snapshot.reservation_sha256,
            expected_profile_sha256,
            expected_envelope_sha256,
        )
        parent_pid = validate_ledger(
            accounting["authorization_ledger"],
            execution,
            contents,
            scientific["cells"][: len(ordinals)],
        )
        return _result(execution, contents, hashes, ordinals, parent_pid)
    except Exception:
        raise schema.StudyRunRecordError(
            "invalid_stopped_study_authorization"
        ) from None


def _result(execution, contents, hashes, ordinals, parent_pid):
    intent = schema.load(contents["study-intent.json"])
    return StoppedStudyAuthorization(
        execution["reservation_sha256"],
        execution["study_profile_sha256"],
        execution["adoption_envelope_sha256"],
        parent_pid,
        ordinals,
        len(ordinals) + 1,
        hashes,
        canonical_bytes(execution),
        records.decoded(intent["profile_bytes"]),
        contents["source-results.json"],
        contents["study-accounting.json"],
    )
