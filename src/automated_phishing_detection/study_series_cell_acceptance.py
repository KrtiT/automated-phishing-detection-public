"""Pure working/publication joins; only the actual parent can accept a live cell."""

from dataclasses import asdict

from . import _operational_cell_publication_records as publication
from . import _operational_input_schema as schema
from . import _study_history_cell_science as science
from . import _study_series_cell_records as records
from . import execution_receipt as receipt
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_process_records import verify_process_records
from ._operational_cell_protocol import PROTOCOL, SNAPSHOT_NAMES, WORKING_NAMES
from ._study_execution_policy import DEADLINES
from ._study_series_cell_results import SeriesWorkingCell
from .study_series_cell import SeriesCellScience


class SeriesCellAcceptanceError(ValueError):
    """The new cell differs from independently held parent expectations."""


def series_cell_identity(metadata_bytes, descriptor_bytes, *, expected_metadata_sha256):
    try:
        return records.identity(
            metadata_bytes, descriptor_bytes, expected_metadata_sha256
        )
    except Exception:
        raise SeriesCellAcceptanceError("invalid_series_cell") from None


def _working(
    payloads, attempt, identity, inputs, context, pins, observation, commands, deadlines
):
    values = publication.inventory(payloads, WORKING_NAMES)
    schema.same(deadlines, dict(DEADLINES))
    restored = records.restore(inputs, attempt.reservation_sha256, context, pins)
    computational = restored.computational
    expected = records.identity(
        computational.accepted_bytes, computational.descriptor_bytes, pins[0]
    )
    schema.same(expected, identity)
    publication.verify_reservation(values["reservation.json"], attempt, expected)
    verify_process_records(
        values,
        inputs=computational,
        observation=observation,
        reservation=attempt.reservation_sha256,
        service_command=commands[0],
        client_command=commands[1],
        expected_deadlines=deadlines,
    )
    summary = canonical_bytes(science.summary(values, computational, context[2]))
    return SeriesWorkingCell(payloads, restored, summary, attempt.reservation_sha256)


def verify_series_working_cell(
    payloads,
    *,
    attempt,
    expected_identity,
    inputs,
    profile_bytes,
    expected_profile_sha256,
    expected_metadata_sha256,
    internal_snapshot,
    external_snapshot,
    observation,
    service_command,
    client_command,
    expected_deadlines,
):
    try:
        return _working(
            payloads,
            attempt,
            expected_identity,
            inputs,
            (profile_bytes, internal_snapshot, external_snapshot),
            (expected_metadata_sha256, expected_profile_sha256),
            observation,
            (service_command, client_command),
            expected_deadlines,
        )
    except Exception:
        raise SeriesCellAcceptanceError("invalid_series_cell") from None


def build_series_cell_public(working, *, reservation_sha256):
    try:
        schema.require(type(working) is SeriesWorkingCell)
        schema.digest(reservation_sha256)
        schema.require(reservation_sha256 == working.reservation_sha256)
        inputs = working.inputs.computational
        identity = records.identity(
            inputs.accepted_bytes,
            inputs.descriptor_bytes,
            records.digest(inputs.accepted_bytes),
        )
        return {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "operational_evidence_published",
            "execution": identity | {"reservation_sha256": reservation_sha256},
            "cell": asdict(inputs.cell),
            "summary": working.summary,
            "private_sha256": publication.private_hashes(working),
        }
    except Exception:
        raise SeriesCellAcceptanceError("invalid_series_cell") from None


def verify_series_published_cell(payloads, *, working, expected_public_bytes):
    try:
        schema.require(type(working) is SeriesWorkingCell)
        schema.require(type(expected_public_bytes) is bytes)
        public = build_series_cell_public(
            working, reservation_sha256=working.reservation_sha256
        )
        schema.require(
            expected_public_bytes == receipt._json_bytes(public, "public_summary")
        )
        values = publication.inventory(payloads, SNAPSHOT_NAMES)
        publication.verify_completion(values, working, expected_public_bytes)
        return SeriesCellScience(
            payloads, working.inputs, working.summary_bytes, working.reservation_sha256
        )
    except Exception:
        raise SeriesCellAcceptanceError("invalid_series_cell") from None
