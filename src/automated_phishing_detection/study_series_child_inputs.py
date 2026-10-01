"""Restore only parent-pinned computational bytes, never parent access authority.

Actual public binding, live channel, held root prefix and parent scientific
verification are separate prerequisites. A caller-constructed frame proves none.
"""

from . import _operational_cell_records as records
from . import _operational_input_schema as schema
from . import _study_series_child_input_context as context
from . import _study_series_input_cells as cells
from ._checkpoint_codec import canonical_bytes
from .http_replay import ReplayRequest
from .operational_cell_inputs import RestoredOperationalCell
from .study_series_inputs import SeriesOperationalCell


class SeriesChildInputError(ValueError):
    """The admitted byte-only computational context fails exact consistency."""


def _descriptor(metadata, content, binding, frame, reservation):
    schema.digest(reservation)
    bound = schema.authenticated(binding, frame.cell_binding_sha256)
    records.binding(bound)
    schema.require(bound["cell_reservation_sha256"] == reservation)
    described = schema.authenticated(content, bound["descriptor_sha256"])
    cell = records.descriptor(described)
    schema.require(cell.ordinal == frame.cell_ordinal)
    schema.require(
        described["root_reservation_sha256"] == metadata["root_reservation_sha256"]
    )
    schema.require(described["accepted_inputs_sha256"] == frame.accepted_inputs_sha256)
    return cell, described


def _restore(metadata_bytes, profile_bytes, payloads, frame, reservation):
    metadata = context.authenticate(metadata_bytes, profile_bytes, frame)
    descriptor_bytes, binding_bytes, manifest_bytes = payloads
    cell, described = _descriptor(
        metadata, descriptor_bytes, binding_bytes, frame, reservation
    )
    rows = records.restored_rows(manifest_bytes, cell, described, cells.view(metadata))
    computational = RestoredOperationalCell(
        metadata_bytes,
        descriptor_bytes,
        binding_bytes,
        manifest_bytes,
        cell,
        tuple(ReplayRequest(row.record_id, row.raw_url) for row in rows),
    )
    return SeriesOperationalCell(canonical_bytes(metadata["origin"]), computational)


def restore_series_child_inputs(
    metadata_bytes,
    profile_bytes,
    descriptor_bytes,
    binding_bytes,
    manifest_bytes,
    *,
    frame,
    expected_cell_reservation_sha256,
):
    """Decode immutable admitted bytes without repeating parent source reconstruction."""
    try:
        return _restore(
            metadata_bytes,
            profile_bytes,
            (descriptor_bytes, binding_bytes, manifest_bytes),
            frame,
            expected_cell_reservation_sha256,
        )
    except Exception:
        raise SeriesChildInputError("invalid_series_child_inputs") from None
