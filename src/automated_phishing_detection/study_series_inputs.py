"""Distinct computational carriers, without observed sources or launch authority.

Source snapshots require independent upstream scientific verification. Snapshot
types and byte consistency do not establish custody, access or execution rights.
"""

from dataclasses import dataclass, field

from . import _study_series_input_cells as cells
from . import _study_series_input_context as context
from ._checkpoint_codec import canonical_bytes
from .operational_cell_inputs import RestoredOperationalCell


class SeriesInputError(ValueError):
    """Series computational input identities fail closed without private details."""


@dataclass(frozen=True)
class SeriesOperationalCell:
    """Unchanged original metadata and validated numerical input, never authority."""

    origin_metadata_bytes: bytes = field(repr=False)
    computational: RestoredOperationalCell = field(repr=False)
    authorizes_execution: bool = field(default=False, init=False)


def build_series_input_metadata(
    origin_bytes,
    profile_bytes,
    internal,
    external,
    *,
    expected_origin_sha256,
    expected_profile_sha256,
    series_reservation_sha256,
    segment_reservation_sha256,
):
    """Retain original same-parent claims only inside explicit historical origin."""
    try:
        return context.build(
            origin_bytes,
            profile_bytes,
            internal,
            external,
            expected_origin_sha256=expected_origin_sha256,
            expected_profile_sha256=expected_profile_sha256,
            series_reservation_sha256=series_reservation_sha256,
            segment_reservation_sha256=segment_reservation_sha256,
        )
    except Exception:
        raise SeriesInputError("invalid_series_operational_inputs") from None


def build_series_cell_descriptor(
    metadata_bytes,
    profile_bytes,
    internal,
    external,
    ordinal,
    *,
    expected_metadata_sha256,
    expected_profile_sha256,
):
    """Select one exact suffix cell from independently verified original sources."""
    try:
        metadata, profile = context.authenticate(
            metadata_bytes,
            profile_bytes,
            internal,
            external,
            expected_metadata_sha256=expected_metadata_sha256,
            expected_profile_sha256=expected_profile_sha256,
        )
        return cells.descriptor(
            metadata_bytes, metadata, profile, internal, external, ordinal
        )
    except Exception:
        raise SeriesInputError("invalid_series_operational_inputs") from None


def restore_series_cell_inputs(
    metadata_bytes,
    profile_bytes,
    internal,
    external,
    descriptor_bytes,
    binding_bytes,
    manifest_bytes,
    *,
    expected_metadata_sha256,
    expected_profile_sha256,
    expected_binding_sha256,
    expected_cell_reservation_sha256,
):
    """Restore the new-kind carrier without invoking legacy same-parent restoration."""
    try:
        return _restore(
            (metadata_bytes, profile_bytes, internal, external),
            (descriptor_bytes, binding_bytes, manifest_bytes),
            (
                expected_metadata_sha256,
                expected_profile_sha256,
                expected_binding_sha256,
                expected_cell_reservation_sha256,
            ),
        )
    except Exception:
        raise SeriesInputError("invalid_series_operational_inputs") from None


def _restore(inputs, payloads, pins):
    metadata_bytes, unused_profile, internal, external = inputs
    metadata_pin, profile_pin, binding_pin, reservation = pins
    metadata, profile = context.authenticate(
        *inputs,
        expected_metadata_sha256=metadata_pin,
        expected_profile_sha256=profile_pin,
    )
    computational = cells.restore(
        metadata_bytes,
        metadata,
        profile,
        internal,
        external,
        *payloads,
        binding_pin,
        reservation,
    )
    return SeriesOperationalCell(canonical_bytes(metadata["origin"]), computational)
