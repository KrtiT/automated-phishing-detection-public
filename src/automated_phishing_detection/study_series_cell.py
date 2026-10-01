"""Reconstruct complete fresh-segment science, never live ownership or access."""

import json
from dataclasses import dataclass, field

from . import _study_history_cell_records as records
from . import _study_history_cell_science as science
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_run_records import decode_run
from .study_series_inputs import SeriesOperationalCell


class SeriesCellScienceError(ValueError):
    """The independently pinned complete series publication is inconsistent."""


@dataclass(frozen=True)
class SeriesCellScience:
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    inputs: SeriesOperationalCell = field(repr=False)
    summary_bytes: bytes = field(repr=False)
    reservation_sha256: str
    authorizes_execution: bool = field(default=False, init=False)

    @property
    def run(self):
        return decode_run(
            dict(self.payloads)["attempt/run.json"], self.inputs.computational
        )

    @property
    def summary(self):
        return json.loads(self.summary_bytes)


def _restore(payloads, expected, context, buffers, pins, directory):
    values = records.authenticate(payloads, expected, buffers[:2], pins[2:])
    from .study_series_inputs import restore_series_cell_inputs

    restored = restore_series_cell_inputs(
        *context,
        *buffers,
        expected_metadata_sha256=pins[0],
        expected_profile_sha256=pins[1],
        expected_binding_sha256=pins[3],
        expected_cell_reservation_sha256=pins[4],
    )
    private, public = records.publication(
        values, restored.computational, pins[4], directory
    )
    summary = canonical_bytes(
        science.summary(private, restored.computational, context[3])
    )
    records.verify_summary(public, summary)
    return SeriesCellScience(payloads, restored, summary, pins[4])


def _checked(
    payloads,
    expected,
    context,
    buffers,
    metadata_pin,
    profile_pin,
    descriptor_pin,
    binding_pin,
    reservation_pin,
    directory,
):
    pins = metadata_pin, profile_pin, descriptor_pin, binding_pin, reservation_pin
    try:
        return _restore(payloads, expected, context, buffers, pins, directory)
    except Exception:
        raise SeriesCellScienceError("invalid_series_cell_science") from None


def verify_series_cell_science(
    payloads,
    *,
    expected_snapshot_sha256,
    metadata_bytes,
    profile_bytes,
    internal_snapshot,
    external_snapshot,
    descriptor_bytes,
    binding_bytes,
    manifest_bytes,
    expected_metadata_sha256,
    expected_profile_sha256,
    expected_descriptor_sha256,
    expected_binding_sha256,
    expected_cell_reservation_sha256,
    expected_attempt_directory,
):
    return _checked(
        payloads,
        expected_snapshot_sha256,
        (metadata_bytes, profile_bytes, internal_snapshot, external_snapshot),
        (descriptor_bytes, binding_bytes, manifest_bytes),
        expected_metadata_sha256,
        expected_profile_sha256,
        expected_descriptor_sha256,
        expected_binding_sha256,
        expected_cell_reservation_sha256,
        expected_attempt_directory,
    )
