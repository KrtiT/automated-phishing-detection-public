"""Reconstruct original cell mathematics without observation or acceptance.

Original root, complete snapshot, source science, process custody and physical
eligibility must be verified separately. Constructible historical snapshots and
this result confer no execution authority, access rights or live acceptance.
"""

import json
from dataclasses import dataclass, field

from . import _study_history_cell_inputs as inputs
from . import _study_history_cell_records as records
from . import _study_history_cell_science as science
from ._checkpoint_codec import canonical_bytes
from ._operational_cell_run_records import decode_run
from .operational_cell_inputs import RestoredOperationalCell


class HistoricalCellScienceError(ValueError):
    """Symbolic historical replay rejection without private row diagnostics."""


@dataclass(frozen=True)
class HistoricalCellScience:
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    inputs: RestoredOperationalCell = field(repr=False)
    summary_bytes: bytes = field(repr=False)
    reservation_sha256: str

    @property
    def run(self):
        return decode_run(dict(self.payloads)["attempt/run.json"], self.inputs)

    @property
    def summary(self):
        return json.loads(self.summary_bytes)


def _restore(payloads, expected, sources, buffers, pins, directory):
    values = records.authenticate(payloads, expected, buffers, pins)
    restored = inputs.restore(*sources, *buffers, pins[1], pins[2])
    private, public = records.publication(values, restored, pins[2], directory)
    summary = canonical_bytes(science.summary(private, restored, sources[2]))
    records.verify_summary(public, summary)
    return HistoricalCellScience(payloads, restored, summary, pins[2])


def verify_historical_cell_science(
    payloads,
    *,
    expected_snapshot_sha256,
    accepted_metadata_bytes,
    internal_snapshot,
    external_snapshot,
    descriptor_bytes,
    binding_bytes,
    expected_descriptor_sha256,
    expected_binding_sha256,
    expected_cell_reservation_sha256,
    expected_attempt_directory,
):
    """Authenticate original bytes and replay unchanged full-cell science."""
    try:
        return _restore(
            payloads,
            expected_snapshot_sha256,
            (accepted_metadata_bytes, internal_snapshot, external_snapshot),
            (descriptor_bytes, binding_bytes),
            (
                expected_descriptor_sha256,
                expected_binding_sha256,
                expected_cell_reservation_sha256,
            ),
            expected_attempt_directory,
        )
    except Exception:
        raise HistoricalCellScienceError("invalid_historical_cell_science") from None
