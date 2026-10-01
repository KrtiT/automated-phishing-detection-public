"""Immutable working science with no independent live observation claim."""

import json
from dataclasses import dataclass, field

from ._operational_cell_protocol import PRIVATE_NAMES
from ._operational_cell_run_records import decode_run
from .study_series_inputs import SeriesOperationalCell


@dataclass(frozen=True)
class SeriesWorkingCell:
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    inputs: SeriesOperationalCell = field(repr=False)
    summary_bytes: bytes = field(repr=False)
    reservation_sha256: str
    authorizes_execution: bool = field(default=False, init=False)

    def payload(self, name):
        return dict(self.payloads)[name]

    @property
    def private_outputs(self):
        return {name: self.payload(name) for name in PRIVATE_NAMES}

    @property
    def summary(self):
        return json.loads(self.summary_bytes)

    @property
    def run(self):
        return decode_run(self.payload("run.json"), self.inputs.computational)
