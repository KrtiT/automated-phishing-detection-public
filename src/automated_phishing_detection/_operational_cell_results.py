"""Immutable bytes back fresh operational response and aggregate views."""

import json
from dataclasses import dataclass, field

from ._operational_cell_protocol import PRIVATE_NAMES
from ._operational_cell_run_records import decode_run
from .operational_cell_inputs import RestoredOperationalCell
from .operational_inputs import AcceptedOperationalInputs


@dataclass(frozen=True)
class VerifiedOperationalWorking:
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    inputs: RestoredOperationalCell = field(repr=False)
    accepted: AcceptedOperationalInputs = field(repr=False)
    summary_bytes: bytes = field(repr=False)
    reservation_sha256: str

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
        return decode_run(self.payload("run.json"), self.inputs)


@dataclass(frozen=True)
class VerifiedOperationalCell:
    payloads: tuple[tuple[str, bytes], ...] = field(repr=False)
    _working: VerifiedOperationalWorking = field(repr=False)

    def payload(self, name):
        return dict(self.payloads)[name]

    @property
    def inputs(self):
        return self._working.inputs

    @property
    def accepted(self):
        return self._working.accepted

    @property
    def summary_bytes(self):
        return self._working.summary_bytes

    @property
    def summary(self):
        return self._working.summary

    @property
    def run(self):
        return self._working.run

    @property
    def reservation_sha256(self):
        return self._working.reservation_sha256
