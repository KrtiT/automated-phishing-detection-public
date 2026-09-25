"""Validate saved administrative cell fields, without replay or process authority."""

import re
from dataclasses import asdict

from . import _study_run_schema as schema
from ._operational_cell_protocol import SNAPSHOT_NAMES
from .operational_schedule import planned_cells

_REFERENCES = {
    "retention",
    "snapshot_sha256",
    "observation_sha256",
    "stage",
    "reservation_sha256",
    "progress_sha256",
    "publishing",
}


def _accepted(value):
    schema.require(
        all(value[name] is None for name in ("stage", "progress_sha256", "publishing"))
    )
    schema.operational.digest(value["observation_sha256"])
    if value["retention"] == "unpacked":
        schema.require(value["snapshot_sha256"] is value["reservation_sha256"] is None)
        return False
    schema.require(value["retention"] == "compact")
    hashes = value["snapshot_sha256"]
    schema.keys(hashes, SNAPSHOT_NAMES)
    for digest in hashes.values():
        schema.operational.digest(digest)
    schema.operational.digest(value["reservation_sha256"])
    schema.require(hashes["attempt/reservation.json"] == value["reservation_sha256"])
    schema.require(hashes["attempt/process-pair.json"] == value["observation_sha256"])
    return True


def _stopped(value):
    schema.require(value["retention"] is value["snapshot_sha256"] is None)
    schema.require(type(value["stage"]) is str)
    schema.require(
        re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", value["stage"]) is not None
    )
    for name in ("observation_sha256", "reservation_sha256", "progress_sha256"):
        if value[name] is not None:
            schema.operational.digest(value[name])
    schema.require(value["publishing"] is None or type(value["publishing"]) is bool)


def validate(values):
    schema.require(type(values) is list and len(values) == 125)
    prefix, reservations = True, set()
    for value, cell in zip(values, planned_cells(), strict=True):
        schema.keys(value, {"cell", "status", *_REFERENCES})
        schema.same(value["cell"], asdict(cell))
        status = value["status"]
        schema.require(type(status) is str)
        if status == "accepted":
            schema.require(prefix)
            prefix = _accepted(value)
            reservation = value["reservation_sha256"]
            schema.require(reservation is None or reservation not in reservations)
            reservations.add(reservation)
        elif status == "stopped":
            schema.require(prefix)
            _stopped(value)
            prefix = False
        else:
            schema.require(status == "unattempted")
            schema.require(all(value[name] is None for name in _REFERENCES))
            prefix = False
