"""Closed administrative shapes, without constructing process authority."""

import re
from hashlib import sha256

from . import _operational_cell_records as cells
from . import _operational_input_schema as schema
from . import execution_receipt as receipt
from ._operational_cell_process_records import _observation
from ._operational_cell_protocol import SNAPSHOT_NAMES
from ._operational_cell_results import (
    VerifiedOperationalCell,
    VerifiedOperationalWorking,
)
from ._operational_process_records import ProcessObservation
from ._study_operational_public import validate as validate_public
from .operational_cell_inputs import RestoredOperationalCell
from .operational_cell_runner import ObservedOperationalCell, OperationalCellFailure
from .operational_schedule import planned_cells, validate_cell


def digest(content):
    schema.require(type(content) is bytes)
    return sha256(content).hexdigest()


def hashes(value):
    schema.require(type(value) is tuple)
    result = {}
    for member in value:
        schema.require(type(member) is tuple and len(member) == 2)
        name, content = member
        schema.require(type(name) is str and name not in result)
        schema.digest(content)
        result[name] = content
    schema.keys(result, SNAPSHOT_NAMES)
    return result


def compact(record):
    from .study_operational_records import AcceptedOperationalRun

    schema.require(type(record) is AcceptedOperationalRun)
    validate_cell(record.cell)
    retained = hashes(record.snapshot_sha256)
    bound = schema.loads(record.binding_bytes)
    cells.binding(bound)
    described = schema.authenticated(
        record.descriptor_bytes, bound["descriptor_sha256"]
    )
    schema.require(cells.descriptor(described) == record.cell)
    schema.require(
        bound["cell_reservation_sha256"] == retained["attempt/reservation.json"]
    )
    schema.require(digest(record.run_bytes) == retained["attempt/run.json"])
    content = record.observation.record
    schema.require(digest(content) == retained["attempt/process-pair.json"])
    _observation(content, record.observation, bound["cell_reservation_sha256"])
    public = validate_public(record, described, bound, retained)
    return described, bound, public


def returned(value):
    schema.require(type(value) is ObservedOperationalCell)
    schema.require(type(value.snapshot) is VerifiedOperationalCell)
    schema.require(type(value.snapshot.inputs) is RestoredOperationalCell)
    schema.require(type(value.observation) is ProcessObservation)
    schema.require(type(value.observation.record) is bytes)
    return validate_cell(value.snapshot.inputs.cell)


def stopped(value):
    from .study_operational_records import StoppedOperationalCell

    schema.require(type(value) is StoppedOperationalCell)
    validate_cell(value.cell)
    schema.require(
        type(value.stage) is str
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", value.stage) is not None
    )
    schema.require(value.progress is None or type(value.progress) is bytes)
    failure = value.failure
    if failure is None:
        return
    schema.require(type(failure) is OperationalCellFailure)
    schema.require(failure.cell is None or validate_cell(failure.cell) == value.cell)
    schema.require(failure.stage == value.stage and type(failure.publishing) is bool)
    schema.require(failure.attempt is None or type(failure.attempt) is receipt.Attempt)
    if failure.attempt is not None:
        schema.digest(failure.attempt.reservation_sha256)
    if failure.observation is not None:
        schema.require(type(failure.observation) is ProcessObservation)
        schema.require(type(failure.observation.record) is bytes)
    for member, expected in (
        (failure.working, VerifiedOperationalWorking),
        (failure.candidate, VerifiedOperationalCell),
    ):
        if member is not None:
            schema.require(
                type(member) is expected and member.inputs.cell == value.cell
            )


def _compact_key(value):
    described, bound, public = compact(value)
    shared = {
        name: member
        for name, member in public["execution"].items()
        if name not in ("descriptor_sha256", "reservation_sha256")
    }
    return (described["accepted_inputs_sha256"], shared), bound[
        "cell_reservation_sha256"
    ]


def _slot_shape(value, cell):
    from .study_operational_records import OperationalCellSlot

    schema.require(
        type(value) is OperationalCellSlot and validate_cell(value.cell) == cell
    )
    schema.require(
        sum(
            member is not None
            for member in (value.accepted, value.stopped, value.returned)
        )
        <= 1
    )


def slots(values):
    schema.require(type(values) is tuple and len(values) == 125)
    prefix, shared, reservations = True, None, set()
    for value, cell in zip(values, planned_cells(), strict=True):
        _slot_shape(value, cell)
        if value.accepted is not None:
            schema.require(prefix and value.accepted.cell == cell)
            identity, reservation = _compact_key(value.accepted)
            schema.require(
                (shared is None or shared == identity)
                and reservation not in reservations
            )
            shared = identity
            reservations.add(reservation)
        elif value.returned is not None:
            schema.require(prefix and returned(value.returned) == cell)
            prefix = False
        elif value.stopped is not None:
            schema.require(prefix and value.stopped.cell == cell)
            stopped(value.stopped)
            prefix = False
        else:
            prefix = False
