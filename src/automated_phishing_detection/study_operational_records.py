"""Immutable same-parent accounting; caller constructors confer no authority."""

from dataclasses import dataclass, field

from . import _operational_input_schema as schema
from . import _study_operational_validation as validation
from ._operational_cell_protocol import SNAPSHOT_NAMES
from ._operational_cell_publication_records import inventory
from ._operational_process_records import ProcessObservation
from ._study_operational_projection import project
from .operational_cell_acceptance import verify_published_cell
from .operational_cell_runner import ObservedOperationalCell, OperationalCellFailure
from .operational_inputs import AcceptedOperationalInputs
from .operational_schedule import OperationalCell, planned_cells


class StudyOperationalRecordError(ValueError):
    """An administrative record is not a closed, ordered study projection."""


@dataclass(frozen=True)
class AcceptedOperationalRun:
    cell: OperationalCell
    observation: ProcessObservation = field(repr=False)
    run_bytes: bytes = field(repr=False)
    descriptor_bytes: bytes = field(repr=False)
    binding_bytes: bytes = field(repr=False)
    public_bytes: bytes = field(repr=False)
    snapshot_sha256: tuple[tuple[str, str], ...] = field(repr=False)


@dataclass(frozen=True)
class StoppedOperationalCell:
    cell: OperationalCell
    stage: str
    failure: OperationalCellFailure | None = field(default=None, repr=False)
    progress: bytes | None = field(default=None, repr=False)


@dataclass(frozen=True)
class OperationalCellSlot:
    cell: OperationalCell
    accepted: AcceptedOperationalRun | None = field(default=None, repr=False)
    stopped: StoppedOperationalCell | None = field(default=None, repr=False)
    returned: ObservedOperationalCell | None = field(default=None, repr=False)

    @property
    def status(self):
        if self.accepted is not None or self.returned is not None:
            return "accepted"
        return "stopped" if self.stopped is not None else "unattempted"


def _freeze_run(completion, cell, payloads):
    inputs = completion.snapshot.inputs
    hashes = tuple(
        sorted((name, validation.digest(content)) for name, content in payloads.items())
    )
    return AcceptedOperationalRun(
        cell,
        completion.observation,
        payloads["attempt/run.json"],
        inputs.descriptor_bytes,
        inputs.binding_bytes,
        payloads["public-summary.json"],
        hashes,
    )


def retain_accepted_cell(completion, *, accepted) -> AcceptedOperationalRun:
    """Retain original accepted bytes without a reference to the bulky snapshot."""
    try:
        cell = validation.returned(completion)
        snapshot = completion.snapshot
        schema.require(
            type(accepted) is AcceptedOperationalInputs
            and snapshot.accepted is accepted
        )
        schema.require(snapshot.inputs.accepted_bytes == accepted.metadata_bytes)
        payloads = inventory(snapshot.payloads, SNAPSHOT_NAMES)
        schema.require(
            payloads["attempt/process-pair.json"] == completion.observation.record
        )
        public = payloads["public-summary.json"]
        verify_published_cell(
            snapshot.payloads, working=snapshot._working, expected_public_bytes=public
        )
        record = _freeze_run(completion, cell, payloads)
        validation.compact(record)
        return record
    except Exception:
        raise StudyOperationalRecordError("invalid_study_operational_records") from None


def _tail(values, index, stopped, returned):
    schema.require(stopped is None or returned is None)
    if stopped is not None:
        schema.require(index < 125)
        validation.stopped(stopped)
        schema.require(stopped.cell == values[index].cell)
        values[index] = OperationalCellSlot(values[index].cell, stopped=stopped)
    if returned is not None:
        schema.require(
            index < 125 and validation.returned(returned) == values[index].cell
        )
        values[index] = OperationalCellSlot(values[index].cell, returned=returned)


def freeze_cell_accounting(completed, *, stopped=None, returned=None):
    """Keep the exact accepted prefix, one stop/fallback, and unattempted suffix."""
    try:
        schema.require(type(completed) is tuple and len(completed) <= 125)
        values = [OperationalCellSlot(cell) for cell in planned_cells()]
        for index, record in enumerate(completed):
            schema.require(
                type(record) is AcceptedOperationalRun
                and record.cell == values[index].cell
            )
            values[index] = OperationalCellSlot(values[index].cell, accepted=record)
        _tail(values, len(completed), stopped, returned)
        result = tuple(values)
        validation.slots(result)
        return result
    except Exception:
        raise StudyOperationalRecordError("invalid_study_operational_records") from None


def cell_accounting_projection(slots):
    """Project administrative status without inventing missing empirical evidence."""
    try:
        validation.slots(slots)
        return [project(slot) for slot in slots]
    except Exception:
        raise StudyOperationalRecordError("invalid_study_operational_records") from None
