"""Preserve actual private parent state without inferring process success."""

from dataclasses import dataclass, field
from functools import partial

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._operational_cell_results import (
    VerifiedOperationalCell,
    VerifiedOperationalWorking,
)
from ._operational_cell_runner_context import OperationalCellExecutionError
from ._operational_process_records import ProcessObservation
from ._prepared_failure_context import carry_failure_context
from .internal_failure import failure_kind
from .operational_cell_inputs import OperationalCapacityError
from .operational_schedule import OperationalCell


@dataclass(frozen=True)
class OperationalCellFailure:
    cell: OperationalCell | None = field(repr=False)
    attempt: receipt.Attempt | None = field(repr=False)
    observation: ProcessObservation | None = field(repr=False)
    working: VerifiedOperationalWorking | None = field(repr=False)
    candidate: VerifiedOperationalCell | None = field(repr=False)
    stage: str
    publishing: bool


class CellProgress:
    def __init__(self):
        self.cell = self.attempt = self.observation = self.completer = None
        self.original = None
        self.stage = "validation"

    def reserve(self, path, identity):
        self.attempt = receipt.reserve_attempt(path, identity=identity)

    def snapshot(self):
        return OperationalCellFailure(
            self.cell,
            self.attempt,
            self.observation,
            None if self.completer is None else self.completer.working,
            None if self.completer is None else self.completer.candidate,
            self.stage,
            self.completer is not None and self.completer.publishing,
        )


def _attach(error, state):
    try:
        values = BaseException.__dict__["__dict__"].__get__(error)
        if not dict.__contains__(values, "operational_failure"):
            dict.__setitem__(values, "operational_failure", state.snapshot())
    except BaseException:
        pass


def _persist(state, error):
    if state.attempt is not None and not state.snapshot().publishing:
        files.deferred(
            partial(
                receipt.record_failure,
                state.attempt,
                stage=state.stage,
                error_type=failure_kind(error),
            )
        )


def reject(state, error):
    if state.original is not None and not isinstance(state.original, Exception):
        error = state.original
    carry_failure_context(error, state.original)
    _attach(error, state)
    try:
        _persist(state, error)
    except BaseException as later:
        if isinstance(error, Exception) and not isinstance(later, Exception):
            carry_failure_context(later, error)
            _attach(later, state)
            error = later
    if not isinstance(error, Exception) or isinstance(error, OperationalCapacityError):
        raise error from None
    rejected = OperationalCellExecutionError("operational_cell_execution_failed")
    carry_failure_context(rejected, error)
    _attach(rejected, state)
    raise rejected from None
