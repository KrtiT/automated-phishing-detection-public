"""Actual same-parent progress survives rejection without inventing observations."""

from dataclasses import dataclass, field

from . import execution_receipt as receipt
from .internal_process_handoff import ObservedInternalCompletion
from .operational_cell_runner import OperationalCellFailure
from .operational_process import process_progress
from .study_operational_records import StoppedOperationalCell, freeze_cell_accounting


@dataclass(frozen=True)
class StudyRunFailure:
    stage: str
    publishing: bool
    attempt: object = field(repr=False)
    preparation: object = field(repr=False)
    sources: object = field(repr=False)
    accepted: object = field(repr=False)
    completed: tuple = field(repr=False)
    returned: object = field(repr=False)
    cells: tuple | None = field(repr=False)
    candidate: object = field(repr=False)
    retention_progress: bytes | None = field(repr=False)


class StudyProgress:
    def __init__(self, binding, profile, paths, deadlines):
        self.binding, self.profile, self.paths = binding, profile, paths
        self.deadlines = deadlines
        self.stage = "validation"
        self.attempt = self.execution = self.outputs = self.writer = None
        self.fresh = self.preparation = self.sources = self.accepted = None
        self.original = self.current = self.returned = self.reduced = None
        self.sources_started = self.accounting_started = self.held = False
        self.completed = []

    def reserve(self, identity):
        self.attempt = receipt.reserve_attempt(self.paths.attempt, identity=identity)
        self.execution = identity | {
            "reservation_sha256": self.attempt.reservation_sha256
        }

    def remember(self, error):
        if self.original is None:
            self.original = error

    def commit_cell(self, retained):
        self.completed.append(retained)
        self.returned = self.current = None

    def cells(self, error=None):
        stopped = None
        if self.current is not None and self.returned is None:
            failure = getattr(error, "operational_failure", None)
            retained = failure if type(failure) is OperationalCellFailure else None
            stopped = StoppedOperationalCell(
                self.current,
                self.stage if retained is None else retained.stage,
                retained,
                process_progress(error),
            )
        return freeze_cell_accounting(
            tuple(self.completed), stopped=stopped, returned=self.returned
        )

    def source_statuses(self, error=None):
        if self.sources is not None:
            return "accepted", "accepted"
        if not self.sources_started:
            return "unattempted", "unattempted"
        internal = getattr(error, "source_internal", None)
        if type(internal) is ObservedInternalCompletion:
            return "accepted", "stopped"
        return "stopped", "unattempted"

    @property
    def publishing(self):
        return self.writer is not None and self.writer.publishing

    def snapshot(self, error):
        try:
            cells = self.cells(error)
        except Exception:
            cells = None
        progress = None if self.writer is None else self.writer.snapshot()
        candidate = None if self.writer is None else self.writer.candidate
        return StudyRunFailure(
            self.stage,
            self.publishing,
            self.attempt,
            self.preparation if self.preparation is not None else self.fresh,
            self.sources,
            self.accepted,
            tuple(self.completed),
            self.returned,
            cells,
            candidate,
            progress,
        )
