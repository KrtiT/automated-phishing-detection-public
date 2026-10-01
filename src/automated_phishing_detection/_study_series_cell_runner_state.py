"""Keep actual fresh-series failures distinct from legacy observed-cell types."""

from dataclasses import dataclass, field

from . import _operational_cell_failure as original
from ._prepared_failure_context import carry_failure_context
from .study_series_ledger import SeriesCellStop


@dataclass(frozen=True)
class SeriesCellFailure:
    cell: object = field(repr=False)
    attempt: object = field(repr=False)
    observation: object = field(repr=False)
    working: object = field(repr=False)
    candidate: object = field(repr=False)
    stage: str
    publishing: bool


class SeriesCellProgress(original.CellProgress):
    def __init__(self, public, metadata, internal, external, cell, admissions):
        super().__init__()
        self.public, self.metadata_bytes = public, metadata
        self.internal_snapshot, self.external_snapshot = internal, external
        self.cell, self.admissions = cell, admissions
        self.paths = self.selected = self.binding_bytes = self.pair_intent_bytes = None
        self.ledger_checked = False

    def snapshot(self):
        return SeriesCellFailure(
            self.cell,
            self.attempt,
            self.observation,
            None if self.completer is None else self.completer.working,
            None if self.completer is None else self.completer.candidate,
            self.stage,
            self.completer is not None and self.completer.publishing,
        )

    def stopped(self, error):
        values = BaseException.__dict__["__dict__"].__get__(error)
        failure = self.snapshot()
        return SeriesCellStop(
            self.stage,
            attempt=self.attempt,
            descriptor_bytes=(
                None if self.selected is None else self.selected.descriptor_bytes
            ),
            binding_bytes=self.binding_bytes,
            pair_intent_bytes=self.pair_intent_bytes,
            observation=self.observation,
            progress_bytes=values.get("progress"),
            candidate=failure.candidate,
            publishing=failure.publishing,
        )


def reject(state, error):
    if state.original is not None and not isinstance(state.original, Exception):
        error = state.original
    carry_failure_context(error, state.original)
    try:
        if state.ledger_checked:
            state.admissions.stop_cell(state.stopped(error))
    except BaseException as later:
        if isinstance(error, Exception) and not isinstance(later, Exception):
            carry_failure_context(later, error)
            error = later
    original.reject(state, error)
