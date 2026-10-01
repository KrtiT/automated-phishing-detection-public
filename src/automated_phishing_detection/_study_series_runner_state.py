"""Actual root progress and candidates, without observed-source type coercion."""

from dataclasses import dataclass, field


class SeriesRunError(ValueError):
    """Series execution stopped; private failure facts remain separately attached."""


@dataclass(frozen=True)
class SeriesRunFailure:
    stage: str
    series_attempt: object
    segment_attempt: object
    completed: tuple = field(repr=False)
    segment_accounting_bytes: object = field(repr=False)
    series_accounting_bytes: object = field(repr=False)
    segment_candidate: object = field(repr=False)
    series_candidate: object = field(repr=False)


@dataclass(frozen=True)
class SeriesRunResult:
    series_attempt: object
    segment_attempt: object
    segment_candidate: tuple = field(repr=False)
    series_candidate: tuple = field(repr=False)
    segment_accounting_bytes: bytes = field(repr=False)
    series_accounting_bytes: bytes = field(repr=False)
    completed: tuple = field(repr=False)
    reduced: object = field(repr=False)


class SeriesProgress:
    def __init__(self, public, profile, lifecycle_check):
        self.public, self.profile, self.lifecycle_check = (
            public,
            profile,
            lifecycle_check,
        )
        self.stage, self.original = "validation", None
        self.history = self.outputs = self.ledger = None
        self.series_attempt = self.segment_attempt = None
        self.series_writer = self.segment_writer = None
        self.metadata_bytes = self.import_bytes = self.intent_bytes = None
        self.segment_accounting_bytes = self.series_accounting_bytes = None
        self.completed, self.reduced = [], None

    def reserve_series(self, path, identity):
        from .execution_receipt import reserve_attempt

        self.series_attempt = reserve_attempt(path, identity=identity)

    def reserve_segment(self, path, identity):
        from .execution_receipt import reserve_attempt

        self.segment_attempt = reserve_attempt(path, identity=identity)

    def snapshot(self):
        return SeriesRunFailure(
            self.stage,
            self.series_attempt,
            self.segment_attempt,
            tuple(self.completed),
            self.segment_accounting_bytes,
            self.series_accounting_bytes,
            None if self.segment_writer is None else self.segment_writer.candidate,
            None if self.series_writer is None else self.series_writer.candidate,
        )


def series_run_failure(error):
    seen = set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        values = BaseException.__dict__["__dict__"].__get__(error)
        if type(values.get("study_failure")) is SeriesRunFailure:
            return values["study_failure"]
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return None
