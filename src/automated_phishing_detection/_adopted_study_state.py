"""Additional private authorization state preserves the existing failure snapshot."""

from dataclasses import dataclass, field

from . import _adopted_study_records as records
from ._adopted_study_ledger import AdmissionLedger
from ._prepared_failure_context import carry_failure_context
from ._study_run_failure import select
from ._study_run_state import StudyProgress


@dataclass(frozen=True)
class AdoptedStudyFailure:
    scientific: object = field(repr=False)
    authorization_ledger: bytes = field(repr=False)


class AdoptedStudyProgress(StudyProgress):
    def __init__(self, authorization):
        super().__init__(
            authorization.base,
            authorization.operational,
            authorization.paths,
            authorization.deadlines,
        )
        self.authorization = authorization
        self.admissions = None
        self.adopted_execution = None

    def reserve(self, identity):
        super().reserve(identity)
        self.adopted_execution = self.execution
        self.execution = records.scientific_execution(self.adopted_execution)
        self.intent = records.study_intent(self.authorization, self.attempt)
        self.admissions = AdmissionLedger(self.authorization, self.attempt, self.intent)


def adopted_study_failure(error):
    seen = set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        value = getattr(error, "adopted_study_failure", None)
        if type(value) is AdoptedStudyFailure:
            return value
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return None


def attach(error, state):
    try:
        if state.admissions is not None:
            error.adopted_study_failure = AdoptedStudyFailure(
                state.snapshot(error), state.admissions.snapshot()
            )
    except BaseException as later:
        selected = select(error, later)
        carry_failure_context(selected, error)
        return selected
    return error
