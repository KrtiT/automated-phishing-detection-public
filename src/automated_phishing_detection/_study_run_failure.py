"""Consume failed roots once, retaining actual state and interruption precedence."""

from functools import partial

from . import _study_preparation_files as files
from . import execution_receipt as receipt
from ._prepared_failure_context import carry_failure_context
from ._study_run_context import StudyRunError
from .internal_failure import failure_kind


def attach(error, state):
    try:
        values = BaseException.__dict__["__dict__"].__get__(error)
        if not dict.__contains__(values, "study_failure"):
            dict.__setitem__(values, "study_failure", state.snapshot(error))
    except BaseException as later:
        selected = select(error, later)
        carry_failure_context(selected, error)
        return selected
    return error


def select(original, later):
    if not isinstance(original, Exception):
        return original
    return later if not isinstance(later, Exception) else original


def _persist(state, error):
    if state.attempt is not None and not state.publishing:
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
    error = attach(error, state)
    try:
        _persist(state, error)
    except BaseException as later:
        selected = select(error, later)
        carry_failure_context(selected, error)
        error = attach(selected, state)
    if not isinstance(error, Exception):
        raise error from None
    rejected = StudyRunError("study_execution_failed")
    carry_failure_context(rejected, error)
    raise attach(rejected, state) from None
