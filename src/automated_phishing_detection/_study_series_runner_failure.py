"""Never replace publication claims or the first interruption during root cleanup."""

from functools import partial

from . import _study_preparation_files as files
from . import _study_series_runner_records as records
from . import execution_receipt as receipt
from ._prepared_failure_context import carry_failure_context
from ._study_run_failure import select
from ._study_series_runner_state import SeriesRunError
from .internal_failure import failure_kind


def failed_accounting(state, error):
    if state.original is None:
        state.original = error
    try:
        records.account(state, "failed")
        records.append_accounting(state, failure=True)
    except BaseException as later:
        selected = select(error, later)
        if selected is not error:
            carry_failure_context(selected, error)
            raise selected from None


def persist(state, error):
    for role in ("segment", "series"):
        attempt, writer = (
            getattr(state, f"{role}_attempt"),
            getattr(state, f"{role}_writer"),
        )
        if attempt is None or (writer is not None and writer.held.publishing):
            continue
        try:
            files.deferred(
                partial(
                    receipt.record_failure,
                    attempt,
                    stage=state.stage,
                    error_type=failure_kind(error),
                )
            )
        except BaseException as later:
            selected = select(error, later)
            carry_failure_context(selected, error)
            error = selected
    return error


def reject(state, error):
    if state.original is not None and not isinstance(state.original, Exception):
        error = state.original
    carry_failure_context(error, state.original)
    try:
        failed_accounting(state, error)
    except BaseException as later:
        error = select(error, later)
    error = persist(state, error)
    if isinstance(error, Exception):
        selected = SeriesRunError("series_execution_failed")
        carry_failure_context(selected, error)
        error = selected
    BaseException.__dict__["__dict__"].__get__(error)["study_failure"] = (
        state.snapshot()
    )
    raise error from None
