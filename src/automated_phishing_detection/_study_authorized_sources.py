"""Reuse prepared source lifecycles through one fresh root's live admissions."""

from .external_source_process import _run_observed_external
from .internal_external_handoff import build_internal_handoff
from .prepared_external_process import (
    ObservedPreparedSourceCompletion,
    _retain_completed,
    _retain_failure,
)
from .prepared_internal_runner import _run_observed_prepared_internal
from .study_execution import recheck_study_execution


def _validate(authorization, admissions):
    from ._adopted_study_ledger import AdmissionLedger
    from .study_execution import StudyExecutionBinding

    if type(authorization) is not StudyExecutionBinding:
        raise ValueError("invalid_study_source_authorization")
    if (
        type(admissions) is not AdmissionLedger
        or admissions.authorization is not authorization
    ):
        raise ValueError("invalid_study_source_admissions")


def _sources(authorization, preparation, admissions):
    internal = _run_observed_prepared_internal(
        authorization.base,
        authorization.paths.internal,
        preparation=preparation,
        study_admissions=admissions,
    )
    try:
        admissions.internal_accepted(internal)
        handoff = build_internal_handoff(internal)
        external = _run_observed_external(
            authorization.base,
            authorization.paths.external,
            handoff,
            preparation=preparation,
            study_admissions=admissions,
        )
        completed = ObservedPreparedSourceCompletion(
            preparation, internal, external, handoff
        )
        try:
            admissions.external_accepted(external)
            recheck_study_execution(authorization)
        except BaseException as error:
            _retain_completed(error, authorization.base, completed)
            raise
        return completed
    except BaseException as error:
        _retain_failure(error, preparation, internal)
        raise


def run_adopted_sources(authorization, preparation, admissions):
    _validate(authorization, admissions)
    try:
        recheck_study_execution(authorization)
        return _sources(authorization, preparation, admissions)
    except BaseException as error:
        _retain_failure(error, preparation, None)
        raise
