"""Continue only the approved prepared source bodies under held study admission."""

from functools import partial

from ._prepared_external_runtime import run_held_preparation
from ._study_child_context import recheck_held_child, require
from .prepared_internal_runner import _run_held
from .study_preparation_context import bound_preparation_context


def run_internal(held):
    authorization, frame = held.authorization, held.admission.frame
    context = bound_preparation_context(authorization.base)
    return _run_held(
        authorization.base,
        authorization.paths.internal,
        context,
        frame.preparation_reservation_sha256,
        frame.preparation_completion_sha256,
        False,
        lifecycle_check=partial(recheck_held_child, held),
    )


def run_external(held, arguments):
    authorization, frame = held.authorization, held.admission.frame
    require(arguments.expected_handoff_sha256 == frame.predecessor_sha256)
    return run_held_preparation(
        authorization.base,
        authorization.paths.external,
        arguments.internal_transport,
        arguments.expected_handoff_sha256,
        frame.preparation_reservation_sha256,
        frame.preparation_completion_sha256,
        lifecycle_check=partial(recheck_held_child, held),
    )
