"""Retained-only preparation under a live, separately authorized adopted root."""

import json
from hashlib import sha256
from pathlib import Path

from . import _study_run_body as original
from . import _study_run_schema as schema
from ._prepared_external_runtime import held_preparation
from ._study_run_context import require
from ._study_urlnorm_completion import derivation
from .retained_study_derivation import run_retained_study_preparation
from .retained_study_hold import hold_prior_study_hold
from .study_preparation_transport import hold_study_preparation


def _prior(state, cleanup, continuation):
    held = original.enter(cleanup, hold_prior_study_hold(state.binding, continuation))
    prior = original.enter(
        cleanup,
        hold_study_preparation(
            Path(held.profile["paths"]["preparation-attempt"]),
            expected_identity=held.expected_identity,
            expected_reservation_sha256=continuation[
                "prior_preparation_reservation_sha256"
            ],
            expected_completion_sha256=continuation[
                "prior_preparation_complete_sha256"
            ],
            source_spec_bytes=held.source_spec_bytes,
            preparation_summary_bytes=held.preparation_summary_bytes,
        ),
    )
    content = prior.payload("feasibility.json")
    require(sha256(content).hexdigest() == held.barrier["feasibility_sha256"])
    schema.same(prior.feasibility, held.barrier["feasibility"])
    return prior


def _retain(state, cleanup, continuation):
    completion = sha256(state.fresh.payload("preparation-complete.json")).hexdigest()
    state.stage = "preparation_retention"
    state.preparation = original.enter(
        cleanup,
        held_preparation(
            state.binding,
            state.paths.external,
            state.fresh.reservation_sha256,
            completion,
        ),
    )
    require(state.preparation.payloads == state.fresh.payloads)
    require(state.preparation.reservation_sha256 == state.fresh.reservation_sha256)
    require(state.preparation.completion_sha256 == completion)
    complete = json.loads(state.preparation.payload("preparation-complete.json"))
    require(type(complete["schema_version"]) is int and complete["schema_version"] == 2)
    require(complete["protocol"] == "study-preparation-derived-v1")
    schema.same(complete.get("derivation"), derivation(continuation))


def prepare(state, cleanup):
    state.stage = "preparation"
    continuation = json.loads(state.authorization.profile_bytes)["continuation"]
    prior = _prior(state, cleanup, continuation)
    state.fresh = run_retained_study_preparation(
        state.binding,
        state.paths.preparation,
        prior_preparation=prior,
        continuation=continuation,
    )
    _retain(state, cleanup, continuation)
    state.outputs.check(())
    state.stage = "prediction_barrier"
    barrier, state.held = original.records.prediction_barrier(
        state.preparation, execution=state.execution
    )
    state.writer.append("prediction-barrier.json", barrier)
