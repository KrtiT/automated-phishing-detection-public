"""Bind copied preparation buffers to independently authorized historical pins."""

import json
from hashlib import sha256

from . import _study_preparation_body as body
from . import _study_run_schema as schema
from .retained_study_hold import _continuation
from .retained_study_preparation import RestoredStudyPreparation
from .study_preparation_retention import PREPARATION_ORDER


def prior_context(prior, continuation):
    profile, unused = _continuation(continuation)
    schema.require(type(prior) is RestoredStudyPreparation)
    schema.require(
        prior.reservation_sha256 == continuation["prior_preparation_reservation_sha256"]
    )
    schema.require(
        prior.completion_sha256 == continuation["prior_preparation_complete_sha256"]
    )
    for name, key in (
        ("preparation-complete.json", "prior_preparation_complete_sha256"),
        ("publisher-source.json", "publisher_source_sha256"),
        ("publisher-summary.json", "publisher_summary_sha256"),
    ):
        schema.require(sha256(prior.payload(name)).hexdigest() == continuation[key])
    complete = json.loads(prior.payload("preparation-complete.json"))
    schema.require(type(complete["schema_version"]) is int)
    schema.require(complete["schema_version"] == 1)
    schema.require(complete["protocol"] == "study-preparation-v1")
    schema.require(complete["reservation_sha256"] == prior.reservation_sha256)
    schema.same(
        complete["input_sha256"],
        {
            name: sha256(prior.payload(name)).hexdigest()
            for name in PREPARATION_ORDER[:-1]
        },
    )
    return profile


def preflight(state, prior, continuation):
    profile = prior_context(prior, continuation)
    with body.deferred_io():
        state.profile = body.resolve_external_source_profile(state.binding)
        state.identity, state.source, state.source_buffers, unused = (
            body.context_for_profile(state.binding, state.profile)
        )
    schema.same(
        prior.execution,
        state.identity
        | {
            "revision": profile["execution"]["revision"],
            "source_profile_sha256": profile["components"]["external"],
        },
    )
    for name, path in (
        ("source-csv", state.paths.source_csv),
        ("suffix-rules", state.paths.suffix_rules),
        ("archive", state.paths.archive),
    ):
        schema.require(str(path) == profile["paths"][name])
    body._paths(state.binding, state.paths)
    body._reserve(state)
