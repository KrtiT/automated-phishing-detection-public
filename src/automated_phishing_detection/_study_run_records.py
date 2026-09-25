"""Pure root records preserve admitted inputs, never grant access or acceptance."""

from hashlib import sha256

from . import _operational_profile as profiles
from . import _study_root_records as retention
from . import _study_run_schema as schema
from . import study_operational_records as cells
from ._checkpoint_codec import canonical_bytes
from .execution_preflight import ExecutionBinding
from .operational_inputs import AcceptedOperationalInputs
from .retained_study_preparation import RestoredStudyPreparation


def study_identity(binding, profile) -> dict:
    schema.require(type(binding) is ExecutionBinding)
    schema.require(type(profile) is profiles.CandidateOperationalProfile)
    schema.require(type(profile.canonical_bytes) is bytes)
    projection = profiles._projection(binding)
    schema.require(profile.canonical_bytes == canonical_bytes(projection))
    result = {
        "kind": "whole_study",
        "protocol": schema.PROTOCOL,
        **projection["execution"],
        "operational_profile_sha256": profile.profile_sha256,
    }
    schema.execution(result, reserved=False)
    return result


def study_intent(binding, profile, attempt, deadlines) -> bytes:
    identity = study_identity(binding, profile)
    retention.reservation(attempt, identity)
    schema.deadlines(deadlines)
    execution = identity | {"reservation_sha256": attempt.reservation_sha256}
    return canonical_bytes(
        schema.envelope("intent", execution)
        | {
            "operational_profile": profile.projection(),
            "protective_deadlines_seconds": deadlines,
        }
    )


def _preparation(preparation, execution):
    schema.require(type(preparation) is RestoredStudyPreparation)
    schema.execution(execution)
    schema.operational.digest(preparation.reservation_sha256)
    schema.operational.digest(preparation.completion_sha256)
    complete = preparation.payload("preparation-complete.json")
    schema.require(sha256(complete).hexdigest() == preparation.completion_sha256)
    completion = schema.load(complete)
    schema.require(completion["reservation_sha256"] == preparation.reservation_sha256)
    schema.same(
        {name: completion["execution"][name] for name in schema.EXECUTION},
        {name: execution[name] for name in schema.EXECUTION},
    )
    content = preparation.payload("feasibility.json")
    schema.require(
        sha256(content).hexdigest() == completion["input_sha256"]["feasibility.json"]
    )
    feasibility = schema.load(content)
    schema.feasibility(feasibility)
    return content, feasibility


def prediction_barrier(preparation, *, execution) -> tuple[bytes, bool]:
    content, feasibility = _preparation(preparation, execution)
    held = bool(feasibility["shortages"])
    status = "whole_study_hold" if held else "necessary_capacity_present"
    result = schema.envelope(status, execution) | {
        "study_preparation_reservation_sha256": preparation.reservation_sha256,
        "study_preparation_complete_sha256": preparation.completion_sha256,
        "feasibility": feasibility,
        "feasibility_sha256": sha256(content).hexdigest(),
        "predictions_started": False,
    }
    return canonical_bytes(result), held


def source_results(accepted, *, execution) -> bytes:
    schema.execution(execution)
    schema.require(type(accepted) is AcceptedOperationalInputs)
    metadata = schema.load(accepted.metadata_bytes)
    schema.accepted_metadata(metadata, execution)
    return canonical_bytes(
        schema.envelope("sources_accepted", execution)
        | {
            "accepted_inputs": metadata,
            "accepted_inputs_sha256": sha256(accepted.metadata_bytes).hexdigest(),
        }
    )


def study_accounting(
    slots, *, execution, stage, status, internal_status, external_status
) -> bytes:
    result = schema.envelope(status, execution) | {
        "stage": stage,
        "internal_status": internal_status,
        "external_status": external_status,
        "cells": cells.cell_accounting_projection(slots),
    }
    schema.accounting(result)
    return canonical_bytes(result)


def root_public_summary(*, execution, checkpoints, reduced=None) -> dict:
    from ._study_run_public import public_summary

    return public_summary(execution, checkpoints, reduced)
