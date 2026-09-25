"""Caller-declared complete metadata only, not 125 historical process observations."""

import json
from dataclasses import asdict
from hashlib import sha256

from operational_cell_runner_fixtures import accepted_source
from operational_input_fixtures import build
from study_run_record_fixtures import api, attempt, capacity

from automated_phishing_detection import operational_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_cell_protocol import SNAPSHOT_NAMES
from automated_phishing_detection.operational_schedule import planned_cells
from automated_phishing_detection.study_reduction import ReducedStudyBytes


def complete_projection():
    result = []
    for cell in planned_cells():
        reservation = sha256(f"reservation{cell.ordinal}".encode()).hexdigest()
        observation = sha256(f"observation{cell.ordinal}".encode()).hexdigest()
        hashes = {name: "f" * 64 for name in SNAPSHOT_NAMES}
        hashes["attempt/reservation.json"] = reservation
        hashes["attempt/process-pair.json"] = observation
        result.append(
            {
                "cell": asdict(cell),
                "status": "accepted",
                "retention": "compact",
                "snapshot_sha256": hashes,
                "observation_sha256": observation,
                "stage": None,
                "reservation_sha256": reservation,
                "progress_sha256": None,
                "publishing": None,
            }
        )
    return result


def accepted_context(manifests):
    source, profile, unused = accepted_source(manifests)
    identity = api().study_identity(source.binding, profile)
    reserved = attempt(identity)
    source.reservation = reserved.reservation_sha256
    accepted = build(operational_inputs, source)
    execution = identity | {"reservation_sha256": reserved.reservation_sha256}
    return source, profile, reserved, accepted, execution


def success_barrier(prepared, accepted, execution):
    barrier, unused = api().prediction_barrier(
        capacity(prepared), execution=prepared.execution
    )
    value = json.loads(barrier)
    original = json.loads(accepted.metadata_bytes)["internal"]["execution"]
    value.update(
        execution=execution,
        **{
            name: original[name]
            for name in (
                "study_preparation_reservation_sha256",
                "study_preparation_complete_sha256",
            )
        },
    )
    return canonical_bytes(value)


def success_accounting(execution):
    return canonical_bytes(
        {
            "schema_version": 1,
            "protocol": "study-root-v1",
            "status": "matrix_accepted",
            "execution": execution,
            "stage": "reduction",
            "internal_status": "accepted",
            "external_status": "accepted",
            "cells": complete_projection(),
        }
    )


def success_case(prepared, manifests):
    source, profile, reserved, accepted, execution = accepted_context(manifests)
    intent = api().study_intent(
        source.binding,
        profile,
        reserved,
        dict(startup=10.0, shutdown=5.0, terminate=2.0, kill=1.0),
    )
    checkpoints = (
        ("study-intent.json", intent),
        ("prediction-barrier.json", success_barrier(prepared, accepted, execution)),
        ("source-results.json", api().source_results(accepted, execution=execution)),
        ("study-accounting.json", success_accounting(execution)),
    )
    reduced = ReducedStudyBytes(
        canonical_bytes({"groups": []}), canonical_bytes({"primary": {}})
    )
    return execution, checkpoints, reduced, reserved
