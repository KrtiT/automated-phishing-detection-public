"""Root public assembly links exact private buffers, without execution authority."""

import json
from hashlib import sha256

import pytest
from study_run_record_fixtures import api, prepared

from automated_phishing_detection import _study_root_records as retention
from automated_phishing_detection import study_operational_records as cells
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["prepared"]


def hold_checkpoints(prepared):
    module = api()
    barrier, held = module.prediction_barrier(
        prepared.preparation, execution=prepared.execution
    )
    assert held
    deadlines = dict(startup=10.0, shutdown=5.0, terminate=2.0, kill=1.0)
    return (
        (
            "study-intent.json",
            module.study_intent(
                prepared.binding, prepared.profile, prepared.attempt, deadlines
            ),
        ),
        ("prediction-barrier.json", barrier),
        (
            "study-accounting.json",
            module.study_accounting(
                cells.freeze_cell_accounting(()),
                execution=prepared.execution,
                stage="prediction_barrier",
                status="whole_study_hold",
                internal_status="unattempted",
                external_status="unattempted",
            ),
        ),
    )


def test_hold_public_has_exact_three_private_hashes_and_original_feasibility(prepared):
    checkpoints = hold_checkpoints(prepared)
    result = api().root_public_summary(
        execution=prepared.execution, checkpoints=checkpoints
    )
    expected = {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "whole_study_hold",
        "execution": prepared.execution,
        "accounting_sha256": sha256(
            dict(checkpoints)["study-accounting.json"]
        ).hexdigest(),
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in checkpoints
        },
        "feasibility": prepared.preparation.feasibility,
    }
    assert result == expected
    identity = {
        name: value
        for name, value in prepared.execution.items()
        if name != "reservation_sha256"
    }
    retention.public_bytes(
        prepared.attempt, identity, dict(checkpoints), dict(checkpoints), False, result
    )
    assert not {"operational", "study", "cells", "accepted_inputs"} & result.keys()


@pytest.mark.parametrize(
    "change", ["order", "extra", "duplicate", "mutable", "mapping"]
)
def test_public_requires_original_closed_checkpoint_inventory(prepared, change):
    checkpoints = hold_checkpoints(prepared)
    if change == "order":
        checkpoints = tuple(reversed(checkpoints))
    elif change == "extra":
        checkpoints += (("unknown.json", b"{}\n"),)
    elif change == "duplicate":
        checkpoints += (checkpoints[-1],)
    elif change == "mapping":
        checkpoints = dict(checkpoints)
    else:
        checkpoints = (
            (checkpoints[0][0], bytearray(checkpoints[0][1])),
            *checkpoints[1:],
        )
    with pytest.raises(ValueError):
        api().root_public_summary(execution=prepared.execution, checkpoints=checkpoints)


@pytest.mark.parametrize(
    "member",
    [
        "schema_version",
        "protocol",
        "execution",
        "status",
        "feasibility_sha256",
        "predictions_started",
    ],
)
def test_forged_barrier_cannot_reach_public_hold(prepared, member):
    checkpoints = list(hold_checkpoints(prepared))
    value = json.loads(checkpoints[1][1])
    value[member] = {
        "schema_version": True,
        "protocol": "other",
        "execution": {},
        "status": "necessary_capacity_present",
        "feasibility_sha256": "f" * 64,
        "predictions_started": True,
    }[member]
    checkpoints[1] = ("prediction-barrier.json", canonical_bytes(value))
    with pytest.raises(ValueError):
        api().root_public_summary(
            execution=prepared.execution, checkpoints=tuple(checkpoints)
        )


def test_public_json_views_do_not_mutate_retained_checkpoints(prepared):
    checkpoints = hold_checkpoints(prepared)
    first = api().root_public_summary(
        execution=prepared.execution, checkpoints=checkpoints
    )
    first["feasibility"]["shortages"].clear()
    second = api().root_public_summary(
        execution=prepared.execution, checkpoints=checkpoints
    )
    assert second["feasibility"]["shortages"]
