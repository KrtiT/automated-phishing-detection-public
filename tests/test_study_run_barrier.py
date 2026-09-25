"""Whole-study hold is administrative; neither branch grants prediction authority."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from study_run_record_fixtures import (
    REQUIREMENTS,
    api,
    branch,
    capacity,
    isolated_shortage,
    prepared,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["prepared"]


def test_genuinely_restored_preparation_preserves_original_feasibility(prepared):
    content, held = api().prediction_barrier(
        prepared.preparation, execution=prepared.execution
    )
    value = json.loads(content)
    assert held is True
    assert value == {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "whole_study_hold",
        "execution": prepared.execution,
        "study_preparation_reservation_sha256": prepared.preparation.reservation_sha256,
        "study_preparation_complete_sha256": prepared.preparation.completion_sha256,
        "feasibility": prepared.preparation.feasibility,
        "feasibility_sha256": sha256(
            prepared.preparation.payload("feasibility.json")
        ).hexdigest(),
        "predictions_started": False,
    }
    assert canonical_bytes(value["feasibility"]) == prepared.preparation.payload(
        "feasibility.json"
    )


@pytest.mark.parametrize("requirement", REQUIREMENTS)
def test_each_original_shortage_holds_every_prediction(prepared, requirement):
    preparation = isolated_shortage(prepared, requirement)
    content, held = api().prediction_barrier(preparation, execution=prepared.execution)
    value = json.loads(content)
    assert held is True and value["predictions_started"] is False
    assert value["status"] == "whole_study_hold"
    assert value["feasibility"]["shortages"][0]["requirement"] == requirement


def test_empty_shortages_mean_necessary_capacity_not_success(prepared):
    content, held = api().prediction_barrier(
        capacity(prepared), execution=prepared.execution
    )
    value = json.loads(content)
    assert held is False
    assert value["status"] == "necessary_capacity_present"
    assert value["predictions_started"] is False
    assert (
        "hypotheses" not in value and "authorized" not in value and "ready" not in value
    )


@pytest.mark.parametrize(
    "field",
    ["revision", "execution_contract_sha256", "runtime_sha256", "source_spec_sha256"],
)
def test_barrier_joins_preparation_to_same_execution(prepared, field):
    execution = prepared.execution | {field: "0" * (40 if field == "revision" else 64)}
    with pytest.raises(ValueError):
        api().prediction_barrier(prepared.preparation, execution=execution)


@pytest.mark.parametrize(
    "change",
    ["count_bool", "negative", "protocol", "extra", "duplicate", "noncanonical"],
)
def test_invalid_feasibility_is_not_reclassified_as_shortage(prepared, change):
    preparation = prepared.preparation
    payloads = dict(preparation.payloads)
    value = json.loads(payloads["feasibility.json"])
    if change in ("count_bool", "negative"):
        value["counts"]["internal"]["retained_rows"] = (
            True if change == "count_bool" else -1
        )
    elif change == "protocol":
        value["protocol"] = "other"
    elif change == "extra":
        value["authorized"] = True
    elif change == "duplicate":
        value["shortages"].append(value["shortages"][0])
    content = canonical_bytes(value)
    altered = branch(prepared, value)
    if change == "noncanonical":
        payloads = dict(altered.payloads) | {"feasibility.json": content + b" "}
        altered = replace(altered, payloads=tuple(payloads.items()))
    with pytest.raises(ValueError):
        api().prediction_barrier(altered, execution=prepared.execution)


def test_barrier_never_reprepares_restores_or_reassesses(prepared, monkeypatch):
    from automated_phishing_detection import (
        phishvn,
        retained_study_preparation,
        study_feasibility,
    )

    def forbidden(*arguments, **keywords):
        pytest.fail("pure barrier repeated preparation or assessment")

    for owner, name in (
        (phishvn, "prepare_external_rows"),
        (study_feasibility, "assess_preparation_feasibility"),
        (retained_study_preparation, "restore_study_preparation"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    api().prediction_barrier(prepared.preparation, execution=prepared.execution)
