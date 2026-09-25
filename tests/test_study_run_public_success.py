"""Successful projection uses exact reducer bytes, never selected new arithmetic."""

import json
from dataclasses import replace
from hashlib import sha256

import pytest
from operational_input_fixtures import candidates, manifests
from study_run_public_fixtures import success_case
from study_run_record_fixtures import api, prepared

from automated_phishing_detection import _study_root_records as retention
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prepared"]


def test_success_public_matches_exact_six_output_retention_schema(prepared, manifests):
    execution, checkpoints, reduced, reserved = success_case(prepared, manifests)
    result = api().root_public_summary(
        execution=execution, checkpoints=checkpoints, reduced=reduced
    )
    outputs = dict(checkpoints) | {
        "operational-summary.json": reduced.operational_bytes,
        "study-evidence.json": reduced.study_bytes,
    }
    assert result == {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "study_evidence_published",
        "execution": execution,
        "accounting_sha256": sha256(checkpoints[-1][1]).hexdigest(),
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in outputs.items()
        },
        "operational": reduced.operational,
        "study": reduced.study,
    }
    identity = {
        name: value for name, value in execution.items() if name != "reservation_sha256"
    }
    retention.public_bytes(reserved, identity, dict(checkpoints), outputs, True, result)
    assert "feasibility" not in result and "cells" not in result


@pytest.mark.parametrize(
    "change",
    [
        "wrong_preparation",
        "wrong_source_digest",
        "hold_accounting",
        "short_matrix",
        "mutable_reduced",
        "noncanonical_reduced",
        "unpacked_fallback",
    ],
)
def test_success_cannot_relabel_incomplete_or_mismatched_inputs(
    prepared, manifests, change
):
    execution, original, reduced, unused = success_case(prepared, manifests)
    checkpoints = list(original)
    if change in ("mutable_reduced", "noncanonical_reduced"):
        content = (
            bytearray(reduced.study_bytes)
            if change == "mutable_reduced"
            else reduced.study_bytes + b" "
        )
        reduced = replace(reduced, study_bytes=content)
    else:
        change_checkpoint(checkpoints, change)
    with pytest.raises(ValueError):
        api().root_public_summary(
            execution=execution, checkpoints=tuple(checkpoints), reduced=reduced
        )


def change_checkpoint(checkpoints, change):
    position = (
        1
        if change == "wrong_preparation"
        else 2
        if change == "wrong_source_digest"
        else 3
    )
    value = json.loads(checkpoints[position][1])
    if change == "wrong_preparation":
        value["study_preparation_complete_sha256"] = "0" * 64
    elif change == "wrong_source_digest":
        value["accepted_inputs_sha256"] = "0" * 64
    elif change == "hold_accounting":
        value["status"] = "whole_study_hold"
    elif change == "unpacked_fallback":
        value["cells"][-1].update(
            retention="unpacked", snapshot_sha256=None, reservation_sha256=None
        )
    else:
        value["cells"].pop()
    checkpoints[position] = (checkpoints[position][0], canonical_bytes(value))


def test_public_does_not_call_reducers_or_replay(prepared, manifests, monkeypatch):
    from automated_phishing_detection import (
        operational_summary,
        study_evidence,
        study_reduction,
    )

    execution, checkpoints, reduced, unused = success_case(prepared, manifests)

    def forbidden(*arguments, **keywords):
        pytest.fail("pure public projection recomputed evidence")

    for owner, name in (
        (operational_summary, "summarize_operational_runs"),
        (study_evidence, "reduce_study_evidence"),
        (study_reduction, "reduce_accepted_study"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    api().root_public_summary(
        execution=execution, checkpoints=checkpoints, reduced=reduced
    )
