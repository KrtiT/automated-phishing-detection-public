"""Invented complete ledgers exercise every saved owned-observation join."""

import json
from hashlib import sha256

import pytest
from adopted_study_ledger_fixtures import complete_ledger
from operational_input_fixtures import candidates, manifests
from study_run_record_fixtures import prepared

from automated_phishing_detection import _adopted_study_records as records
from automated_phishing_detection._adopted_study_ledger_validation import (
    validate_ledger,
)
from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "manifests", "prepared"]


def test_exact_fixed_252_admissions_join_all_original_observations(prepared, manifests):
    case = complete_ledger(prepared, manifests)
    validate_ledger(case.ledger, case.execution, case.contents, True)


@pytest.mark.parametrize("position", [0, 1, 2, 3, 250, 251])
@pytest.mark.parametrize(
    "field", ["launched_pid", "exit_observed", "exit_code", "accepted"]
)
def test_no_substituted_owned_observation_accepts(prepared, manifests, position, field):
    case = complete_ledger(prepared, manifests)
    entry = case.ledger["admissions"][position]
    entry[field] = {
        "launched_pid": 999,
        "exit_observed": False,
        "exit_code": 1,
        "accepted": False,
    }[field]
    with pytest.raises(ValueError):
        validate_ledger(case.ledger, case.execution, case.contents, True)


def test_rehashed_command_frame_must_still_match_original_observation(
    prepared, manifests
):
    case = complete_ledger(prepared, manifests)
    entry = case.ledger["admissions"][2]
    value = json.loads(records.decoded(entry["frame_bytes"]))
    value["command_sha256"] = "0" * 64
    content = canonical_bytes(value).removesuffix(b"\n")
    entry.update(
        frame_bytes=records.encoded(content),
        frame_sha256=sha256(content).hexdigest(),
        command_sha256="0" * 64,
    )
    with pytest.raises(ValueError):
        validate_ledger(case.ledger, case.execution, case.contents, True)


def test_saved_ledger_cannot_omit_final_cell(prepared, manifests):
    case = complete_ledger(prepared, manifests)
    case.ledger["admissions"] = case.ledger["admissions"][:-2]
    case.ledger["cell_acceptances"].pop()
    with pytest.raises(ValueError):
        validate_ledger(case.ledger, case.execution, case.contents, True)


def repin(case, payload_key, file_name, value):
    content = canonical_bytes(value)
    case.ledger["cell_acceptances"][0][payload_key] = records.encoded(content)
    outer = json.loads(case.contents["study-accounting.json"])
    scientific = json.loads(records.decoded(outer["scientific_accounting_bytes"]))
    projection = scientific["cells"][0]
    for prefix in ("attempt", "attempt/evidence"):
        projection["snapshot_sha256"][f"{prefix}/{file_name}"] = sha256(
            content
        ).hexdigest()
    if payload_key == "observation_bytes":
        projection["observation_sha256"] = sha256(content).hexdigest()
    outer["scientific_accounting_bytes"] = records.encoded(canonical_bytes(scientific))
    case.contents["study-accounting.json"] = canonical_bytes(outer)


@pytest.mark.parametrize(
    "change", ["extra_observation_field", "forced", "deadline", "bool_ordinal"]
)
def test_rehashed_cell_evidence_preserves_original_closed_contract(
    prepared, manifests, change
):
    case = complete_ledger(prepared, manifests)
    bundle = case.ledger["cell_acceptances"][0]
    if change == "bool_ordinal":
        bundle["cell_ordinal"] = True
    elif change == "deadline":
        value = json.loads(records.decoded(bundle["pair_intent_bytes"]))
        value["deadlines"]["startup"] = 301
        repin(case, "pair_intent_bytes", "process-pair-intent.json", value)
    else:
        value = json.loads(records.decoded(bundle["observation_bytes"]))
        if change == "forced":
            value["service"]["forced"] = True
        else:
            value["unexpected"] = True
        repin(case, "observation_bytes", "process-pair.json", value)
    with pytest.raises(ValueError):
        validate_ledger(case.ledger, case.execution, case.contents, True)
