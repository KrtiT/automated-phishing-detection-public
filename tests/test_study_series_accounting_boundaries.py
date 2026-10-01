"""Saved validation cannot create live facts, access files or alter history."""

import base64
import json

import pytest
from study_series_accounting_fixtures import (
    accounting_case,
    api,
    completed_accounting,
    prefix_arguments,
    series,
    verify_segment,
)
from study_series_ledger_fixtures import candidates, manifests, series_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._operational_process_records import ProcessObservation
from automated_phishing_detection._study_history_snapshot_records import digest

__all__ = [
    "accounting_case",
    "completed_accounting",
    "candidates",
    "manifests",
    "series_case",
]


def test_rehashed_series_cannot_drop_failed_history(
    accounting_case, completed_accounting
):
    case = accounting_case
    value = json.loads(series(case, segment_bytes=completed_accounting))
    value["prior_attempts"].pop()
    changed = canonical_bytes(value)
    with pytest.raises(api().SeriesAccountingError):
        api().verify_series_accounting(
            changed,
            case.prefix.binding,
            case.prefix.series,
            case.index,
            expected_sha256=digest(changed),
            segment_attempt=case.prefix.segment,
            segment_accounting_bytes=completed_accounting,
            **prefix_arguments(case.prefix),
        )


def stopped(value):
    value.update(status="failed", stage="cell_execution")
    slot = value["ledger"]["cells"][0]
    slot.update(status="stopped", stage="finalization", holders_closed=False)
    for entry in value["ledger"]["admissions"]:
        entry["accepted"] = False
    return slot


def test_stopped_candidate_requires_retained_binding(
    accounting_case, completed_accounting
):
    value = json.loads(completed_accounting)
    slot = stopped(value)
    slot.update(binding_bytes=None, observation_bytes=None, pair_intent_bytes=None)
    value["ledger"]["admissions"] = []
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(accounting_case.prefix, canonical_bytes(value))


def test_observed_stopped_record_cannot_claim_forced_clean_exit(
    accounting_case, completed_accounting
):
    from automated_phishing_detection._operational_process_records import _bytes

    value = json.loads(completed_accounting)
    slot = stopped(value)
    observation = json.loads(base64.b64decode(slot["observation_bytes"]))
    observation["service"].update(forced=True, signals=[15])
    content = _bytes(observation)
    slot["observation_bytes"] = base64.b64encode(content).decode("ascii")
    for name in ("attempt/process-pair.json", "attempt/evidence/process-pair.json"):
        slot["snapshot_sha256"][name] = digest(content)
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(accounting_case.prefix, canonical_bytes(value))


def test_saved_validation_uses_no_io_or_observation_constructor(
    accounting_case, completed_accounting, monkeypatch
):
    def forbidden(*unused, **ignored):
        raise AssertionError("saved validator crossed live or IO boundary")

    with monkeypatch.context() as patch:
        patch.setattr("builtins.open", forbidden)
        patch.setattr("os.open", forbidden)
        patch.setattr("subprocess.Popen", forbidden)
        patch.setattr(ProcessObservation, "__init__", forbidden)
        value = verify_segment(accounting_case.prefix, completed_accounting)
        content = series(
            accounting_case, status="complete", segment_bytes=completed_accounting
        )
    assert value["status"] == json.loads(content)["status"] == "complete"


@pytest.mark.parametrize("content", [b'{"a":1,"a":2}\n', b'{"a":NaN}\n', b"{}"])
def test_noncanonical_or_duplicate_json_fails(accounting_case, content):
    with pytest.raises(api().SeriesAccountingError):
        verify_segment(accounting_case.prefix, content)
