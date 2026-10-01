"""Exact process lifecycle checks receive the actual supplied observation object."""

import json
from dataclasses import replace

import pytest
from study_series_cell_fixtures import _publication
from study_series_ledger_fixtures import (
    candidates,
    ledger_cell,
    manifests,
    prefix_case,
    series_case,
)
from test_study_series_ledger_acceptance import ready
from test_study_series_ledger_ordering import rejection

from automated_phishing_detection._operational_cell_protocol import WORKING_NAMES
from automated_phishing_detection._operational_process_records import (
    ProcessObservation,
    _bytes,
)

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


def changed_process(selected, change):
    values = {
        name: dict(selected.candidate.payloads)[f"attempt/{name}"]
        for name in WORKING_NAMES
    }
    record = json.loads(selected.observation.record)
    role, field, value = change
    target = record if role is None else record[role]
    target[field] = value
    values["process-pair.json"] = _bytes(record)
    payloads = _publication(
        values,
        selected.inputs.computational,
        selected.attempt,
        selected.identity,
        selected.candidate.summary,
    )
    return replace(
        selected.candidate, payloads=tuple(payloads.items())
    ), ProcessObservation(values["process-pair.json"])


@pytest.mark.parametrize(
    "change",
    [
        ("service", "forced", True),
        ("client", "signals", ["terminate"]),
        ("service", "exit_observed", False),
        ("client", "exit_code", 1),
        ("service", "pid", 987),
        ("client", "pid", 321),
        (None, "failure", "parent_cancelled"),
        (None, "record_failures", ["run.json"]),
        (None, "status", "failed"),
        (None, "stop_sent", False),
    ],
)
def test_coherent_publication_still_rejects_bad_actual_process(ledger_cell, change):
    candidate, observation = changed_process(ledger_cell, change)
    with rejection():
        ready(ledger_cell).accept_cell(candidate, observation, ledger_cell.pair)


def test_real_observation_object_is_passed_unchanged_without_construction(
    ledger_cell, monkeypatch
):
    from automated_phishing_detection import (
        _study_series_ledger_acceptance as acceptance,
    )

    current = ready(ledger_cell)
    original = acceptance.process.verify_process_records
    calls = []

    def verify(*args, **kwargs):
        calls.append(kwargs["observation"])
        return original(*args, **kwargs)

    monkeypatch.setattr(acceptance.process, "verify_process_records", verify)
    monkeypatch.setattr(
        ProcessObservation,
        "__init__",
        lambda *args, **kwargs: pytest.fail("constructed historical observation"),
    )
    current.accept_cell(
        ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
    )
    assert calls == [ledger_cell.observation] and calls[0] is ledger_cell.observation
