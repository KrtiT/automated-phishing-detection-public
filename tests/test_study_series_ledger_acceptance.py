"""Only exact current science, full publication, callbacks and actual observation join."""

from dataclasses import replace

import pytest
from study_series_ledger_fixtures import (
    api,
    candidates,
    ledger,
    ledger_cell,
    manifests,
    observed,
    prefix_case,
    series_case,
    snapshot,
    start,
)
from test_study_series_ledger_ordering import rejection

from automated_phishing_detection._operational_process_records import (
    ProcessObservation,
)

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


def ready(selected):
    current = ledger(selected.prefix)
    start(current, selected)
    observed(current, selected)
    return current


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "wrong_type",
        "reservation",
        "summary",
        "origin",
        "descriptor",
        "binding",
        "metadata",
        "manifest",
    ],
)
def test_candidate_exact_type_and_all_input_bytes_match_parent(ledger_cell, change):
    candidate = ledger_cell.candidate
    if change in ("none", "wrong_type"):
        candidate = None if change == "none" else object()
    elif change == "reservation":
        candidate = replace(candidate, reservation_sha256="0" * 64)
    elif change == "summary":
        candidate = replace(candidate, summary_bytes=b"{}\n")
    elif change == "origin":
        candidate = replace(
            candidate, inputs=replace(candidate.inputs, origin_metadata_bytes=b"{}\n")
        )
    else:
        field = {"metadata": "accepted_bytes"}.get(change, f"{change}_bytes")
        computational = replace(
            candidate.inputs.computational, **{field: b"wrong bytes"}
        )
        candidate = replace(
            candidate, inputs=replace(candidate.inputs, computational=computational)
        )
    current = ready(ledger_cell)
    with rejection():
        current.accept_cell(candidate, ledger_cell.observation, ledger_cell.pair)
    current.stop_cell(api().SeriesCellStop("finalization"))
    assert all(not entry["accepted"] for entry in snapshot(current)["admissions"])


@pytest.mark.parametrize(
    "operation",
    [
        "missing",
        "extra",
        "duplicate",
        "mutable",
        "list",
        "private_copy",
        "public",
        "claim",
        "outcome",
    ],
)
def test_full_36_publication_is_joined_before_any_accepted_entry(
    ledger_cell, operation
):
    payloads = ledger_cell.candidate.payloads
    values = dict(payloads)
    if operation in ("private_copy", "public", "claim", "outcome"):
        name = {
            "private_copy": "attempt/evidence/run.json",
            "public": "public-summary.json",
            "claim": "attempt/finalize.claim",
            "outcome": "attempt/outcome.json",
        }[operation]
        values[name] = b"changed"
        payloads = tuple(values.items())
    else:
        payloads = {
            "missing": payloads[1:],
            "extra": (*payloads, ("extra", b"extra")),
            "duplicate": (*payloads, payloads[0]),
            "mutable": ((payloads[0][0], bytearray(payloads[0][1])), *payloads[1:]),
            "list": list(payloads),
        }[operation]
    current = ready(ledger_cell)
    with rejection():
        current.accept_cell(
            replace(ledger_cell.candidate, payloads=payloads),
            ledger_cell.observation,
            ledger_cell.pair,
        )


@pytest.mark.parametrize(
    "change",
    [
        "wrong_type",
        "changed_record",
        "mutable_record",
        "pair_missing",
        "pair_changed",
        "pair_mutable",
    ],
)
def test_actual_observation_and_intent_cannot_be_substituted(ledger_cell, change):
    observation, pair = ledger_cell.observation, ledger_cell.pair
    if change == "wrong_type":
        observation = object()
    elif change in ("changed_record", "mutable_record"):
        observation = ProcessObservation(
            b"changed" if change == "changed_record" else bytearray(observation.record)
        )
    else:
        pair = {
            "pair_missing": None,
            "pair_changed": b"changed",
            "pair_mutable": bytearray(pair),
        }[change]
    with rejection():
        ready(ledger_cell).accept_cell(ledger_cell.candidate, observation, pair)
