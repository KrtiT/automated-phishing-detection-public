"""Ordered live ledger callbacks never import historical PIDs or acceptance."""

from study_series_ledger_fixtures import (
    api,
    candidates,
    commands,
    ledger,
    ledger_cell,
    manifests,
    observed,
    prefix_case,
    series_case,
    snapshot,
    start,
)

__all__ = ["candidates", "manifests", "series_case", "prefix_case", "ledger_cell"]


def test_new_live_ledger_api_exists():
    assert callable(api().snapshot_series_ledger)


def test_initialized_ledger_contains_only_unattempted_fresh_suffix(prefix_case):
    value = snapshot(ledger(prefix_case))
    first = prefix_case.source.profile["segment"]["start_ordinal"]
    assert value["admissions"] == []
    assert [slot["ordinal"] for slot in value["cells"]] == list(range(first, 126))
    assert all(slot["status"] == "unattempted" for slot in value["cells"])
    assert all(
        all(
            member is None
            for key, member in slot.items()
            if key not in {"ordinal", "status"}
        )
        for slot in value["cells"]
    )


def test_actual_supplied_callbacks_and_observation_accept_one_cell(ledger_cell):
    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    observed(current, ledger_cell)
    current.accept_cell(
        ledger_cell.candidate, ledger_cell.observation, ledger_cell.pair
    )
    value = snapshot(current)
    assert [entry["accepted"] for entry in value["admissions"]] == [True, True]
    assert [entry["launched_pid"] for entry in value["admissions"]] == [321, 654]
    assert value["cells"][0]["status"] == "accepted"
    assert value["cells"][0]["holders_closed"] is True
    assert value["cells"][0]["publishing"] is True
    assert len(value["cells"][0]["snapshot_sha256"]) == 36
    assert all(slot["status"] == "unattempted" for slot in value["cells"][1:])


def test_service_only_stop_preserves_admission_and_prevents_next_issue(ledger_cell):
    import pytest

    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    with current.issue("service", commands(ledger_cell)[0]) as admission:
        admission.launched(321)
        admission.observed(True, 0)
    current.stop_cell(api().SeriesCellStop(stage="observation"))
    value = snapshot(current)
    assert value["cells"][0]["status"] == "stopped"
    assert (
        value["cells"][0]["reservation_sha256"]
        == ledger_cell.attempt.reservation_sha256
    )
    assert value["admissions"][0]["accepted"] is False
    with pytest.raises(api().SeriesLedgerError):
        current.issue("client", commands(ledger_cell)[1])


def test_both_roles_share_exact_static_import_and_validate_child_prefix(ledger_cell):
    from automated_phishing_detection.study_series_prefix import (
        validate_series_child_prefix,
    )

    current = ledger(ledger_cell.prefix)
    start(current, ledger_cell)
    frames = []
    for index, role in enumerate(("service", "client")):
        admission = current.issue(role, commands(ledger_cell)[index])
        validate_series_child_prefix(
            ledger_cell.prefix.binding, admission.frame, ledger_cell.prefix.payloads
        )
        admission.on_launched((321, 654)[index])
        frames.append(admission.frame)
    assert frames[0].predecessor_sha256 == frames[1].predecessor_sha256
    assert frames[0].cell_binding_sha256 == frames[1].cell_binding_sha256
