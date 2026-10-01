"""Independent locator-only checks over invented profiles and file references."""

import builtins
import importlib
import os
import subprocess
from pathlib import Path

import pytest
from study_series_history_index_fixtures import api, make_case, refresh, validate


def all_references(value):
    references = [value["interruption_review"]]
    for attempt in value["attempts"]:
        references.extend(
            attempt[name]
            for name in ("profile", "envelope", "inventory", "interruption_review")
        )
    references.extend(
        value["original_hold"][name] for name in ("profile", "envelope", "inventory")
    )
    references.extend(value["selected_root"]["payloads"].values())
    for source in value["accepted_sources"].values():
        references.extend(source.values())
    for cell in (*value["accepted_cells"], value["stopped_cell"]):
        references.extend(cell["payloads"].values())
    for group in ("physical_observations", "supervisor_files"):
        references.extend(value[group].values())
    references.extend(value["stopped_cell"]["input_payloads"].values())
    return tuple(sorted({(item["path"], item["sha256"]) for item in references}))


def test_returned_reference_inventory_preserves_every_declared_category():
    case = make_case()
    result = validate(case)
    assert result.file_refs == all_references(case.index)
    case.index["attempts"][0]["inventory"]["path"] += "-changed"
    assert result.file_refs != all_references(case.index)
    assert result.authorizes_execution is False


@pytest.mark.parametrize(
    "output",
    (
        "series_attempt",
        "series_public_summary",
        "segment_attempt",
        "segment_public_summary",
        "historical_inputs_dir",
        "cells_dir",
        "physical_records_dir",
    ),
)
@pytest.mark.parametrize("relation", ("same", "ancestor", "descendant"))
def test_physical_file_ref_cannot_overlap_any_future_output(output, relation):
    case = make_case()
    location = case.profile["paths"][output]
    selected = {
        "same": location,
        "ancestor": str(Path(location).parent),
        "descendant": location + "/observation.json",
    }[relation]
    case.index["physical_observations"]["pre.json"]["path"] = selected
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_source_spec_must_also_match_the_series_scientific_pin():
    case = make_case()
    case.profile["scientific_pins"]["source_spec_sha256"] = "0" * 64
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize("member", ("expected_index_sha256", "expected_profile_sha256"))
def test_bad_independent_pin_stops_before_index_semantics(monkeypatch, member):
    from automated_phishing_detection import _study_series_index_records as records

    case = make_case()

    def forbidden(*args):
        pytest.fail("unauthenticated bytes reached index semantic validation")

    monkeypatch.setattr(records, "index", forbidden)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case, **{member: "0" * 64})


def deny_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("locator validation attempted IO or scientific reconstruction")

    for module_name, entry in (
        ("study_history_internal", "verify_historical_internal_science"),
        ("study_history_external", "verify_historical_external_science"),
        ("study_history_cell", "verify_historical_cell_science"),
        ("study_history_snapshots", "verify_study_history_snapshots"),
        ("stopped_study_authorization", "verify_stopped_study_authorization"),
        ("stopped_study_timeline", "verify_stopped_study_timeline"),
    ):
        module = importlib.import_module(f"automated_phishing_detection.{module_name}")
        monkeypatch.setattr(module, entry, forbidden)
    for owner, names in (
        (builtins, ("open",)),
        (os, ("open", "listdir")),
        (Path, ("open", "read_bytes", "read_text", "stat", "resolve")),
        (subprocess, ("Popen",)),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)


def test_no_catalog_custody_or_scientific_access_occurs(monkeypatch):
    case = make_case()
    api()
    with monkeypatch.context() as isolated:
        deny_effects(isolated)
        result = validate(case)
    assert result.file_refs == all_references(case.index)
    assert result.authorizes_execution is False
