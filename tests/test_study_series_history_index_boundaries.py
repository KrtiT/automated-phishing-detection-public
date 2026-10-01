"""Pinned locators cannot redirect history or acquire file-access authority."""

import builtins
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from study_series_adoption_fixtures import digest
from study_series_history_index_fixtures import (
    api,
    make_case,
    refresh,
    selected,
    validate,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes

LOCATORS = (
    ("selected_root.payloads", "attempt/study-intent.json"),
    ("accepted_sources.internal", "source/data/sources.json"),
    ("accepted_sources.internal", "attempt/evidence/bindings.json"),
    ("accepted_sources.external", "public-summary.json"),
    ("accepted_cells.0.payloads", "public-summary.json"),
    ("accepted_cells.0.payloads", "attempt/reservation.json"),
    ("stopped_cell.payloads", "attempt/process-pair.json"),
    ("stopped_cell.input_payloads", "manifest"),
)


@pytest.mark.parametrize(("group", "member"), LOCATORS)
def test_every_determined_locator_must_equal_original_location(group, member):
    case = make_case()
    selected(case.index, group)[member]["path"] += "-substituted"
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize(
    "path",
    (
        "relative",
        "/unit/../escape",
        "/unit//alias",
        "/unit/./alias",
        "/unit/alias/",
        "//unit/alias",
        "/unit/secret\0",
        "/invented/fresh/cells_dir/child",
        "/invented/fresh",
    ),
)
def test_unconstrained_catalog_locations_still_obey_lexical_and_output_guards(path):
    case = make_case()
    case.index["attempts"][0]["inventory"]["path"] = path
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_same_path_with_conflicting_hashes_is_rejected():
    case = make_case()
    references = case.index["physical_observations"]
    references["post.json"]["path"] = references["pre.json"]["path"]
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_identical_repeated_catalog_ref_rejects_duplicate_archive():
    case = make_case()
    case.index["attempts"][1]["inventory"] = deepcopy(
        case.index["attempts"][0]["inventory"]
    )
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_excluded_archive_cannot_be_counted_as_two_attempts():
    case = make_case()
    case.index["attempts"][1] = deepcopy(case.index["attempts"][0])
    case.index["attempts"][1]["ordinal"] = 2
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_refs_cannot_conflict_with_profile_pinned_history_locators():
    case = make_case()
    case.index["attempts"][0]["inventory"]["path"] = case.profile["history"][
        "exposure_record_path"
    ]
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_profile_history_locators_cannot_disagree_on_the_same_path():
    case = make_case()
    history = case.profile["history"]
    history["eligible_prefix_review_path"] = history["exposure_record_path"]
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize(
    "name", ("data/sources.json", "reports/phiusiil-preparation-summary.json")
)
def test_public_sources_match_original_profile_scope(name):
    case = make_case()
    case.index["accepted_sources"]["internal"][f"source/{name}"]["sha256"] = "0" * 64
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize("kind", ("internal", "external"))
def test_scientific_binding_pins_are_joined_without_reconstruction(kind):
    case = make_case()
    case.profile["scientific_pins"][f"{kind}_bindings_sha256"] = "0" * 64
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize("member", ("expected_index_sha256", "expected_profile_sha256"))
def test_original_bytes_require_independent_pins(member):
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(make_case(), **{member: "0" * 64})


@pytest.mark.parametrize("member", ("index_bytes", "profile_bytes"))
def test_noncanonical_bytes_reject_after_independent_rehash(member):
    case = make_case()
    content = b" " + canonical_bytes(
        case.index if member == "index_bytes" else case.profile
    )
    pin = (
        "expected_index_sha256"
        if member == "index_bytes"
        else "expected_profile_sha256"
    )
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case, **{member: content, pin: sha256(content).hexdigest()})


def test_profile_revalidation_cannot_be_replaced_by_self_consistent_metadata():
    case = make_case()
    case.profile["transition"]["unchanged_sha256"]["src/unreviewed.py"] = "0" * 64
    case.profile["source_artifact_scope"]["src/unreviewed.py"] = "0" * 64
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_no_files_catalogs_or_history_payloads_are_read(monkeypatch):
    from automated_phishing_detection import (
        execution_preflight,
        stopped_study_authorization,
    )

    case = make_case()
    api()

    def forbidden(*args, **kwargs):
        pytest.fail("locator metadata attempted IO or historical payload verification")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(execution_preflight, "bind_execution", forbidden)
    monkeypatch.setattr(
        stopped_study_authorization, "verify_stopped_study_authorization", forbidden
    )
    assert validate(case).authorizes_execution is False


def test_profile_index_pin_must_match_index_bytes():
    case = make_case()
    case.profile["history"]["index_sha256"] = "0" * 64
    assert digest(case.index) != "0" * 64
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)
