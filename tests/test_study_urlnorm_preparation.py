"""Amended root composition uses constructed capacities, not measured results."""

import json
from contextlib import contextmanager

import adopted_study_fixtures as adopted
import pytest
import study_urlnorm_preparation_fixtures as fixtures
from adopted_study_matrix_fixtures import install
from study_run_record_fixtures import (
    REQUIREMENTS,
    capacity,
    isolated_shortage,
    prepared,
)

from automated_phishing_detection._study_urlnorm_scope import PIN_FIELDS

__all__ = ["prepared"]


@pytest.mark.parametrize("requirement", REQUIREMENTS)
def test_each_amended_shortage_holds_before_any_source_or_cell(
    tmp_path, prepared, monkeypatch, requirement
):
    case = fixtures.setup(
        tmp_path, prepared, monkeypatch, isolated_shortage(prepared, requirement)
    )
    original = case.prior.payloads
    result = adopted.execute(case)
    public = json.loads(result.snapshot.payload("public-summary.json"))
    accounting = json.loads(result.snapshot.payload("attempt/study-accounting.json"))
    assert public["status"] == "whole_study_hold"
    assert accounting["authorization_ledger"]["admissions"] == []
    assert len(result.cells) == 125
    assert all(slot.status == "unattempted" for slot in result.cells)
    assert not case.paths.internal.attempt.exists()
    assert not case.paths.external.attempt.exists()
    assert case.prior.payloads == original
    assert case.events == [
        "hold_prior_root",
        "hold_prior_preparation",
        "derive",
        "hold_preparation",
        "release_preparation",
        "release_prior_preparation",
        "release_prior_root",
    ]


def test_amended_success_keeps_both_ancestors_through_125_cells_and_252_admissions(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    install(case, monkeypatch)
    result = adopted.execute(case)
    assert case.events.count("derive") == case.events.count("sources") == 1
    assert [event for event in case.events if type(event) is int] == list(range(1, 126))
    assert case.events.index("hold_prior_preparation") < case.events.index("sources")
    assert case.events.index(125) < case.events.index("reduce")
    assert case.events.index("reduce") < case.events.index("release_preparation")
    assert case.events[-2:] == ["release_prior_preparation", "release_prior_root"]
    assert all(slot.status == "accepted" for slot in result.cells)
    ledger = json.loads(result.snapshot.payload("attempt/study-accounting.json"))[
        "authorization_ledger"
    ]
    assert len(ledger["admissions"]) == 252
    assert len(ledger["cell_acceptances"]) == 125
    assert all(entry["accepted"] for entry in ledger["admissions"])
    assert (
        json.loads(result.snapshot.payload("public-summary.json"))["status"]
        == "study_evidence_published"
    )


@pytest.mark.parametrize("field", ["feasibility", "feasibility_sha256"])
def test_old_root_and_old_preparation_capacity_must_join_before_derivation(
    tmp_path, prepared, monkeypatch, field
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    case.old_barrier[field] = {} if field == "feasibility" else "0" * 64
    with pytest.raises(case.module.StudyRunError) as caught:
        adopted.execute(case)
    fixtures.failure(case, caught.value)
    assert "derive" not in case.events
    assert case.events[-2:] == ["release_prior_preparation", "release_prior_root"]


@pytest.mark.parametrize("field", ["representation", *PIN_FIELDS])
def test_new_completion_cannot_substitute_any_authorized_lineage_pin(
    tmp_path, prepared, monkeypatch, field
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    complete = json.loads(case.retained.payload("preparation-complete.json"))
    complete["derivation"][field] = "raw_url" if field == "representation" else "0" * 64
    fixtures.repin(case, complete)
    with pytest.raises(case.module.StudyRunError) as caught:
        adopted.execute(case)
    fixtures.failure(case, caught.value)
    assert case.events.count("derive") == 1
    assert case.events[-2:] == ["release_prior_preparation", "release_prior_root"]


@pytest.mark.parametrize("change", ["missing", "extra", "prior_profile"])
def test_new_completion_derivation_is_exact_not_an_open_subset(
    tmp_path, prepared, monkeypatch, change
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    complete = json.loads(case.retained.payload("preparation-complete.json"))
    if change == "missing":
        del complete["derivation"]
    else:
        complete["derivation"][change] = {}
    fixtures.repin(case, complete)
    with pytest.raises(case.module.StudyRunError) as caught:
        adopted.execute(case)
    fixtures.failure(case, caught.value)


def test_derivation_failure_releases_ancestors_and_never_falls_back_to_raw_inputs(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))

    def fail(*arguments, **keywords):
        case.events.append("failed_derivation")
        raise ValueError("invented derived preparation failure")

    monkeypatch.setattr(case.wrapper, "run_retained_study_preparation", fail)
    with pytest.raises(case.module.StudyRunError) as caught:
        adopted.execute(case)
    fixtures.failure(case, caught.value)
    assert case.events.count("failed_derivation") == 1
    assert case.events[-2:] == ["release_prior_preparation", "release_prior_root"]


def test_late_ancestor_rejection_preserves_published_hold_without_retry(
    tmp_path, prepared, monkeypatch
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch)
    original = case.wrapper.hold_prior_study_hold

    @contextmanager
    def late(*arguments):
        with original(*arguments) as held:
            yield held
        raise ValueError("invented late ancestry change")

    monkeypatch.setattr(case.wrapper, "hold_prior_study_hold", late)
    with pytest.raises(case.module.StudyRunError) as caught:
        adopted.execute(case)
    failure = case.module.adopted_study_failure(caught.value)
    assert failure.scientific.publishing
    assert failure.scientific.candidate is not None
    assert case.paths.public_summary.exists()
    assert case.events.count("derive") == 1
    assert json.loads(failure.authorization_ledger)["admissions"] == []


def test_original_authorization_retains_original_preparation_dispatch(
    tmp_path, prepared, monkeypatch
):
    module = fixtures.api()
    case = adopted.setup(tmp_path, prepared, monkeypatch)
    monkeypatch.setattr(module, "prepare", fixtures.forbidden)
    result = adopted.execute(case)
    assert case.events.count("prepare") == 1
    assert (
        json.loads(result.snapshot.payload("public-summary.json"))["status"]
        == "whole_study_hold"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 1),
        ("schema_version", 3),
        ("protocol", "study-preparation-v1"),
        ("protocol", "unknown"),
        ("payloads", ()),
        ("reservation_sha256", "0" * 64),
        ("completion_sha256", "0" * 64),
    ],
)
def test_mixed_completion_or_changed_holder_rejects_and_releases_all_ancestors(
    tmp_path, prepared, monkeypatch, field, value
):
    case = fixtures.setup(tmp_path, prepared, monkeypatch, capacity(prepared))
    fixtures.substitute(case, monkeypatch, field, value)
    with pytest.raises(case.module.StudyRunError) as caught:
        adopted.execute(case)
    fixtures.failure(case, caught.value)
    assert case.events.count("derive") == 1
    assert case.events[-3:] == [
        "release_preparation",
        "release_prior_preparation",
        "release_prior_root",
    ]
