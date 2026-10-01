"""The locator index seals metadata, not catalog completeness or eligibility."""

from dataclasses import FrozenInstanceError, replace

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


def test_complete_closed_index_returns_immutable_locator_facts():
    case = make_case()
    result = validate(case)
    assert result.index_bytes == canonical_bytes(case.index)
    assert result.index_sha256 == digest(case.index)
    assert result.profile_sha256 == digest(case.profile)
    assert result.accepted_ordinals == (1, 2)
    assert result.stopped_ordinal == 3 and result.selected_attempt_ordinal == 4
    assert result.scope == "closed_series_history_locator_metadata_only"
    assert result.authorizes_execution is False
    assert result.file_refs == tuple(sorted(set(result.file_refs)))
    assert "/invented" not in repr(result) and "index_bytes" not in repr(result)
    with pytest.raises(FrozenInstanceError):
        result.stopped_ordinal = 4
    with pytest.raises(ValueError):
        replace(result, authorizes_execution=True)


@pytest.mark.parametrize("prefix", (1, 72, 124))
def test_entire_declared_prefix_is_required_without_claiming_eligibility(prefix):
    result = validate(make_case(prefix))
    assert result.accepted_ordinals == tuple(range(1, prefix + 1))
    assert result.stopped_ordinal == prefix + 1
    assert result.authorizes_execution is False


@pytest.mark.parametrize(
    "path",
    (
        "selected_root.payloads",
        "accepted_sources.internal",
        "accepted_sources.external",
        "accepted_cells.0.payloads",
        "stopped_cell.payloads",
        "stopped_cell.input_payloads",
        "physical_observations",
        "supervisor_files",
    ),
)
@pytest.mark.parametrize("operation", ("missing", "extra"))
def test_each_inventory_is_closed_even_when_rehashed(path, operation):
    case = make_case()
    members = selected(case.index, path)
    if operation == "missing":
        members.pop(next(iter(members)))
    else:
        members["invented-extra"] = {"path": "/invented/extra", "sha256": "0" * 64}
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        ("schema_version", True),
        ("selected_attempt_ordinal", 3),
        ("attempts.0.ordinal", True),
        ("attempts.0.disposition", "selected_entire_eligible_prefix"),
        ("attempts.3.disposition", "preserve_excluded"),
        ("attempts.3.profile.sha256", "0" * 64),
        ("attempts.3.envelope.sha256", "0" * 64),
        ("attempts.3.root_reservation_sha256", "0" * 64),
        ("original_hold.profile.sha256", "0" * 64),
        ("original_hold.envelope.sha256", "0" * 64),
        ("original_hold.reservation_sha256", "0" * 64),
        ("selected_root.reservation_sha256", "0" * 64),
        ("accepted_cells.0.ordinal", 2),
        ("accepted_cells.0.reservation_sha256", "0" * 64),
        ("stopped_cell.ordinal", 2),
        ("stopped_cell.ordinal", True),
        ("stopped_cell.inventory_kind", "measured_client_failure"),
        ("stopped_cell.reservation_sha256", "0" * 64),
        ("interruption_review.sha256", "0" * 64),
    ),
)
def test_original_ancestry_and_identity_joins_reject_rehashed_substitutions(
    path, replacement
):
    case = make_case()
    parent, _, name = path.rpartition(".")
    target = selected(case.index, parent) if parent else case.index
    target[name] = replacement
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


def test_predecessor_accounting_is_the_exact_selected_root_record():
    case = make_case()
    case.index["selected_root"]["payloads"]["attempt/study-accounting.json"][
        "sha256"
    ] = "0" * 64
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)


@pytest.mark.parametrize("part", ("attempts", "accepted_cells"))
@pytest.mark.parametrize("operation", ("missing", "duplicate", "reordered"))
def test_ordered_rosters_cannot_be_trimmed_duplicated_or_reordered(part, operation):
    case = make_case()
    roster = case.index[part]
    if operation == "missing":
        roster.pop()
    elif operation == "duplicate":
        roster.append(roster[0])
    else:
        roster.reverse()
    refresh(case)
    with pytest.raises(api().SeriesHistoryIndexError):
        validate(case)
