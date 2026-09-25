import json
from dataclasses import replace

import pytest
from retained_preparation_mutation_fixtures import (
    changed_retained,
    external_candidate,
    normalized_preparation,
)
from retained_study_preparation_fixtures import api, changed, restore, retained_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["api", "retained_case"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("record_id", "substituted-id"),
        ("raw_url", "https://gold.example/path"),
        ("registrable_domain", "example"),
    ],
)
def test_retained_rows_link_exact_publisher_identity(api, retained_case, field, value):
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, changed_retained(retained_case, **{field: value}))


def test_valid_changed_mapping_cannot_replace_publisher_mapping(api, retained_case):
    rows = tuple(
        replace(row, confidence_tier="silver") if row.published_id == "gold" else row
        for row in retained_case.publisher.rows
    )
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, normalized_preparation(retained_case, rows=rows))


def test_original_quarantined_only_overlap_cannot_be_retained(api, retained_case):
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, normalized_preparation(retained_case, domains=frozenset()))


@pytest.mark.parametrize(
    "field,value",
    [
        ("published_id", "wrong-quarantine-id"),
        ("canonical_url_sha256", "0" * 64),
    ],
)
def test_quarantine_coordinates_link_exact_publisher_identity(
    api, retained_case, field, value
):
    first, *remaining = retained_case.external.quarantine
    quarantine = (replace(first, **{field: value}), *remaining)
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, external_candidate(retained_case, quarantine=quarantine))


def test_complete_preparation_inventory_must_match_saved_publisher(api, retained_case):
    counts = dict(retained_case.publisher.published_split_counts)
    counts["train"] += 1
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, external_candidate(retained_case, counts=counts))


@pytest.mark.parametrize(
    "field,value",
    [("shortages", []), ("scope", "scoring_authorized"), ("schema_version", True)],
)
def test_feasibility_is_an_exact_derived_record(api, retained_case, field, value):
    record = json.loads(retained_case.snapshot.payload("feasibility.json"))
    record[field] = value
    candidate = changed(
        retained_case, {"feasibility.json": canonical_bytes(record)}, repin=True
    )
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, candidate)
