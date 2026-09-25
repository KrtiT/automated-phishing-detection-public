import json
from hashlib import sha256

import pytest
from retained_preparation_mutation_fixtures import overlap_candidate
from retained_study_preparation_fixtures import api, changed, restore, retained_case

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["api", "retained_case"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_ordinal", True),
        ("source_ordinal", 1.0),
        ("source_ordinal", 2),
        ("record_id", "invented-wrong-id"),
        ("status", "unknown"),
        ("canonical_url_sha256", "A" * 64),
        ("canonical_url_sha256", None),
        ("registrable_domain", "DOMAIN-00.example"),
        ("registrable_domain", None),
        ("extra", None),
    ],
)
def test_manifest_membership_exact_schema(api, retained_case, field, value):
    manifest = json.loads(retained_case.snapshot.payload("source-overlap.json"))
    manifest["rows"][0][field] = value
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, overlap_candidate(retained_case, manifest))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "reordered",
        "domain_removed",
        "domain_extra",
        "domain_duplicate",
    ],
)
def test_manifest_requires_complete_ordered_original_universe(
    api, retained_case, mutation
):
    manifest = json.loads(retained_case.snapshot.payload("source-overlap.json"))
    if mutation == "missing":
        manifest["rows"].pop()
    elif mutation == "duplicate":
        manifest["rows"][-1] = manifest["rows"][0]
    elif mutation == "reordered":
        manifest["rows"].reverse()
    elif mutation == "domain_removed":
        manifest["domains"].remove("invalid-label.example")
    elif mutation == "domain_extra":
        manifest["domains"].append("unobserved.example")
    else:
        manifest["domains"].append(manifest["domains"][-1])
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, overlap_candidate(retained_case, manifest))


@pytest.mark.parametrize("field", ["canonical_url_sha256", "registrable_domain"])
def test_partition_records_join_original_membership(api, retained_case, field):
    manifest = json.loads(retained_case.snapshot.payload("source-overlap.json"))
    selected = retained_case.internal.records[0]
    row = next(
        row for row in manifest["rows"] if row["record_id"] == selected.record_id
    )
    other = next(
        domain
        for domain in manifest["domains"]
        if domain != selected.registrable_domain
    )
    row[field] = "0" * 64 if field == "canonical_url_sha256" else other
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, overlap_candidate(retained_case, manifest))


def test_full_overlap_domains_are_psl_validated(api, retained_case):
    manifest = json.loads(retained_case.snapshot.payload("source-overlap.json"))
    selected = next(
        row
        for row in manifest["rows"]
        if row["registrable_domain"] == "invalid-label.example"
    )
    selected["registrable_domain"] = "example"
    manifest["domains"] = sorted(
        {
            row["registrable_domain"]
            for row in manifest["rows"]
            if row["registrable_domain"] is not None
        }
    )
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, overlap_candidate(retained_case, manifest))


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution", {}),
        ("reservation_sha256", "0" * 64),
        ("schema_version", True),
        ("checkpoint_sha256", {}),
        ("extra", None),
    ],
)
def test_original_reconstruction_receipt_is_closed(api, retained_case, field, value):
    receipt = json.loads(retained_case.snapshot.payload("source-reconstruction.json"))
    receipt[field] = value
    candidate = changed(
        retained_case,
        {"source-reconstruction.json": canonical_bytes(receipt)},
        repin=True,
    )
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, candidate)


def test_coherent_partition_rehash_does_not_replace_public_pin(api, retained_case):
    rows = [
        json.loads(line)
        for line in retained_case.snapshot.payload("group_test.jsonl").splitlines()
    ]
    rows[0]["is_phishing"] = 1 - rows[0]["is_phishing"]
    content = b"".join(canonical_bytes(row) for row in rows)
    receipt = json.loads(retained_case.snapshot.payload("source-reconstruction.json"))
    receipt["checkpoint_sha256"]["group_test.jsonl"] = sha256(content).hexdigest()
    candidate = changed(
        retained_case,
        {
            "group_test.jsonl": content,
            "source-reconstruction.json": canonical_bytes(receipt),
        },
        repin=True,
    )
    with pytest.raises(api.StudyPreparationRestoreError):
        restore(api, candidate)
