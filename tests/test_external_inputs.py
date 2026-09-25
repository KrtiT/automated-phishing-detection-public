"""Validate rich prepared state before any external scoring begins."""

import importlib
import importlib.util
from dataclasses import replace

import pytest
from external_producer_fixtures import (
    prepare_normalized,
    prepared_duplicate_canonical,
    prepared_external,
    relink_quarantine,
    relink_rows,
)

from automated_phishing_detection import phishvn


@pytest.fixture
def inputs():
    name = "automated_phishing_detection._external_inputs"
    assert importlib.util.find_spec(name), "missing prepared external validation"
    return importlib.import_module(name)


@pytest.mark.parametrize("count", [0, 1, 5, 320])
def test_prepared_identity_order_and_nullable_controls_survive(inputs, count):
    prepared = prepared_external(count)
    rows = inputs.validate_prepared_external(prepared)
    assert rows == prepared.retained and len(rows) == count
    assert [row.file_position for row in rows] == list(range(1, count + 1))
    assert all(row.is_phishing is None for row in rows if row.role == "tranco")


@pytest.mark.parametrize(
    "changes",
    [
        {"is_phishing": None},
        {"is_phishing": True},
        {"role": "tranco"},
        {"source_group": "private-sensitive-source"},
        {"confidence_tier": "silver"},
        {"record_id": "row-1"},
        {"file_position": 2},
        {"file_position": True},
        {"source_split": "val", "published_split": "val"},
        {"canonical_url_sha256": "0" * 64},
        {"registrable_domain": "other.com"},
        {"raw_url": "example0.com"},
    ],
)
def test_relinked_row_corruption_rejects(inputs, changes):
    prepared = prepared_external()
    rows = (replace(prepared.retained[0], **changes), *prepared.retained[1:])
    with pytest.raises(inputs.ExternalInputError) as failure:
        inputs.validate_prepared_external(relink_rows(prepared, rows))
    assert "private-sensitive" not in str(failure.value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("retained_test_rows", 4),
        ("retained_test_domain_count", True),
        ("protected_evaluation_authorized", True),
        ("schema_verified", True),
        ("input_row_count", 10),
        ("valid_non_test_rows", 1),
    ],
)
def test_changed_public_preparation_counts_reject(inputs, field, value):
    prepared = prepared_external()
    summary = prepared.public_summary | {field: value}
    with pytest.raises(inputs.ExternalInputError):
        inputs.validate_prepared_external(replace(prepared, public_summary=summary))


@pytest.mark.parametrize(
    "name", ["retained-test.jsonl", "quarantine.jsonl", "inventory.json"]
)
def test_private_bytes_are_required_and_hash_bound(inputs, name):
    prepared = prepared_external()
    for private in (
        {key: value for key, value in prepared.private_outputs.items() if key != name},
        prepared.private_outputs | {name: prepared.private_outputs[name] + b" "},
    ):
        with pytest.raises(inputs.ExternalInputError):
            inputs.validate_prepared_external(
                replace(prepared, private_outputs=private)
            )


def test_relinked_record_removal_cannot_hide_missing_test_position(inputs):
    prepared = prepared_external()
    with pytest.raises(inputs.ExternalInputError):
        inputs.validate_prepared_external(relink_rows(prepared, prepared.retained[:-1]))


def test_kernel_produced_quarantine_is_valid(inputs):
    prepared = prepared_external(quarantine_count=2)
    assert inputs.validate_prepared_external(prepared) == prepared.retained


@pytest.mark.parametrize(
    "changes",
    [
        {"reason_codes": ("invalid_or_missing_url", "private-sensitive-reason")},
        {"reason_codes": ()},
        {"reason_codes": ["invalid_or_missing_url"]},
        {"reason_codes": "invalid_or_missing_url"},
        {"reason_codes": (True,)},
        {"reason_codes": ("invalid_or_missing_url", "invalid_or_missing_url")},
        {"reason_codes": ("undefined_phishvn_mapping", "invalid_or_missing_url")},
        {"published_id": "private sensitive identifier"},
        {"published_id": True},
        {"published_id": None},
        {"published_id": "row-0"},
        {"canonical_url_sha256": "invalid"},
        {"canonical_url_sha256": "0" * 64},
        {"file_position": True},
        {"file_position": 0},
        {"source_split": "other"},
    ],
)
def test_relinked_quarantine_metadata_rejects_before_public_counts(inputs, changes):
    prepared = prepared_external(quarantine_count=1)
    quarantine = (replace(prepared.quarantine[0], **changes),)
    with pytest.raises(inputs.ExternalInputError) as failure:
        inputs.validate_prepared_external(relink_quarantine(prepared, quarantine))
    assert "private-sensitive" not in str(failure.value)


@pytest.mark.parametrize("mutation", ["reverse", "duplicate_id", "duplicate_position"])
def test_quarantine_order_and_visible_identity_are_checked(inputs, mutation):
    prepared = prepared_external(quarantine_count=2)
    first, second = prepared.quarantine
    if mutation == "reverse":
        quarantine = (second, first)
    elif mutation == "duplicate_id":
        quarantine = (first, replace(second, published_id=first.published_id))
    else:
        quarantine = (first, replace(second, file_position=first.file_position))
    with pytest.raises(inputs.ExternalInputError):
        inputs.validate_prepared_external(relink_quarantine(prepared, quarantine))


def test_kernel_canonical_duplicate_can_span_retained_and_quarantined(inputs):
    prepared = prepared_duplicate_canonical()
    assert len(prepared.retained) == len(prepared.quarantine) == 1
    assert (
        prepared.retained[0].canonical_url_sha256
        == prepared.quarantine[0].canonical_url_sha256
    )
    assert inputs.validate_prepared_external(prepared) == prepared.retained


def test_duplicate_stable_ids_already_fail_in_the_preparation_kernel():
    with pytest.raises(
        phishvn.ExternalPreparationError, match="duplicate published ID"
    ):
        prepared_duplicate_canonical(duplicate_id=True)


def test_kernel_missing_id_and_url_quarantine_is_valid(inputs):
    prepared = prepare_normalized(
        (
            phishvn.NormalizedExternalRow(
                None, "test", 1, None, "ncsc", "phishing", "gold", "test"
            ),
        )
    )
    assert inputs.validate_prepared_external(prepared) == ()
    assert prepared.quarantine[0].reason_codes == (
        "invalid_or_missing_url",
        "missing_or_invalid_published_id",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("public_summary", []),
        ("private_outputs", []),
        ("retained", []),
        ("quarantine", (None,)),
    ],
)
def test_malformed_container_values_are_symbolically_rejected(inputs, field, value):
    with pytest.raises(inputs.ExternalInputError):
        inputs.validate_prepared_external(
            replace(prepared_external(), **{field: value})
        )
