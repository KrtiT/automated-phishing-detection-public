"""Normalized synthetic publisher rows only; no external file schema is assumed."""

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection.protocol_preflight import parse_suffix_rules

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def adapter():
    assert (ROOT / "src/automated_phishing_detection/phishvn.py").is_file(), (
        "missing normalized external preparation"
    )
    from automated_phishing_detection import phishvn

    return phishvn


def row(adapter, identity, position, *, split="test", **fields):
    values = {
        "published_id": identity,
        "source_split": split,
        "file_position": position,
        "raw_url": f"https://{identity}.example.com/path",
        "source_group": "ncsc",
        "source_class": "phishing",
        "confidence_tier": "gold",
        "published_split": split,
    }
    values.update(fields)
    return adapter.NormalizedExternalRow(**values)


def rows(adapter):
    return (
        row(adapter, "train-row", 1, split="train", raw_url="https://development.org/"),
        row(adapter, "gold-row", 1),
        row(
            adapter,
            "certified-row",
            2,
            source_group="trusted_registry",
            source_class="legitimate",
            confidence_tier="certified",
        ),
        row(adapter, "silver-row", 3, confidence_tier="silver"),
        row(
            adapter,
            "bronze-row",
            4,
            source_group="chongluadao_openphish",
            confidence_tier="bronze",
        ),
        row(
            adapter,
            "tranco-row",
            5,
            source_group="tranco",
            source_class="reference_negative",
            confidence_tier="control",
        ),
    )


def prepare(adapter, supplied=None, *, coverage=None, **kwargs):
    supplied = rows(adapter) if supplied is None else tuple(supplied)
    coverage = {"train": 1, "test": 5} if coverage is None else coverage
    return adapter.prepare_external_rows(
        supplied,
        published_split_counts=coverage,
        test_split="test",
        suffix_rules=parse_suffix_rules("com\norg\nco.uk\n"),
        phiusiil_domains=kwargs.pop("phiusiil_domains", frozenset()),
        **kwargs,
    )


def test_exact_existing_mappings_keep_all_roles_in_published_order(adapter):
    result = prepare(adapter)
    assert [record.role for record in result.retained] == [
        "gold",
        "certified",
        "secondary",
        "secondary",
        "tranco",
    ]
    assert [record.is_phishing for record in result.retained] == [1, 0, 1, 1, None]
    assert [record.file_position for record in result.retained] == [1, 2, 3, 4, 5]
    assert [record.record_id for record in result.retained] == [
        source.published_id for source in rows(adapter)[1:]
    ]
    assert [metadata.role for metadata in result.metadata] == [
        record.role for record in result.retained
    ]
    assert result.quarantine == ()
    assert result.public_summary["valid_non_test_rows"] == 1
    assert result.public_summary["retained_test_rows"] == 5


def test_positions_not_input_iteration_or_identifier_sort_define_test_order(adapter):
    original = prepare(adapter)
    reversed_input = prepare(adapter, tuple(reversed(rows(adapter))))
    assert original == reversed_input


@pytest.mark.parametrize(
    "coverage",
    [
        None,
        {"test": 5},
        {"train": 1},
        {"train": 1, "test": 4},
        {"train": True, "test": 5},
        {"train": 1, "test": 5, "validation": 1},
    ],
)
def test_requires_explicit_exact_all_split_inventory(adapter, coverage):
    with pytest.raises(adapter.ExternalPreparationError):
        adapter.prepare_external_rows(
            rows(adapter),
            published_split_counts=coverage,
            test_split="test",
            suffix_rules=parse_suffix_rules("com\norg\n"),
            phiusiil_domains=frozenset(),
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"file_position": 0},
        {"file_position": True},
        {"file_position": 2},
        {"source_split": "unknown"},
    ],
)
def test_ambiguous_source_positions_or_inventory_stop_before_preparation(
    adapter, changes
):
    source = list(rows(adapter))
    source[1] = replace(source[1], **changes)
    with pytest.raises(adapter.ExternalPreparationError):
        prepare(adapter, source)


@pytest.mark.parametrize("identical_payload", [False, True])
def test_duplicate_published_id_is_rejected_not_arbitrarily_selected(
    adapter, identical_payload
):
    source = list(rows(adapter))
    other = source[1] if identical_payload else source[2]
    source[2] = replace(other, published_id=source[1].published_id, file_position=2)
    with pytest.raises(adapter.ExternalPreparationError, match="published ID"):
        prepare(adapter, source)


@pytest.mark.parametrize(
    "fields,reason",
    [
        ({"raw_url": None}, "invalid_or_missing_url"),
        ({"published_id": None}, "missing_or_invalid_published_id"),
        ({"source_group": ""}, "missing_phishvn_mapping_field"),
        ({"source_class": None}, "missing_phishvn_mapping_field"),
        ({"confidence_tier": "unknown"}, "undefined_phishvn_mapping"),
        ({"source_group": "NCSC"}, "undefined_phishvn_mapping"),
        ({"source_class": "0"}, "undefined_phishvn_mapping"),
        ({"source_class": 1}, "missing_phishvn_mapping_field"),
    ],
)
def test_invalid_values_are_quarantined_without_imputing_identity_or_label(
    adapter, fields, reason
):
    source = list(rows(adapter))
    source[1] = replace(source[1], **fields)
    result = prepare(adapter, source)
    entry = next(item for item in result.quarantine if item.file_position == 1)
    assert reason in entry.reason_codes
    assert [item.file_position for item in result.retained] == [2, 3, 4, 5]
    assert all(
        item.is_phishing is None for item in result.retained if item.role == "tranco"
    )


@pytest.mark.parametrize("published_split", [None, "unknown", "train"])
def test_missing_unknown_or_conflicting_published_split_quarantines_entire_domain(
    adapter, published_split
):
    source = list(rows(adapter))
    source[1] = replace(source[1], published_split=published_split)
    result = prepare(adapter, source)
    assert (
        result.retained == ()
    )  # Every test fixture has registrable domain example.com.
    assert len(result.quarantine) == 5
    assert all(
        "domain_invalid_published_split" in entry.reason_codes
        for entry in result.quarantine
    )


def test_canonical_conflicting_outcome_mapping_quarantines_whole_url_group(adapter):
    source = list(rows(adapter))
    source[2] = replace(source[2], raw_url=source[1].raw_url)
    result = prepare(adapter, source)
    assert [item.file_position for item in result.retained] == [3, 4, 5]
    assert all(
        "canonical_url_conflicting_mapping" in entry.reason_codes
        for entry in result.quarantine
    )


def test_same_label_duplicate_keeps_lexicographically_smallest_published_id(adapter):
    source = list(rows(adapter))
    source[1] = replace(source[1], published_id="z-id")
    source[3] = replace(source[3], published_id="a-id", raw_url=source[1].raw_url)
    result = prepare(adapter, source)
    assert [item.file_position for item in result.retained] == [2, 3, 4, 5]
    winner = next(item for item in result.retained if item.record_id == "a-id")
    assert winner.role == "secondary"  # Never prefer gold because of its outcome role.
    assert result.quarantine[0].published_id == "z-id"
    assert "canonical_url_duplicate_same_mapping" in result.quarantine[0].reason_codes


def test_missing_stable_identifier_quarantines_entire_duplicate_group(adapter):
    source = list(rows(adapter))
    source[3] = replace(source[3], published_id=None, raw_url=source[1].raw_url)
    result = prepare(adapter, source)
    assert [item.file_position for item in result.retained] == [2, 4, 5]
    assert all(
        "canonical_url_missing_stable_id" in item.reason_codes
        for item in result.quarantine
    )


def test_undefined_mapping_cannot_leave_a_same_url_primary_record_eligible(adapter):
    source = list(rows(adapter))
    source[3] = replace(source[3], source_class=None, raw_url=source[1].raw_url)
    result = prepare(adapter, source)
    assert [item.file_position for item in result.retained] == [2, 4, 5]
    assert all(
        "canonical_url_invalid_mapping" in item.reason_codes
        for item in result.quarantine
    )


def test_cross_split_domain_exclusion_uses_all_rows_even_invalid_outcomes(adapter):
    source = list(rows(adapter))
    source[0] = replace(
        source[0], raw_url="https://train.example.com/", source_class=None
    )
    result = prepare(adapter, source)
    assert result.retained == ()
    assert len(result.quarantine) == 6
    assert all(
        "domain_crosses_published_splits" in entry.reason_codes
        for entry in result.quarantine
    )


def test_phiusiil_overlap_quarantines_every_external_domain_member(adapter):
    result = prepare(adapter, phiusiil_domains=frozenset({"example.com"}))
    assert result.retained == ()
    assert len(result.quarantine) == 5
    assert (
        result.public_summary["quarantine_reason_counts"]["phiusiil_domain_overlap"]
        == 5
    )


def test_canonicalization_and_psl_use_existing_primitives(adapter):
    source = list(rows(adapter))
    source[1] = replace(source[1], raw_url="HTTPS://Host.Example.CO.UK:443/%ab")
    result = prepare(adapter, source)
    record = result.retained[0]
    assert record.raw_url == source[1].raw_url
    assert record.registrable_domain == "example.co.uk"
    assert (
        record.canonical_url_sha256
        == sha256(b"https://host.example.co.uk/%AB").hexdigest()
    )


def test_private_evidence_is_deterministic_and_public_output_is_aggregate_only(adapter):
    result = prepare(adapter)
    assert result == prepare(adapter)
    serialized = json.dumps(result.public_summary)
    for source in rows(adapter):
        assert source.published_id not in serialized
        assert source.raw_url not in serialized
    assert "example.com" not in serialized
    assert result.public_summary["schema_verified"] is False
    assert result.public_summary["protected_evaluation_authorized"] is False
    assert result.public_summary["scope"] == "normalized_inputs_only"
    for name, content in result.private_outputs.items():
        assert (
            result.public_summary["private_sha256"][name] == sha256(content).hexdigest()
        )
    saved = [
        json.loads(line)
        for line in result.private_outputs["retained-test.jsonl"].splitlines()
    ]
    assert [item["record_id"] for item in saved] == [
        item.record_id for item in result.retained
    ]


@pytest.mark.parametrize(
    "domains",
    [set(), frozenset({"com"}), frozenset({"Example.com"}), frozenset({True})],
)
def test_phiusiil_overlap_claim_requires_exact_canonical_domain_set(adapter, domains):
    with pytest.raises(adapter.ExternalPreparationError):
        prepare(adapter, phiusiil_domains=domains)


def test_control_and_certified_same_url_conflict_without_relabeling_controls(adapter):
    source = list(rows(adapter))
    source[5] = replace(source[5], raw_url=source[2].raw_url)
    result = prepare(adapter, source)
    assert [record.role for record in result.retained] == [
        "gold",
        "secondary",
        "secondary",
    ]
    assert {entry.file_position for entry in result.quarantine} == {2, 5}
    assert all(
        "canonical_url_conflicting_mapping" in entry.reason_codes
        for entry in result.quarantine
    )


def test_exclusion_rows_are_disjoint_even_when_reason_incidences_overlap(adapter):
    source = list(rows(adapter))
    source[3] = replace(source[3], raw_url=source[1].raw_url)
    result = prepare(adapter, source, phiusiil_domains=frozenset({"example.com"}))
    summary = result.public_summary
    assert (
        summary["input_row_count"]
        == summary["retained_test_rows"]
        + summary["valid_non_test_rows"]
        + summary["quarantined_rows"]
    )
    assert summary["quarantined_rows"] == 5
    assert sum(summary["quarantine_reason_counts"].values()) == 6
    assert (
        len({(entry.source_split, entry.file_position) for entry in result.quarantine})
        == 5
    )


def test_absent_declared_partition_cannot_be_hidden_by_quarantine(adapter):
    source = rows(adapter)[1:]
    with pytest.raises(adapter.ExternalPreparationError, match="cover"):
        prepare(adapter, source, phiusiil_domains=frozenset({"example.com"}))


@pytest.mark.parametrize(
    "make_rows", [lambda value: iter(value), lambda value: [object()]]
)
def test_only_materialized_typed_rows_are_accepted(adapter, make_rows):
    with pytest.raises(adapter.ExternalPreparationError):
        adapter.prepare_external_rows(
            make_rows(rows(adapter)),
            published_split_counts={"train": 1, "test": 5},
            test_split="test",
            suffix_rules=parse_suffix_rules("com\norg\n"),
            phiusiil_domains=frozenset(),
        )
