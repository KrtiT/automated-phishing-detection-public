"""Byte-only preparation over existing invented source fixtures."""

import importlib
import importlib.util
from hashlib import sha256

import pytest
from test_source_overlap import source as source_fixture

from automated_phishing_detection import evaluation_producer, source_overlap


def test_preparation_api_exists():
    assert importlib.util.find_spec(
        "automated_phishing_detection.study_preparation_inputs"
    ), "missing byte-only study preparation boundary"


@pytest.fixture
def api():
    return importlib.import_module(
        "automated_phishing_detection.study_preparation_inputs"
    )


@pytest.fixture
def internal_case():
    case = source_fixture.__wrapped__()
    split = case.report["splits"]["group_test"]
    case.source = {
        "expected_sha256": case.report["output_hashes"]["group_test.jsonl"],
        "source_csv_sha256": case.pins["source_csv_sha256"],
        "suffix_rules_sha256": case.pins["suffix_rules_sha256"],
        "expected_row_count": split["row_count"],
        "expected_domain_count": split["domain_count"],
        "expected_class_counts": split["class_counts"],
    }
    return case


def prepare_internal(api, case, **changes):
    buffers = case.buffers | changes
    return api.prepare_internal_inputs(
        buffers["csv_bytes"],
        buffers["suffix_rules_bytes"],
        buffers["source_spec_bytes"],
        buffers["preparation_summary_bytes"],
        pins=source_overlap.SourceOverlapPins(**case.pins),
        source=case.source,
    )


def test_internal_returns_existing_typed_preparation_without_sampling(
    api, internal_case
):
    reconstructed, prepared = prepare_internal(api, internal_case)
    assert type(reconstructed) is source_overlap.ReconstructedSource
    assert type(prepared) is evaluation_producer.PreparedInternal
    assert reconstructed.group_test_bytes == internal_case.outputs["group_test.jsonl"]
    assert (
        prepared.partition_sha256 == sha256(reconstructed.group_test_bytes).hexdigest()
    )
    assert prepared.class_counts == tuple(
        internal_case.source["expected_class_counts"][label] for label in ("0", "1")
    )
    assert "invalid-label.example" in reconstructed.overlap_domains
    assert "conflict.example" in reconstructed.overlap_domains
    assert all(record.split == "group_test" for record in prepared.records)


@pytest.mark.parametrize(
    "field",
    (
        "csv_bytes",
        "suffix_rules_bytes",
        "source_spec_bytes",
        "preparation_summary_bytes",
    ),
)
def test_internal_authenticates_every_buffer_before_parsing(
    api, internal_case, monkeypatch, field
):
    def forbidden(*args, **kwargs):
        pytest.fail("parsed unauthenticated source bytes")

    monkeypatch.setattr(source_overlap.phiusiil, "_parse_csv_rows", forbidden)
    monkeypatch.setattr(source_overlap, "_public_binding", forbidden)
    with pytest.raises(
        api.StudyPreparationInputError, match="^invalid_internal_inputs$"
    ):
        prepare_internal(api, internal_case, **{field: b"private-secret-canary"})


@pytest.mark.parametrize("field", ["source_csv_sha256", "suffix_rules_sha256"])
def test_internal_parser_claims_must_match_authenticated_pins(
    api, internal_case, field
):
    internal_case.source = internal_case.source | {field: "f" * 64}
    with pytest.raises(
        api.StudyPreparationInputError, match="^invalid_internal_inputs$"
    ):
        prepare_internal(api, internal_case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("expected_row_count", True),
        ("expected_domain_count", 0),
        ("expected_class_counts", {"0": 0, "1": 0}),
        ("unexpected", "private-secret-canary"),
    ],
)
def test_internal_keeps_existing_partition_acceptance(api, internal_case, field, value):
    internal_case.source = internal_case.source | {field: value}
    with pytest.raises(
        api.StudyPreparationInputError, match="^invalid_internal_inputs$"
    ):
        prepare_internal(api, internal_case)
