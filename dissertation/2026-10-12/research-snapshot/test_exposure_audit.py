"""Synthetic identity-only tests for the local historical-exposure audit."""

import importlib
import json

import pytest

from automated_phishing_detection.protocol_preflight import parse_suffix_rules


@pytest.fixture
def audit():
    return importlib.import_module("exposure_audit")


def test_full_and_nested_records_resolve_the_same_identity(audit):
    rules = parse_suffix_rules("test")
    record = {
        "record_id": "private-id",
        "raw_url": "https://sub.example.test/secret",
        "registrable_domain": "example.test",
    }
    assert audit.identity(record, rules) == ("private-id", "example.test")
    assert audit.identity(
        {"record": record, "probability": "must_not_read"}, rules
    ) == ("private-id", "example.test")


@pytest.mark.parametrize(
    "record",
    [
        {
            "record_id": "identifier",
            "raw_url": "https://actual.test/",
            "registrable_domain": "wrong.test",
        },
        {"record_id": "identifier", "registrable_domain": "sub.example.test"},
        {"raw_url": "https://example.test/", "registrable_domain": "example.test"},
        {"record_id": ""},
    ],
)
def test_unresolvable_schema_or_domain_fails(audit, record):
    with pytest.raises(ValueError):
        audit.identity(record, parse_suffix_rules("test"))


def test_id_only_predictions_require_an_unambiguous_reference(audit, tmp_path):
    path = tmp_path / "validation-predictions.jsonl"
    path.write_text(
        json.dumps({"record_id": "private-id", "probability": "ignored"}) + "\n"
    )
    entry, domains = audit.scan(
        path, tmp_path, parse_suffix_rules("test"), {"private-id": "example.test"}
    )
    assert domains == {"example.test"}
    assert entry["identity_joined_rows"] == 1
    assert entry["unresolved_rows"] == 0
    assert "private-id" not in json.dumps(entry)
    assert "example.test" not in json.dumps(entry)
    entry, domains = audit.scan(path, tmp_path, parse_suffix_rules("test"), {})
    assert domains == set()
    assert entry["unresolved_rows"] == 1


def test_disagreement_with_known_identity_is_not_silently_accepted(audit, tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text(
        json.dumps({"record_id": "known", "registrable_domain": "new.test"}) + "\n"
    )
    with pytest.raises(ValueError, match="identity domain disagreement"):
        audit.scan(path, tmp_path, parse_suffix_rules("test"), {"known": "old.test"})


def test_candidate_bound_excludes_additional_history(audit):
    from retained_domain_preflight import domain_summary
    from test_retained_domain_preflight import document

    source = document([("train", "phishing", "new.test", "https://new.test/")])
    result = domain_summary(
        source,
        parse_suffix_rules("test"),
        set(),
        set(),
        historical_domains={"new.test"},
    )
    assert result["groups"][0]["candidate_upper_bound_domains"] == 0
    assert result["groups"][0]["exclusion_incidences"]["historical_domain_overlap"] == 1


def test_discovery_includes_original_group_test_without_quarantine(audit, tmp_path):
    partition = (
        tmp_path
        / "gwu_working/automated-phishing-detection-public/data/processed/phiusiil-v1"
    )
    partition.mkdir(parents=True)
    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "group_test.jsonl",
        "quarantine.jsonl",
    ):
        (partition / filename).write_text("{}\n")
    assert {path.name for path in audit.discover(tmp_path)} == {
        "train.jsonl",
        "validation.jsonl",
        "group_test.jsonl",
    }
