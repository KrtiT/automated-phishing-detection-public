"""Synthetic checks for conservative domain-level feasibility, not admission."""

import json

import pytest
from retained_domain_preflight import domain_summary

from automated_phishing_detection.protocol_preflight import parse_suffix_rules


def document(entries):
    fields = [
        "split",
        "source",
        "tier",
        "label",
        "domain",
        "url_norm",
        "collected_at",
        "label_source",
    ]
    return {
        "headers": {"table": fields},
        "rows": [
            {
                "source_member": "table",
                "cells": [split, "registry", "gold", label, host, url, "", "registry"],
            }
            for split, label, host, url in entries
        ],
    }


def audit(entries, original=(), exposed=()):
    return domain_summary(
        document(entries),
        parse_suffix_rules("test\nco.test"),
        set(original),
        set(exposed),
    )


def test_subdomains_are_not_independent_units_and_no_identifiers_escape():
    result = audit(
        [
            (
                "train",
                "phishing",
                "one.site.co.test",
                "https://one.site.co.test/secret",
            ),
            (
                "train",
                "phishing",
                "two.site.co.test",
                "https://two.site.co.test/secret",
            ),
        ]
    )
    group = result["groups"][0]
    assert group["valid_rows"] == 2
    assert group["registered_domains"] == 1
    assert group["candidate_upper_bound_domains"] == 1
    assert group["largest_domain_rows"] == 2
    assert "site.co.test" not in json.dumps(result)
    assert "secret" not in json.dumps(result)


def test_all_known_exposure_and_split_overlap_are_excluded():
    result = audit(
        [
            ("train", "phishing", "prior.test", "https://prior.test/a"),
            ("train", "phishing", "followup.test", "https://followup.test/a"),
            ("train", "phishing", "oldtest.test", "https://oldtest.test/a"),
            ("test", "phishing", "oldtest.test", "https://oldtest.test/b"),
            ("train", "phishing", "shared.test", "https://shared.test/a"),
            ("val", "phishing", "shared.test", "https://shared.test/b"),
            ("train", "phishing", "fresh.test", "https://fresh.test/a"),
        ],
        original=["prior.test"],
        exposed=["followup.test"],
    )
    group = next(item for item in result["groups"] if item["split"] == "train")
    assert group["candidate_upper_bound_rows"] == 1
    assert group["candidate_upper_bound_domains"] == 1
    assert group["exclusion_incidences"] == {
        "original_domain_overlap": 1,
        "followup_domain_overlap": 1,
        "publisher_test_domain": 1,
        "train_validation_domain_overlap": 1,
    }


def test_duplicate_and_conflicting_labels_cannot_inflate_yield():
    result = audit(
        [
            ("train", "phishing", "fresh.test", "https://fresh.test"),
            ("train", "phishing", "fresh.test", "https://fresh.test/"),
            ("train", "phishing", "conflict.test", "https://conflict.test/a"),
            ("train", "benign", "conflict.test", "https://conflict.test/a"),
        ]
    )
    group = next(item for item in result["groups"] if item["label"] == "phishing")
    assert group["candidate_upper_bound_rows"] == 2
    assert group["candidate_upper_bound_unique_urls"] == 1
    assert group["conflicting_canonical_label_rows"] == 1
    assert group["canonical_duplicate_excess"] == 1


def test_domain_mismatch_and_unsupported_urls_are_not_candidates():
    group = audit(
        [
            ("train", "phishing", "wrong.test", "https://actual.test/a"),
            ("train", "phishing", "127.0.0.1", "http://127.0.0.1/a"),
        ]
    )["groups"][0]
    assert group["candidate_upper_bound_rows"] == 0
    assert group["invalid_url_rows"] == 1
    assert group["publisher_hostname_mismatch_rows"] == 1


def test_combined_primary_sources_count_shared_domains_once():
    source = document(
        [
            ("train", "benign", "one.shared.test", "https://one.shared.test/"),
            ("train", "benign", "two.shared.test", "https://two.shared.test/"),
            ("train", "phishing", "fresh.test", "https://fresh.test/"),
        ]
    )
    for row, name in zip(
        source["rows"], ("tinnhiem_org", "tinnhiem_web", "tinnhiemmang"), strict=True
    ):
        row["cells"][1] = name
    result = domain_summary(source, parse_suffix_rules("test"), set(), set())
    assert (
        result["primary_candidate_upper_bounds"]["certified_negative"][
            "registered_domains"
        ]
        == 1
    )
    assert result["primary_candidate_upper_bounds"]["certified_negative"]["rows"] == 2
    assert (
        result["primary_candidate_upper_bounds"]["gold_positive"]["registered_domains"]
        == 1
    )


@pytest.mark.parametrize("split,label", [("unknown", "phishing"), ("train", "unknown")])
def test_unknown_strata_fail_instead_of_silent_label_mapping(split, label):
    with pytest.raises(ValueError, match="unrecognized"):
        audit([(split, label, "fresh.test", "https://fresh.test/a")])
