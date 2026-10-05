import csv
import io
from collections import Counter
from importlib import import_module

import pytest

from automated_phishing_detection.protocol_preflight import parse_suffix_rules


def module():
    return import_module("automated_phishing_detection.followup_population")


def source(rows, fields=("url", "status", "ignored_publisher_feature")):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(fields)
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def prepare(rows, **kwargs):
    return module().prepare_population(
        source(rows), parse_suffix_rules("com\nexample\nblogspot.com\n"), **kwargs
    )


def test_only_raw_url_and_publisher_class_are_used():
    result = prepare(
        [
            ("HTTPS://Safe.Example/Case", "legitimate", "arbitrary"),
            ("http://risk.example/login", "phishing", "not a number"),
        ],
        blocked_domains={"phiusiil": set(), "phishvn": set()},
    )
    assert [row["is_phishing"] for row in result["retained"]] == [0, 1]
    assert result["retained"][0]["raw_url"] == "HTTPS://Safe.Example/Case"
    assert result["retained"][0]["registrable_domain"] == "safe.example"
    assert all("ignored_publisher_feature" not in row for row in result["retained"])
    assert result["summary"]["admitted"] is False


def test_overlap_covers_subdomains_and_preserves_reason_incidences():
    result = prepare(
        [("https://sub.blocked.example/path", "phishing", 1)],
        blocked_domains={
            "phiusiil": {"blocked.example"},
            "phishvn": {"blocked.example"},
        },
    )
    assert result["retained"] == []
    assert result["quarantine"][0]["reasons"] == [
        "phiusiil_domain_overlap",
        "phishvn_domain_overlap",
    ]
    assert result["summary"]["overlap_before_filtering"] == {
        "phiusiil": 1,
        "phishvn": 1,
    }


def test_same_label_duplicate_keeps_first_and_conflicts_quarantine_every_copy():
    result = prepare(
        [
            ("https://same.example", "phishing", 1),
            ("https://SAME.example/", "phishing", 2),
            ("https://conflict.example", "phishing", 3),
            ("https://conflict.example/", "legitimate", 4),
        ],
        blocked_domains={},
    )
    assert [row["source_ordinal"] for row in result["retained"]] == [1]
    assert Counter(
        reason for row in result["quarantine"] for reason in row["reasons"]
    ) == {
        "same_label_canonical_duplicate": 1,
        "conflicting_canonical_labels": 2,
    }


@pytest.mark.parametrize(
    "url", ["not absolute", "https://127.0.0.1", "https://x.example/%QQ"]
)
def test_invalid_urls_are_preserved_and_never_repaired(url):
    result = prepare([(url, "phishing", 1)], blocked_domains={})
    assert result["retained"] == []
    assert result["quarantine"][0]["raw_url"] == url
    assert result["quarantine"][0]["reasons"] == ["invalid_url"]


@pytest.mark.parametrize("label", ["1", "0", "malicious", "Phishing", "", "phishing "])
def test_label_semantics_cannot_be_guessed(label):
    with pytest.raises(ValueError, match="publisher label"):
        prepare([("https://safe.example", label, 0)], blocked_domains={})


@pytest.mark.parametrize(
    "fields", [("URL", "status"), ("url", "label"), ("url", "status", "status")]
)
def test_unknown_or_duplicate_headers_stop_preparation(fields):
    with pytest.raises(ValueError, match="header"):
        module().prepare_population(
            source([], fields), parse_suffix_rules("com"), blocked_domains={}
        )


def test_admission_requires_both_class_row_and_domain_minima():
    rows = []
    for label in ("legitimate", "phishing"):
        for index in range(1000):
            rows.append((f"https://{label}{index % 250}.example/{index}", label, 0))
    result = prepare(rows, blocked_domains={})
    assert result["summary"]["admitted"] is True
    assert result["summary"]["retained_class_counts"] == {"0": 1000, "1": 1000}
    assert result["summary"]["retained_class_domain_counts"] == {"0": 250, "1": 250}
    assert prepare(rows[:-1], blocked_domains={})["summary"]["admitted"] is False


def test_private_suffixes_do_not_merge_different_tenants():
    result = prepare(
        [
            ("https://one.blogspot.com", "phishing", 0),
            ("https://two.blogspot.com", "legitimate", 0),
        ],
        blocked_domains={"phiusiil": {"one.blogspot.com"}},
    )
    assert [row["registrable_domain"] for row in result["retained"]] == [
        "two.blogspot.com"
    ]
