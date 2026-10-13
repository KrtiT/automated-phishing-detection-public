"""Identity-only cohort selection cannot depend on prediction outcomes."""

import hashlib
import json

import pytest
import revision_population as population

from automated_phishing_detection.protocol_preflight import parse_suffix_rules


def document(entries):
    headers = [
        "split",
        "source",
        "tier",
        "label",
        "domain",
        "url_norm",
        "collected_at",
        "label_source",
    ]
    rows = []
    for ordinal, entry in enumerate(entries):
        values = {
            "split": "train",
            "source": "tinnhiemmang",
            "tier": "gold",
            "label": "phishing",
            "domain": "positive.test",
            "url_norm": "https://positive.test/",
            "collected_at": "2025-01-01",
            "label_source": "publisher",
            **entry,
        }
        rows.append(
            {
                "source_member": "table",
                "file_position": ordinal + 2,
                "cells": [values[name] for name in headers],
            }
        )
    return {"headers": {"table": headers}, "rows": rows}


def benign(**changes):
    return {
        "source": "tinnhiem_web",
        "label": "benign",
        "domain": "negative.test",
        "url_norm": "https://negative.test/",
        "collected_at": "",
        **changes,
    }


def select(entries, excluded=(), expected=None):
    return population.select_cohort(
        document(entries),
        parse_suffix_rules("test\nco.test"),
        set(excluded),
        expected_counts=expected or {"gold_positive": 1, "certified_negative": 1},
    )


def test_requires_the_selector_implementation():
    assert callable(getattr(population, "select_cohort", None))


def test_preserves_exact_input_and_source_locator_with_domain_balanced_rows():
    entries = [
        {"url_norm": "https://positive.test/A?Token=X"},
        {"url_norm": "https://positive.test/Z?Token=Y"},
        benign(),
    ]
    rows, summary = select(entries)
    assert len(rows) == 2
    positive = next(row for row in rows if row["is_phishing"] == 1)
    expected = min(
        entries[:2],
        key=lambda row: hashlib.sha256(
            ("advisor-review-20261012:row:" + row["url_norm"]).encode()
        ).hexdigest(),
    )["url_norm"]
    assert positive["raw_url"] == expected
    assert positive["source_locator"]["source_member"] == "table"
    assert positive["source_locator"]["file_position"] in {2, 3}
    assert summary["counts"] == {"gold_positive": 1, "certified_negative": 1}
    assert summary["strata"]["certified_negative"]["missing_date_rows"] == 1
    assert summary["model_predictions"] == 0
    assert summary["execution_authorized"] is False
    serialized = json.dumps(summary)
    assert "positive.test" not in serialized and "Token" not in serialized


def test_representative_tie_uses_source_order_but_not_label_or_length():
    rows, unused = select(
        [
            {"url_norm": "https://positive.test", "collected_at": "2025-01-02"},
            {"url_norm": "https://positive.test/", "collected_at": "2025-01-03"},
            benign(),
        ]
    )
    positive = next(row for row in rows if row["is_phishing"] == 1)
    assert positive["raw_url"] == "https://positive.test"
    assert positive["source_locator"]["publisher_ordinal"] == 0


def test_exposure_test_and_train_validation_domains_are_excluded():
    entries = [{}, benign()]
    for domain in ("exposed.test", "testonly.test", "crosssplit.test"):
        entries.append({"domain": domain, "url_norm": f"https://{domain}/"})
    entries.extend(
        [
            {
                "domain": "testonly.test",
                "url_norm": "https://testonly.test/b",
                "split": "test",
            },
            {
                "domain": "crosssplit.test",
                "url_norm": "https://crosssplit.test/b",
                "split": "val",
            },
        ]
    )
    rows, unused = select(entries, excluded={"exposed.test"})
    assert {row["registrable_domain"] for row in rows} == {
        "positive.test",
        "negative.test",
    }


def test_unqualified_tier_missing_label_source_and_host_mismatch_excluded():
    rows, unused = select(
        [
            {},
            benign(),
            {
                "domain": "silver.test",
                "url_norm": "https://silver.test/",
                "tier": "silver",
            },
            {
                "domain": "unknown.test",
                "url_norm": "https://unknown.test/",
                "label_source": " ",
            },
            {"domain": "wrong.test", "url_norm": "https://mismatch.test/"},
        ]
    )
    assert len(rows) == 2


def test_conflicting_canonical_labels_are_excluded():
    rows, unused = select(
        [
            {},
            benign(),
            {"domain": "conflict.test", "url_norm": "https://conflict.test/"},
            benign(domain="conflict.test", url_norm="https://conflict.test/"),
        ]
    )
    assert len(rows) == 2


def test_cross_label_domain_is_a_hold_not_an_arbitrary_class_choice():
    with pytest.raises(ValueError, match="cross_label_domain"):
        select([{}, benign(domain="positive.test", url_norm="https://positive.test/b")])


def test_count_mismatch_holds_complete_cohort():
    with pytest.raises(ValueError, match="cohort_count_mismatch"):
        select([{}])


def test_fixed_order_and_identifiers_are_stable_on_unique_input_permutation():
    first, unused = select([{}, benign()])
    second, unused = select([benign(), {}])
    fields = ("record_id", "registrable_domain", "raw_url", "is_phishing")
    assert [tuple(row[name] for name in fields) for row in first] == [
        tuple(row[name] for name in fields) for row in second
    ]
    assert len({row["record_id"] for row in first}) == len(first)


def test_workload_uses_identity_order_and_fixed_labels_not_predictions():
    rows = [
        {"record_id": f"row-{number}", "is_phishing": number % 2} for number in range(8)
    ]
    workload = population.select_workload(rows, positives=2, negatives=3)
    assert [row["record_id"] for row in workload] == [
        "row-0",
        "row-1",
        "row-2",
        "row-3",
        "row-4",
    ]
    with pytest.raises(ValueError, match="workload_shortage"):
        population.select_workload(rows, positives=5, negatives=3)


@pytest.mark.parametrize(
    "change", ["new", "missing", "changed", "unresolved", "bool_count"]
)
def test_roster_verification_fails_closed(change):
    baseline = {
        "inventory": [
            {
                "path": "prior/a.jsonl",
                "sha256": "a" * 64,
                "rows": 1,
                "registered_domains": 1,
                "identity_joined_rows": 0,
                "unresolved_rows": 0,
            }
        ],
        "input_sha256": {"source": "b" * 64},
        "unresolved_identity_rows": 0,
    }
    current = json.loads(json.dumps(baseline))
    if change == "new":
        current["inventory"].append(
            {**current["inventory"][0], "path": "prior/b.jsonl"}
        )
    elif change == "missing":
        current["inventory"] = []
    elif change == "changed":
        current["inventory"][0]["sha256"] = "c" * 64
    elif change == "unresolved":
        current["unresolved_identity_rows"] = 1
    else:
        current["inventory"][0]["rows"] = True
    with pytest.raises(ValueError, match="exposure_roster"):
        population.verify_roster(baseline, current)


def test_matching_roster_does_not_grant_execution():
    roster = {"inventory": [], "input_sha256": {}, "unresolved_identity_rows": 0}
    assert population.verify_roster(roster, roster) is None
