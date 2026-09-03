from hashlib import sha256

import pytest

from automated_phishing_detection.phiusiil import (
    CANONICAL_URL_VERSION,
    DOMAIN_SPLIT_VERSION,
    SPLIT_SEED,
    PhiUSIILRow,
    PreparationError,
    allocate_domain_counts,
    assign_splits,
    canonicalize_url,
    record_id_for_row,
    resolve_rows,
)
from automated_phishing_detection.protocol_preflight import parse_suffix_rules

CSV_SHA256 = "a" * 64
SUFFIX_RULES = parse_suffix_rules("com\nexample\n")


def test_algorithm_identifiers_are_frozen():
    assert CANONICAL_URL_VERSION == "canonical-url-v1"
    assert DOMAIN_SPLIT_VERSION == "phiusiil-domain-split-v1"
    assert SPLIT_SEED == "20260816"


@pytest.mark.parametrize(
    ("raw_url", "expected"),
    [
        pytest.param(
            "HTTP://User:%7e@BÜCHER.Example.:00080/a/../b?x=%2f&x=1#Frag%aa",
            "http://User:%7E@xn--bcher-kva.example/a/../b?x=%2F&x=1#Frag%AA",
            id="all-normalizations",
        ),
        pytest.param(
            "https://EXAMPLE.com:00443",
            "https://example.com/",
            id="https-default-port-and-empty-path",
        ),
        pytest.param(
            "https://Example.com:00081/path?#",
            "https://example.com:81/path?#",
            id="nondefault-port-and-empty-delimiters",
        ),
    ],
)
def test_canonical_url_v1_goldens(raw_url, expected):
    assert canonicalize_url(raw_url) == expected


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("https://example.com/a/../b", "https://example.com/b"),
        ("https://example.com/?a=1&b=2", "https://example.com/?b=2&a=1"),
        ("https://example.com/#one", "https://example.com/#two"),
        ("https://example.com/%2F", "https://example.com//"),
        ("https://user@example.com/", "https://USER@example.com/"),
    ],
)
def test_canonical_url_v1_preserves_semantically_relevant_spelling(left, right):
    assert canonicalize_url(left) != canonicalize_url(right)


@pytest.mark.parametrize(
    "raw_url",
    [
        pytest.param("", id="empty"),
        pytest.param(None, id="not-string"),
        pytest.param(" https://example.com/", id="space"),
        pytest.param("https://example.com/\x00", id="control"),
        pytest.param(r"https://example.com\path", id="backslash"),
        pytest.param("ftp://example.com/", id="scheme"),
        pytest.param("example.com/path", id="relative"),
        pytest.param("https://127.0.0.1/", id="ip-literal"),
        pytest.param("https://example.com/%xy", id="malformed-percent"),
        pytest.param("https://example.com:/", id="empty-port"),
        pytest.param("https://example.com:not-a-port/", id="nonnumeric-port"),
    ],
)
def test_canonical_url_v1_rejects_ambiguous_or_unsupported_urls(raw_url):
    with pytest.raises(PreparationError):
        canonicalize_url(raw_url)


def test_record_id_uses_source_hash_and_one_based_hex_row_ordinal():
    assert record_id_for_row(CSV_SHA256, 1) == (
        f"phiusiil-row-v1:{CSV_SHA256}:0000000000000001"
    )
    assert record_id_for_row(CSV_SHA256, 0xABCDEF) == (
        f"phiusiil-row-v1:{CSV_SHA256}:0000000000abcdef"
    )


@pytest.mark.parametrize("cell", ["", " 0", "0 ", "00", "1.0", 0, 1, None])
def test_only_exact_zero_and_one_csv_cells_are_adapted(cell):
    resolution = resolve_rows(
        [PhiUSIILRow(1, "https://only.example/path", cell)],
        csv_sha256=CSV_SHA256,
        suffix_rules=SUFFIX_RULES,
    )

    assert resolution.retained == ()
    assert [item.reason_code for item in resolution.quarantine] == [
        "invalid_phiusiil_native_label"
    ]


def test_exact_native_labels_use_the_existing_reversed_mapping():
    resolution = resolve_rows(
        [
            PhiUSIILRow(1, "https://zero.example/path", "0"),
            PhiUSIILRow(2, "https://one.example/path", "1"),
        ],
        csv_sha256=CSV_SHA256,
        suffix_rules=SUFFIX_RULES,
    )

    assert [record.is_phishing for record in resolution.retained] == [1, 0]
    assert resolution.native_label_counts == {"0": 1, "1": 1, "invalid": 0}


def test_invalid_url_is_quarantined_individually_without_a_canonical_hash():
    resolution = resolve_rows(
        [
            PhiUSIILRow(1, "not absolute", "0"),
            PhiUSIILRow(2, "https://valid.example/", "1"),
        ],
        csv_sha256=CSV_SHA256,
        suffix_rules=SUFFIX_RULES,
    )

    assert len(resolution.retained) == 1
    assert resolution.quarantine[0].reason_code == "invalid_or_missing_url"
    assert resolution.quarantine[0].canonical_url_sha256 is None


def test_invalid_label_quarantines_every_member_of_the_canonical_group():
    resolution = resolve_rows(
        [
            PhiUSIILRow(1, "https://Login.Example:443/a", "0"),
            PhiUSIILRow(2, "https://login.example/a", " 0"),
        ],
        csv_sha256=CSV_SHA256,
        suffix_rules=SUFFIX_RULES,
    )

    assert resolution.retained == ()
    assert [item.reason_code for item in resolution.quarantine] == [
        "invalid_phiusiil_native_label",
        "invalid_phiusiil_native_label",
    ]


def test_conflicting_mapping_quarantines_every_member_of_the_canonical_group():
    resolution = resolve_rows(
        [
            PhiUSIILRow(1, "https://Login.Example:443/a", "0"),
            PhiUSIILRow(2, "https://login.example/a", "1"),
        ],
        csv_sha256=CSV_SHA256,
        suffix_rules=SUFFIX_RULES,
    )

    assert resolution.retained == ()
    assert [item.reason_code for item in resolution.quarantine] == [
        "canonical_url_conflicting_mapping",
        "canonical_url_conflicting_mapping",
    ]


def test_same_mapping_duplicate_selection_is_independent_of_iteration_order():
    rows = (
        PhiUSIILRow(9, "HTTPS://LOGIN.EXAMPLE:443/a%2f", "0"),
        PhiUSIILRow(3, "https://login.example/a%2F", "0"),
        PhiUSIILRow(7, "https://other.example/", "1"),
    )

    first = resolve_rows(rows, csv_sha256=CSV_SHA256, suffix_rules=SUFFIX_RULES)
    second = resolve_rows(
        reversed(rows), csv_sha256=CSV_SHA256, suffix_rules=SUFFIX_RULES
    )

    assert first == second
    assert [record.record_id for record in first.retained] == [
        record_id_for_row(CSV_SHA256, 3),
        record_id_for_row(CSV_SHA256, 7),
    ]
    assert first.quarantine[0].record_id == record_id_for_row(CSV_SHA256, 9)
    assert first.quarantine[0].reason_code == "canonical_url_duplicate_same_mapping"


def test_twenty_domains_apportion_to_exact_hamilton_counts():
    assert allocate_domain_counts(20) == {
        "train": 14,
        "validation": 3,
        "group_test": 3,
    }


def _two_class_domain_rows():
    rows = []
    for index in range(20):
        domain = f"domain-{index:02d}.example"
        rows.extend(
            [
                PhiUSIILRow(index * 2 + 1, f"https://left.{domain}/", "0"),
                PhiUSIILRow(index * 2 + 2, f"https://right.{domain}/", "1"),
            ]
        )
    return rows


def test_split_assignment_is_domain_disjoint_deterministic_and_two_class():
    resolution = resolve_rows(
        _two_class_domain_rows(),
        csv_sha256=CSV_SHA256,
        suffix_rules=SUFFIX_RULES,
    )

    assigned = assign_splits(resolution.retained)
    shuffled = assign_splits(tuple(reversed(resolution.retained)))

    assert assigned == shuffled
    domains_by_split = {
        split: {
            record.registrable_domain for record in assigned if record.split == split
        }
        for split in ("train", "validation", "group_test")
    }
    assert {split: len(domains) for split, domains in domains_by_split.items()} == {
        "train": 14,
        "validation": 3,
        "group_test": 3,
    }
    assert not (domains_by_split["train"] & domains_by_split["validation"])
    assert not (domains_by_split["train"] & domains_by_split["group_test"])
    assert not (domains_by_split["validation"] & domains_by_split["group_test"])
    for split in domains_by_split:
        assert {record.is_phishing for record in assigned if record.split == split} == {
            0,
            1,
        }

    ranked_domains = sorted(
        set().union(*domains_by_split.values()),
        key=lambda domain: (
            sha256(
                b"phiusiil-domain-split-v1\0" + b"20260816\0" + domain.encode("ascii")
            ).digest(),
            domain,
        ),
    )
    assert domains_by_split["train"] == set(ranked_domains[:14])
    assert domains_by_split["validation"] == set(ranked_domains[14:17])
    assert domains_by_split["group_test"] == set(ranked_domains[17:])


def test_assignment_fails_instead_of_rebalancing_single_class_splits():
    rows = [
        PhiUSIILRow(index + 1, f"https://domain-{index:02d}.example/", "0")
        for index in range(20)
    ]
    resolution = resolve_rows(rows, csv_sha256=CSV_SHA256, suffix_rules=SUFFIX_RULES)

    with pytest.raises(PreparationError, match="both classes"):
        assign_splits(resolution.retained)
