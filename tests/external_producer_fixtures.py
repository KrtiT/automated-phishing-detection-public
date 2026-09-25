"""Invented prepared publisher streams; no released records or model files."""

from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256

from automated_phishing_detection import phishvn, protocol_preflight

MAPPINGS = (
    ("ncsc", "phishing", "gold"),
    ("trusted_registry", "legitimate", "certified"),
    ("ncsc", "phishing", "silver"),
    ("chongluadao_openphish", "phishing", "bronze"),
    ("tranco", "reference_negative", "control"),
)


def prepared_external(count=5, *, quarantine_count=0):
    rows = tuple(
        phishvn.NormalizedExternalRow(
            f"row-{index}",
            "test",
            index + 1,
            f"https://host{index}.example{index}.com/path" if index < count else None,
            *MAPPINGS[index % len(MAPPINGS)],
            "test",
        )
        for index in range(count + quarantine_count)
    )
    return phishvn.prepare_external_rows(
        rows,
        published_split_counts={"train": 0, "val": 0, "test": count + quarantine_count},
        test_split="test",
        suffix_rules=protocol_preflight.parse_suffix_rules("com\n"),
        phiusiil_domains=frozenset(),
    )


def relink_rows(prepared, rows):
    private = dict(prepared.private_outputs)
    private["retained-test.jsonl"] = b"".join(
        phishvn._json_bytes(asdict(row)) for row in rows
    )
    summary = dict(prepared.public_summary)
    summary["private_sha256"] = {
        name: sha256(content).hexdigest() for name, content in private.items()
    }
    return replace(
        prepared, retained=tuple(rows), private_outputs=private, public_summary=summary
    )


def relink_quarantine(prepared, quarantine):
    private = dict(prepared.private_outputs)
    private["quarantine.jsonl"] = b"".join(
        phishvn._json_bytes(asdict(row)) for row in quarantine
    )
    reasons = Counter(reason for row in quarantine for reason in row.reason_codes)
    summary = prepared.public_summary | {
        "quarantined_rows": len(quarantine),
        "quarantined_test_rows": sum(row.source_split == "test" for row in quarantine),
        "valid_non_test_rows": prepared.public_summary["input_row_count"]
        - len(prepared.retained)
        - len(quarantine),
        "quarantine_reason_counts": dict(sorted(reasons.items())),
        "private_sha256": {
            name: sha256(content).hexdigest() for name, content in private.items()
        },
    }
    return replace(
        prepared,
        quarantine=tuple(quarantine),
        private_outputs=private,
        public_summary=summary,
    )


def prepare_normalized(rows):
    counts = {"train": 0, "val": 0, "test": 0} | dict(
        Counter(row.source_split for row in rows)
    )
    return phishvn.prepare_external_rows(
        rows,
        published_split_counts=counts,
        test_split="test",
        suffix_rules=protocol_preflight.parse_suffix_rules("com\n"),
        phiusiil_domains=frozenset(),
    )


def prepared_duplicate_canonical(*, duplicate_id=False):
    rows = tuple(
        phishvn.NormalizedExternalRow(
            "row-0" if duplicate_id else f"row-{index}",
            "test",
            index + 1,
            "https://shared.example.com/path",
            *MAPPINGS[0],
            "test",
        )
        for index in range(2)
    )
    return prepare_normalized(rows)
