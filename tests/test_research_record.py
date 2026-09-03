import json
import re
from hashlib import sha256
from pathlib import Path

from automated_phishing_detection import phiusiil

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "advisor-approval" / "2026-08-16-realignment-matrix.md"


def test_recorded_protocol_hash_matches_current_protocol():
    expected = sha256(PROTOCOL.read_bytes()).hexdigest()
    pattern = re.compile(r"\| Protocol SHA-256 \| `([0-9a-f]{64})` \|")

    for relative_path in (
        "docs/advisor-approval/approval-status.md",
        "docs/research-evidence-outline.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert pattern.findall(text) == [expected]


def test_published_summary_matches_frozen_algorithms_and_count_invariants():
    summary = json.loads(
        (ROOT / "reports" / "phiusiil-preparation-summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["algorithms"] == {
        "allocation_basis": "unique_ascii_domain_groups",
        "allocation_version": "hamilton-largest-remainder-v1",
        "canonicalization_version": phiusiil.CANONICAL_URL_VERSION,
        "domain_split_version": phiusiil.DOMAIN_SPLIT_VERSION,
        "record_identifier_version": "phiusiil-row-v1",
        "seed": phiusiil.SPLIT_SEED,
        "split_percentages": dict(zip(phiusiil.SPLITS, phiusiil.SPLIT_WEIGHTS)),
    }

    overall = summary["overall_counts"]
    assert overall["input_rows"] == (
        overall["retained_rows"] + overall["quarantined_rows"]
    )
    assert overall["retained_rows"] == sum(
        split["row_count"] for split in summary["splits"].values()
    )
    assert overall["retained_domains"] == sum(
        split["domain_count"] for split in summary["splits"].values()
    )
    assert overall["quarantined_rows"] == sum(
        summary["quarantine_reason_counts"].values()
    )
    assert sum(summary["native_label_counts"].values()) == overall["input_rows"]
    assert sum(summary["local_label_counts"].values()) == overall["retained_rows"]
