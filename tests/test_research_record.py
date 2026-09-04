import json
import re
from hashlib import sha256
from pathlib import Path

from automated_phishing_detection import phiusiil

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "advisor-approval" / "2026-08-16-realignment-matrix.md"
STATUS = ROOT / "docs" / "advisor-approval" / "approval-status.md"
EVIDENCE_OUTLINE = ROOT / "docs" / "research-evidence-outline.md"
README = ROOT / "README.md"
DECISION_MATRIX_SHA256 = (
    "8f23af76f4cf5bfb508f99726eed35d1fec3d9b1c20bf03a8cf070fc3511d208"
)


def _section(text: str, heading: str, level: int = 2) -> str:
    marker = f"{'#' * level} {heading}"
    match = re.search(
        rf"^{re.escape(marker)}\n(?P<body>.*?)(?=^#{{1,{level}}} |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing section: {marker}"
    return match.group("body")


def test_recorded_protocol_hash_matches_current_protocol():
    expected = sha256(PROTOCOL.read_bytes()).hexdigest()
    pattern = re.compile(r"\| Protocol SHA-256 \| `([0-9a-f]{64})` \|")

    for relative_path in (
        "docs/advisor-approval/approval-status.md",
        "docs/research-evidence-outline.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert pattern.findall(text) == [expected]


def test_protocol_v13_records_completed_preparation_and_next_milestone():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    preamble = protocol.split("## Study Plan", maxsplit=1)[0]

    assert "**Version:** 1.3 | **Date:** 2026-09-03" in preamble
    assert (
        "PhiUSIIL development-data preparation and its aggregate record are complete."
        in preamble
    )
    assert "frozen raw-URL feature extraction" in preamble
    assert "`length-only` and `Logistic-L1` baseline implementation" in preamble
    assert "using only the train and validation partitions" in preamble
    assert "H1, H2, and H3 remain undecided" in preamble
    assert "No PhishVN record has been accessed" in preamble

    for record in (STATUS, EVIDENCE_OUTLINE):
        text = record.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "| Protocol version | `1.3` |" in text
        assert "| Development source schema | `2` |" in text
        assert "| Source-freeze release tag | `phiusiil-development-v1` |" in text
        assert (
            "https://github.com/KrtiT/automated-phishing-detection-public/"
            "releases/tag/phiusiil-development-v1" in text
        )
        assert "| Development preparation | `complete` |" in text
        assert "frozen raw-url feature extraction" in lowered
        assert "`length-only` and `logistic-l1` baseline implementation" in lowered
        assert "H1 `undecided`; H2 `undecided`; H3 `undecided`" in text


def test_protocol_distinguishes_reference_classifications_from_ground_truth():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    outcome_contract = _section(protocol, "Outcome-label contract", level=3)
    mapping_rules = _section(
        protocol, "Mechanical mapping and quarantine rules", level=3
    )

    assert "[research basis](../research-basis.md)" in outcome_contract
    assert (
        "publisher-provided, source-derived reference classifications"
        in outcome_contract
    )
    assert "not independently verified ground truth" in outcome_contract
    assert (
        "Native label `0` maps to local `is_phishing=1`; native label `1` maps "
        "to local `is_phishing=0`." in mapping_rules
    )
    assert "quarantine an invalid or missing URL" in mapping_rules
    assert (
        "Quarantine the entire affected canonical-URL or registrable-domain group"
        in (mapping_rules)
    )
    assert (
        "Neither Krti nor another individual adjudicates an outcome or exception"
        in (mapping_rules)
    )


def test_contribution_and_gmm_claims_match_the_research_basis():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    contribution = _section(protocol, "Contribution boundary", level=3)
    lowered = contribution.lower()

    assert "[research basis](../research-basis.md)" in contribution
    assert "established components and closest current prior work" in contribution
    assert "prospective joint systems evaluation" in contribution
    assert "input distribution `P(X)`" in contribution
    assert "consistent with covariate/source shift" in contribution
    for unsupported_inference in (
        "harmful drift",
        "label shift",
        "concept drift",
        "causal performance degradation",
    ):
        assert unsupported_inference in contribution
    assert not re.search(r"\b(first|novel|unprecedented)\b", lowered)


def test_all_numerical_targets_are_identified_as_study_defined_gates():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    target_basis = _section(protocol, "Basis of Numerical Targets")

    assert "study-defined operating gates" in target_basis
    assert "not prescribed by the literature" in target_basis
    assert "not already achieved by prior work" in target_basis
    for gate in (
        "`<= 1%` FPR",
        "`<= 200 ms` p95 latency",
        "`< 0.1%` request errors",
        "`2000 ms` client timeout",
        "`>= 80%` shift-window detection",
        "`<= 5%` false alerts",
        "`-0.02` recall-noninferiority margin",
        "`<= 30%` transformer invocation",
    ):
        assert gate in target_basis


def test_rq_hypothesis_and_decision_gate_contracts_are_preserved():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    decision_matrix = _section(protocol, "Decision Matrix")

    assert sha256(decision_matrix.encode("utf-8")).hexdigest() == (
        DECISION_MATRIX_SHA256
    )

    critical_contracts = (
        "**RQ1:** What incremental value do structural URL features and character-level representations provide under registrable-domain-disjoint and external evaluation?",
        "**H1:** At the validation-selected FPR ceiling, full structural features improve recall over a length-only baseline, and the character-transformer cascade improves recall over `Logistic-L1`.",
        "**RQ2:** Can GMM-based monitoring detect an external source/domain shift and guide escalation without exceeding the low-FPR operating constraint?",
        "**H2:** GMM detects at least 80% of prespecified external shift windows at no more than 5% false alerts, and prospective routing reduces the false-negative rate relative to the fixed cascade while retaining FPR <= 1%.",
        "At each alert, route only the next 256 future requests through the transformer.",
        "**RQ3:** What detection, escalation, throughput, and latency tradeoffs determine whether the fixed cascade is viable inline?",
        "**H3:** For each system, observed FPR on certified trusted-registry negatives is <= 1%, and the Tranco reference-negative alert rate is <= 1% as a mandatory secondary safeguard.",
        "external-window detection is >= 80%",
        "independent reference false alerts are <= 5%",
        "must be >= `-0.02`",
        "stage 2 must be invoked for <= 30%",
        "pooled p95 across all measured concurrency-64 requests must be <= 200 ms",
        "all 50,000 measured concurrency-64 requests must be < 0.1%",
    )
    for contract in critical_contracts:
        assert contract in decision_matrix

    assert (
        decision_matrix.count(
            "observed FPR on certified trusted-registry negatives is <= 1% and the "
            "Tranco reference-negative alert rate is <= 1%"
        )
        == 1
    )


def test_protocol_records_published_source_freeze():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    safeguards = _section(protocol, "Research Safeguards")

    assert "GitHub Release is the source-freeze record for this milestone" in safeguards
    assert "exact licensed UCI source archive outside Git history" in safeguards
    assert "archive and CSV SHA-256 checksums" in safeguards
    assert (
        "https://github.com/KrtiT/automated-phishing-detection-public/"
        "releases/tag/phiusiil-development-v1" in safeguards
    )


def test_v13_change_record_describes_clarification_without_claiming_results():
    status = STATUS.read_text(encoding="utf-8")
    change_record = _section(status, "Change Record")
    v13_rows = [line for line in change_record.splitlines() if "| 1.3 |" in line]

    assert len(v13_rows) == 1
    v13_row = v13_rows[0].lower()
    for clarification in ("claim", "provenance", "target basis"):
        assert clarification in v13_row
    assert "no experiment was run" in v13_row


def test_readme_links_research_basis_and_limits_synthetic_urls_to_unit_tests():
    readme = README.read_text(encoding="utf-8")
    readme_prose = " ".join(readme.split())

    assert "[research basis](docs/research-basis.md)" in readme
    assert "Invented or synthetic URLs are unit-test fixtures only" in readme_prose
    assert "never research observations" in readme_prose


def test_public_research_records_exclude_stale_or_approval_gating_language():
    records = (PROTOCOL, STATUS, EVIDENCE_OUTLINE, README)
    combined = "\n".join(path.read_text(encoding="utf-8") for path in records).lower()

    for stale_phrase in (
        "pending approval",
        "written approval",
        "cannot proceed",
        "inherited operational constraints",
        "preparation is the current implementation step",
        "current work is limited to phiusiil development-data preparation",
        "phiusiil-development-v1` (planned)",
        "the planned github release",
        "publication occurs only when that release is created",
    ):
        assert stale_phrase not in combined


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
