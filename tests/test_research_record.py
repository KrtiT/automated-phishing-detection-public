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
SAGA_DIAGNOSTIC = ROOT / "scripts" / "rq1_saga_convergence_diagnostic.py"
DECISION_MATRIX_SHA256 = (
    "bb8350b12ba5f16737826d819b3bc8e2f6a1626851d1da0109828c8a52bed526"
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


def _compact(text: str) -> str:
    return " ".join(text.lower().split())


def test_recorded_protocol_hash_matches_current_protocol():
    expected = sha256(PROTOCOL.read_bytes()).hexdigest()
    pattern = re.compile(r"\| Protocol SHA-256 \| `([0-9a-f]{64})` \|")

    for relative_path in (
        "docs/advisor-approval/approval-status.md",
        "docs/research-evidence-outline.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert pattern.findall(text) == [expected]


def test_protocol_records_current_saga_diagnostic_hash():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    expected = sha256(SAGA_DIAGNOSTIC.read_bytes()).hexdigest()
    pattern = re.compile(
        r"`scripts/rq1_saga_convergence_diagnostic\.py`, SHA-256 "
        r"`([0-9a-f]{64})`"
    )

    assert pattern.findall(protocol) == [expected]


def test_protocol_v15_records_current_convergence_work_without_a_result():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    preamble = protocol.split("## Study Plan", maxsplit=1)[0]
    contract = _section(protocol, "RQ1 baseline contract", level=3)

    assert "**Version:** 1.5 | **Date:** 2026-09-04" in preamble
    assert "PhiUSIIL development-data preparation is complete." in preamble
    assert "rq1-baselines-v1" in preamble
    assert (
        "contract `rq1-baselines-v1`, SHA-256 "
        "`594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4`" in contract
    )
    assert "tolerance observation" in preamble
    assert "provenance-incomplete" in preamble
    assert "prospective SAGA convergence diagnostic is `not_run`" in preamble
    assert (
        "No baseline model, threshold, or validation result has been accepted"
        in preamble
    )
    assert "H1, H2, and H3 remain undecided" in preamble
    assert "No PhishVN record has been accessed" in preamble


def test_live_records_preserve_v14_failure_and_qualify_tolerance_observation():
    audit_sections = (
        (
            STATUS,
            "Execution Audit",
            2,
            "| RQ1 baseline contract | `rq1-baselines-v1` |",
        ),
        (
            EVIDENCE_OUTLINE,
            "RQ1 Baseline Execution Note",
            3,
            "| RQ1 baseline contract | `data/rq1-baseline-contract.json` "
            "(`rq1-baselines-v1`) |",
        ),
    )
    for record, heading, level, contract_row in audit_sections:
        text = record.read_text(encoding="utf-8")
        audit = _section(text, heading, level=level)

        assert re.search(
            r"(?:protocol[^.]*v1\.4|v1\.4[^.]*protocol)",
            audit,
            flags=re.IGNORECASE,
        )
        assert "`rq1-baselines-v1`" in audit
        assert "| Protocol version | `1.5` |" in text
        assert contract_row in text
        assert (
            "| RQ1 baseline contract SHA-256 | "
            "`594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4` |"
            in text
        )
        assert "| Development source schema | `2` |" in text
        assert "| Source-freeze release tag | `phiusiil-development-v1` |" in text
        assert (
            "https://github.com/KrtiT/automated-phishing-detection-public/"
            "releases/tag/phiusiil-development-v1" in text
        )
        assert "| Development preparation | `complete` |" in text
        assert "`stopped_nonconverged`" in text
        assert "`max_iter=5000`" in audit
        assert "`tol=1e-8`" in audit
        assert (
            "2c2956e7cf958f9d2d948a2b1b665e214e12b175d84e4b73c766cc0a6e3be4de" in audit
        )
        assert (
            "594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4" in audit
        )
        assert "c79e8aefb47560c6ae982dbd5848cf2b707c99a4" in audit
        assert "error: Logistic-L1 did not converge" in audit
        assert "Atomic publication left no model" in audit
        assert "e535586c6162a306a8dac7a5a6546f55dc09136f" in audit
        assert "67107874b9e46457ed710db42f050e35c1ca5ea2" in audit
        tolerance_record = re.search(
            r"A later exploratory local tolerance check.*?not research evidence\.",
            audit,
            flags=re.IGNORECASE | re.DOTALL,
        )
        assert tolerance_record is not None
        compact_audit = _compact(tolerance_record.group())
        for diagnostic_fact in (
            "intended to use training data only",
            "`tol=1e-4`",
            "`n_iter=5000`",
            "`convergencewarning=true`",
            "`22,119.35` seconds",
            "`elapsed_seconds=22119.348848833004`",
            "exact command, executed code, environment, and raw console record were "
            "not preserved",
            "constructed the estimator from a hard-coded `tol=1e-8`",
            "cannot verify that `tol=1e-4` reached the fitted estimator",
            "cannot independently verify its input boundary",
            "provenance-incomplete",
            "does not establish that tolerance alone failed",
            "not research evidence",
        ):
            assert diagnostic_fact in compact_audit
        assert "H1 `undecided`; H2 `undecided`; H3 `undecided`" in text
        assert "No PhishVN record" in text


def test_next_saga_diagnostic_is_prospective_and_training_only():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    diagnostic = _compact(
        _section(protocol, "Prospective convergence diagnostic", level=3)
    )
    for rule in (
        "one two-model saga diagnostic",
        "executed twice in fresh processes",
        "coordinator launches both fresh-process executions sequentially",
        "reports the overall diagnostic as `passed` only after both runs pass",
        "`length-only` uses only `raw_url_codepoint_length`",
        "`logistic-l1` uses all 25 predictors",
        '`solver="saga"`',
        '`penalty="l1"`',
        "`c=1.0`",
        '`class_weight="balanced"`',
        "`fit_intercept=true`",
        "`max_iter=5000`",
        "`tol=1e-4`",
        "`random_state=42`",
        "training data only",
        "saga leaves its intercept unpenalized",
        "individual coefficients and the selected sparsity pattern will not be "
        "interpreted as feature importance",
        "`raw_url_codepoint_length = raw_url_ascii_letter_count + "
        "raw_url_ascii_digit_count + raw_url_other_codepoint_count`",
        "no validation, group-test, or phishvn input",
        "publishes no model or summary",
        "both models in both runs must complete without a warning",
        "report `classes_=[0, 1]`",
        "`0 < n_iter < 5000`",
        "parameter shapes matching their declared feature counts",
        "produce only finite scaler values, fitted parameters, training decision "
        "scores, and training probabilities",
        "fresh-process repeat must reproduce each iteration count and fitted-state "
        "sha-256",
        "elapsed time and the nonzero-coefficient count are recorded but are not "
        "selection criteria",
        "coefficient values, signs, and sizes are not emitted or interpreted",
        "`scripts/rq1_saga_convergence_diagnostic.py`",
        "`575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0`",
        "`1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e`",
        "clean git head",
        "`uv.lock` sha-256",
        "fixed order: model name, ordered feature names, scaler configuration, "
        "classifier configuration, input hashes, scaler `mean_`, `scale_`, `var_`, "
        "and `n_samples_seen_`, then classifier `classes_`, `coef_`, `intercept_`, "
        "and `n_iter_`",
        "floating arrays are normalized to little-endian `float64` and integer "
        "arrays to little-endian `int64`",
        "compact utf-8 json with keys sorted and nonfinite values rejected",
        'the canonical object is exactly `{"ordered_state":[...],"schema_version":1}`',
        'each ordered-state entry is `{"name":<field name>,"value":<field value>}`',
        "each normalized array value is "
        '`{"dtype":"<f8" or "<i8","hex":<lowercase hex>,"shape":[...]}`',
        "`json.dumps(allow_nan=false, ensure_ascii=false, "
        'separators=(",", ":"), sort_keys=true)`',
    ):
        assert rule in diagnostic

    for record in (STATUS, EVIDENCE_OUTLINE):
        text = _compact(record.read_text(encoding="utf-8"))
        assert "prospective saga convergence diagnostic is `not_run`" in text
        assert "two-model saga diagnostic" in text
        assert "both models in both fresh-process runs" in text
        assert "fitted-state sha-256" in text
        assert (
            "no baseline model, threshold, or validation result has been accepted"
            in text
        )

    readme = README.read_text(encoding="utf-8")
    assert "uv run --locked python scripts/rq1_saga_convergence_diagnostic.py" in readme


def test_live_records_preserve_group_test_analyst_access_caveat():
    access_sections = (
        _section(STATUS.read_text(encoding="utf-8"), "Execution Audit"),
        _section(
            EVIDENCE_OUTLINE.read_text(encoding="utf-8"),
            "Internal Holdout Access Note",
            level=3,
        ),
    )
    for section in access_sections:
        section = " ".join(section.lower().split())
        for disclosure in (
            "analyst access",
            "no group-test prediction or metric was produced",
            "`access.group_test_accessed=false`",
            "describes only the `fit-baselines` process input boundary",
            "does not negate the analyst access recorded here",
        ):
            assert disclosure in section


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
        "Paired predictions on the PhiUSIIL group test; NCSC `gold` positives for primary external recall and certified trusted-registry negatives for primary external FPR.",
        "Window alerts plus paired fixed-cascade and prospective-policy predictions. Primary error inference uses NCSC `gold` positives and certified trusted-registry negatives.",
        "Paired recall on NCSC `gold` positives; primary FPR on certified trusted-registry negatives; the deterministic reference invocation trace; and five measured HTTP runs.",
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


def test_rq1_group_test_disclosure_and_single_frozen_pass_are_explicit():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    decision_matrix = _section(protocol, "Decision Matrix")
    rq1_rows = [
        line
        for line in decision_matrix.splitlines()
        if line.startswith("| **RQ1 / H1**")
    ]

    assert len(rq1_rows) == 1
    rq1_row = rq1_rows[0]
    group_test_rule = (
        "PhiUSIIL group test is held out and model-unscored but analyst-exposed; "
        "its raw partition receives exactly one later noninteractive frozen "
        "processing pass only "
        "after all four RQ1 models, thresholds, evaluator, manifest specifications "
        "and selection rules, software environment, and hashes are frozen"
    )
    assert rq1_row.count(group_test_rule) == 1
    assert "untouched phiusiil group test" not in protocol.lower()

    safeguards = _compact(_section(protocol, "Research Safeguards"))
    assert (
        "the evaluator accepts no tuning arguments and publishes atomically"
        in safeguards
    )
    assert "a failure leaves the affected hypotheses undecided" in safeguards
    assert "result-informed revision" in safeguards
    assert "does not reopen or rescan the raw group-test partition" in safeguards

    operational_replay = _compact(protocol)
    assert (
        "the same raw-partition pass produces the paired rq1 predictions and "
        "reference manifest" in operational_replay
    )
    assert "later http replay consumes only that frozen manifest" in operational_replay


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


def test_v15_change_record_describes_access_and_convergence_controls():
    status = STATUS.read_text(encoding="utf-8")
    change_record = _section(status, "Change Record")
    v15_rows = [line for line in change_record.splitlines() if "| 1.5 |" in line]

    assert len(v15_rows) == 1
    v15_row = v15_rows[0].lower()
    for detail in (
        "analyst-exposed but model-unscored",
        "one later frozen noninteractive raw-partition pass",
        "provenance-incomplete tolerance observation",
        "tol=1e-8",
        "tol=1e-4",
        "saga diagnostic",
        "no baseline result",
    ):
        assert detail in v15_row


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
