import json
import re
from hashlib import sha256
from math import isfinite
from pathlib import Path

from automated_phishing_detection import phiusiil

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs" / "advisor-approval" / "2026-08-16-realignment-matrix.md"
STATUS = ROOT / "docs" / "advisor-approval" / "approval-status.md"
EVIDENCE_OUTLINE = ROOT / "docs" / "research-evidence-outline.md"
RESEARCH_BASIS = ROOT / "docs" / "research-basis.md"
README = ROOT / "README.md"
SAGA_DIAGNOSTIC = ROOT / "scripts" / "rq1_saga_convergence_diagnostic.py"
SAGA_V1_RECEIPT = ROOT / "reports" / "rq1-saga-convergence-v1-execution.json"
SAGA_V2_RECEIPT = ROOT / "reports" / "rq1-saga-convergence-v2-execution.json"
BASELINE_V2_SUMMARY = ROOT / "reports" / "rq1-baseline-v2-summary.json"
BASELINE_V1_CONTRACT = ROOT / "data" / "rq1-baseline-contract.json"
BASELINE_V2_CONTRACT = ROOT / "data" / "rq1-baseline-contract-v2.json"
TRANSFORMER_CONTRACT = ROOT / "data" / "rq1-transformer-cascade-contract-v1.json"
TRANSFORMER_CONTRACT_V2 = ROOT / "data" / "rq1-transformer-cascade-contract-v2.json"
GMM_CONTRACT = ROOT / "data" / "rq2-gmm-development-contract-v1.json"
GMM_SUMMARY = ROOT / "reports" / "rq2-gmm-development-v1-summary.json"
TRANSFORMER_EXECUTION = ROOT / "reports" / "rq1-transformer-cascade-v2-execution.json"
FINAL_TRANSFORMER_CODE_COMMIT = "0793ca3dbc36e49b561cd0ac74968a4644060426"
INITIAL_TRANSFORMER_CODE_COMMIT = "a8ee067bda8fd45d19f5c4b794ba21f58d1947fc"
TRANSFORMER_CONTRACT_SHA256 = (
    "aeaa84534c4cadf0459cf6d2f010dc802684d4801cce563ce18242f36359fb54"
)
TRANSFORMER_CONTRACT_V2_SHA256 = (
    "686c0d86b33b8a6c2e09cd6e174003db0bd2f7c30b087faf5470e6a270524213"
)
V19_MATRIX_SHA256 = "f24eac919cb79d24d2248a94b3a74208f7b4d809ad778b963ad2e62315d78a38"
SEPTEMBER_REPORT_SHA256 = (
    "b72da89a4cc8a5b06f6ca88d79fe78dd54e3199a96b7450209ea53b4a4c04215"
)
DECISION_MATRIX_SHA256 = (
    "ae46e3cb883f8335c9b2fbca1d753e4830a94d3d4b9cf08d1447dcc9d7e493ec"
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


def test_v15_saga_failure_receipt_is_aggregate_and_immutable():
    assert sha256(SAGA_V1_RECEIPT.read_bytes()).hexdigest() == (
        "7309f52f704150f85e6c17d44d96adcde917d2539c7b75264bf775ccec3aa6f4"
    )
    receipt = json.loads(SAGA_V1_RECEIPT.read_bytes())

    assert receipt["diagnostic_id"] == "rq1-saga-convergence-v1"
    assert receipt["status"] == "failed"
    assert receipt["environment"] == {
        "git_head": "cbc62f1715d8685ac5c91d49973b5602253cdefd",
        "tracked_worktree_clean": True,
        "uv_lock_sha256": (
            "15fadb4ad1f3c702a902b40d587a55294e7a26c33f268e8708ba8d941e6a51f0"
        ),
    }
    assert receipt["failure"] == {
        "message": "one or more fresh runs failed",
        "runs": [
            {
                "failure": {
                    "message": "divide by zero encountered in matmul",
                    "model_name": "Logistic-L1",
                    "type": "RuntimeWarning",
                },
                "run_number": run_number,
            }
            for run_number in (1, 2)
        ],
        "type": "FreshRunFailure",
    }
    assert len(receipt["runs"]) == 2
    for run in receipt["runs"]:
        assert run["run_passed"] is False
        assert run["models"] == {
            "length-only": {
                "elapsed_seconds": run["models"]["length-only"]["elapsed_seconds"],
                "feature_count": 1,
                "n_iter": 69,
                "nonzero_coefficient_count": 1,
                "state_sha256": (
                    "d6a510973bdcafb9c9baa07a80528efbb322dc3328ba0169b660c89de1542b3d"
                ),
            }
        }
        assert run["models"]["length-only"]["elapsed_seconds"] >= 0.0

    forbidden_keys = {
        "coefficients",
        "decision_scores",
        "domains",
        "probabilities",
        "raw_urls",
        "records",
    }

    def visit(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(receipt)


def test_v16_saga_pass_receipt_is_aggregate_and_immutable():
    assert sha256(SAGA_V2_RECEIPT.read_bytes()).hexdigest() == (
        "487714ea17a095e369d381da5a27452f3b263f1be1a05db2ebe061eecdefebdf"
    )
    receipt = json.loads(SAGA_V2_RECEIPT.read_bytes())

    assert receipt["diagnostic_id"] == "rq1-saga-convergence-v2"
    assert receipt["status"] == "passed"
    assert receipt["scope"] == "training_only"
    assert receipt["environment"] == {
        "git_head": "69a67d4e5cb81d49009d6a90e87f4c0c5f2cea87",
        "numpy_blas_name": "accelerate",
        "platform_machine": "arm64",
        "sys_platform": "darwin",
        "tracked_worktree_clean": True,
        "uv_lock_sha256": (
            "15fadb4ad1f3c702a902b40d587a55294e7a26c33f268e8708ba8d941e6a51f0"
        ),
    }
    assert len(receipt["runs"]) == 2
    expected_warnings = [
        {"category": "RuntimeWarning", "message": message, "stage": stage}
        for stage in ("decision_function", "predict_proba")
        for message in (
            "divide by zero encountered in matmul",
            "overflow encountered in matmul",
            "invalid value encountered in matmul",
        )
    ]
    for run in receipt["runs"]:
        assert run["run_passed"] is True
        assert run["models"]["length-only"] == {
            "allowed_warnings": [],
            "elapsed_seconds": run["models"]["length-only"]["elapsed_seconds"],
            "feature_count": 1,
            "max_absolute_decision_difference": 0.0,
            "max_absolute_probability_difference": 0.0,
            "n_iter": 69,
            "nonzero_coefficient_count": 1,
            "state_sha256": (
                "d6a510973bdcafb9c9baa07a80528efbb322dc3328ba0169b660c89de1542b3d"
            ),
        }
        assert run["models"]["Logistic-L1"] == {
            "allowed_warnings": expected_warnings,
            "elapsed_seconds": run["models"]["Logistic-L1"]["elapsed_seconds"],
            "feature_count": 25,
            "max_absolute_decision_difference": 4.263256414560601e-14,
            "max_absolute_probability_difference": 5.551115123125783e-16,
            "n_iter": 4783,
            "nonzero_coefficient_count": 22,
            "state_sha256": (
                "1884d7c72bb1ce88392318b1ddecd90c0d9cfa8b2210340743245e15e353be35"
            ),
        }
        assert run["models"]["length-only"]["elapsed_seconds"] >= 0.0
        assert run["models"]["Logistic-L1"]["elapsed_seconds"] >= 0.0

    forbidden_keys = {
        "coefficients",
        "decision_scores",
        "domains",
        "probabilities",
        "raw_urls",
        "records",
        "predictions",
    }

    def visit(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(receipt)


def test_v2_baseline_summary_is_aggregate_and_immutable():
    assert sha256(BASELINE_V2_SUMMARY.read_bytes()).hexdigest() == (
        "bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c"
    )
    summary = json.loads(BASELINE_V2_SUMMARY.read_bytes())

    assert set(summary) == {
        "access",
        "analysis_stage",
        "contract_id",
        "hypothesis_status",
        "input_counts",
        "input_hashes",
        "models",
        "pipeline",
        "schema_version",
        "software_versions",
    }
    assert summary["schema_version"] == 2
    assert summary["analysis_stage"] == "development_validation_only"
    assert summary["contract_id"] == "rq1-baselines-v2"
    assert summary["hypothesis_status"] == {
        "H1": "undecided",
        "H2": "undecided",
        "H3": "undecided",
    }
    assert summary["access"] == {
        "group_test_accessed": False,
        "phishvn_accessed": False,
    }
    assert summary["input_hashes"] == {
        "contract": "05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba",
        "preparation_summary": (
            "1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e"
        ),
        "train": "575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0",
        "validation": (
            "970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a"
        ),
    }
    assert summary["input_counts"] == {
        "train": {"0": 94373, "1": 71875, "rows": 166248},
        "validation": {"0": 20209, "1": 12486, "rows": 32695},
    }
    assert summary["pipeline"]["classifier"] == {
        "C": 1.0,
        "class": "LogisticRegression",
        "class_weight": "balanced",
        "fit_intercept": True,
        "max_iter": 5000,
        "penalty": "l1",
        "random_state": 42,
        "solver": "saga",
        "tol": 1e-4,
    }
    assert summary["pipeline"]["scoring_integrity_policy_id"] == (
        "rq1-scoring-integrity-v1"
    )

    length_only = summary["models"]["length-only"]
    assert length_only["artifact_sha256"] == (
        "b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799"
    )
    assert length_only["n_iter"] == [69]
    assert length_only["validation_scoring_audit"] == {
        "max_absolute_decision_difference": 0.0,
        "max_absolute_probability_difference": 0.0,
        "platform_identity": {
            "numpy_blas_name": "accelerate",
            "platform_machine": "arm64",
            "sys_platform": "darwin",
        },
        "warning_records": [],
    }
    assert length_only["validation_threshold"] == {
        "candidate_count": 221,
        "counts": {
            "false_negative": 8472,
            "false_positive": 135,
            "negative": 20209,
            "positive": 12486,
            "true_negative": 20074,
            "true_positive": 4014,
        },
        "fpr_upper_95": 0.007702192373035135,
        "observed_fpr": 0.006680191993666189,
        "recall": 0.3214800576645843,
        "status": "selected",
        "threshold": 0.7612031186147,
    }

    logistic = summary["models"]["Logistic-L1"]
    assert logistic["artifact_sha256"] == (
        "71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a"
    )
    assert logistic["n_iter"] == [4783]
    assert logistic["validation_scoring_audit"] == {
        "max_absolute_decision_difference": 2.1316282072803006e-14,
        "max_absolute_probability_difference": 3.3306690738754696e-16,
        "platform_identity": {
            "numpy_blas_name": "accelerate",
            "platform_machine": "arm64",
            "sys_platform": "darwin",
        },
        "warning_records": [
            {"category": "RuntimeWarning", "message": message, "stage": stage}
            for stage in ("decision_function", "predict_proba")
            for message in (
                "divide by zero encountered in matmul",
                "overflow encountered in matmul",
                "invalid value encountered in matmul",
            )
        ],
    }
    assert logistic["validation_threshold"] == {
        "candidate_count": 11279,
        "counts": {
            "false_negative": 197,
            "false_positive": 177,
            "negative": 20209,
            "positive": 12486,
            "true_negative": 20032,
            "true_positive": 12289,
        },
        "fpr_upper_95": 0.009915480854183582,
        "observed_fpr": 0.008758473947251225,
        "recall": 0.9842223290084895,
        "status": "selected",
        "threshold": 0.2670846328466124,
    }

    forbidden_keys = {
        "canonical_url",
        "canonical_url_sha256",
        "coefficients",
        "decision_scores",
        "domains",
        "feature_row",
        "feature_rows",
        "feature_vector",
        "feature_vectors",
        "intercept",
        "mean",
        "predictions",
        "probabilities",
        "raw_url",
        "raw_urls",
        "record_id",
        "record_ids",
        "records",
        "row_score",
        "row_scores",
        "scale",
        "variance",
    }
    forbidden_string_markers = (
        "http://",
        "https://",
        "phiusiil-row-v1:",
        "/Users/",
        ".example",
    )

    def visit(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            assert not any(marker in value for marker in forbidden_string_markers)

    visit(summary)


def test_protocol_v110_preserves_transformer_history_and_freezes_gmm_without_result():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    preamble = protocol.split("## Study Plan", maxsplit=1)[0]
    contract = _section(protocol, "RQ1 baseline contract", level=3)

    assert "**Version:** 1.10 | **Date:** 2026-09-17" in preamble
    assert "PhiUSIIL development-data preparation is complete." in preamble
    assert "rq1-baselines-v2" in preamble
    expected_contract_hash = sha256(BASELINE_V2_CONTRACT.read_bytes()).hexdigest()
    assert (
        f"contract `rq1-baselines-v2`, SHA-256 `{expected_contract_hash}`" in contract
    )
    assert "tolerance observation" in preamble
    assert "provenance-incomplete" in preamble
    assert "v1.5 SAGA diagnostic is `stopped_platform_warning`" in preamble
    assert "v1.6 SAGA diagnostic is `passed_training_only`" in preamble
    assert "Protocol v1.7 froze `rq1-baselines-v2` before validation" in preamble
    assert (
        "rq1-baselines-v2 execution is `completed_development_validation`" in preamble
    )
    assert "development validation only" in preamble
    assert "Protocol v1.8 froze `rq1-transformer-cascade-v1`" in preamble
    assert "`superseded_unrun`" in preamble
    assert "Protocol v1.9 freezes `rq1-transformer-cascade-v2`" in preamble
    assert "contract status `frozen_not_run`" in preamble
    assert (
        "pre-execution implementation status was `frozen_implemented_not_run`"
        in preamble
    )
    assert "`running_development_validation`" in preamble
    assert "no completed transformer, threshold, or cascade result" in preamble
    assert "H1, H2, and H3 remain undecided" in preamble
    assert "group test remains analyst-exposed but model-unscored" in preamble
    assert "No PhishVN record has been accessed" in preamble


def test_protocol_v19_supersedes_unrun_v1_with_the_publication_only_v2_amendment():
    assert sha256(TRANSFORMER_CONTRACT.read_bytes()).hexdigest() == (
        TRANSFORMER_CONTRACT_SHA256
    )
    assert sha256(TRANSFORMER_CONTRACT_V2.read_bytes()).hexdigest() == (
        TRANSFORMER_CONTRACT_V2_SHA256
    )

    protocol = _compact(PROTOCOL.read_text(encoding="utf-8"))
    status = _compact(STATUS.read_text(encoding="utf-8"))
    evidence = _compact(EVIDENCE_OUTLINE.read_text(encoding="utf-8"))
    basis = _compact(RESEARCH_BASIS.read_text(encoding="utf-8"))
    readme = _compact(README.read_text(encoding="utf-8"))

    assert "**version:** 1.10 | **date:** 2026-09-17" in protocol
    assert "governed by protocol v1.10" in readme
    assert "rq1-transformer-cascade-v2" in protocol
    assert TRANSFORMER_CONTRACT_V2_SHA256 in protocol
    assert "rq1-transformer-cascade-v1" in protocol
    assert TRANSFORMER_CONTRACT_SHA256 in protocol
    assert "superseded_unrun" in protocol
    assert "no completed transformer" in protocol

    assert "rq1-transformer-cascade-v2" in status
    assert TRANSFORMER_CONTRACT_V2_SHA256 in status
    assert "superseded_unrun" in status
    assert "`frozen_not_run`" in status
    assert "`frozen_implemented_not_run`" in status

    assert "rq1-transformer-cascade-v2" in readme
    assert "frozen_implemented_not_run" in readme
    assert "no completed transformer" in readme
    assert "rq1-transformer-cascade-v2" in evidence
    assert "no completed transformer" in evidence
    assert "rq1-transformer-cascade-v2" in basis
    assert "current execution record" in basis

    assert "public summary is the completion marker" in protocol
    assert "completed result requires both" in protocol
    assert "incomplete_not_result" in protocol
    assert "not cross-destination atomic" in protocol
    assert "verify" in protocol and "remove" in protocol and "before rerun" in protocol
    assert "identity, schema and protocol version, date" not in protocol
    assert "identity, version, date, and publication" not in " ".join(
        (status, evidence, basis, readme)
    )
    for record in (protocol, status, readme):
        assert (
            "contract identity, schema version, protocol version, and publication "
            "semantics" in record
        )

    combined = " ".join((status, evidence, readme))
    assert "reviewed repository commit" in combined
    assert "final reviewed code commit" not in combined
    assert "final executable code commit" not in combined


def test_live_records_capture_v2_validation_without_deciding_hypotheses():
    records = {
        "README": README.read_text(encoding="utf-8"),
        "status": STATUS.read_text(encoding="utf-8"),
        "evidence": EVIDENCE_OUTLINE.read_text(encoding="utf-8"),
    }
    for name, record in records.items():
        compact = _compact(record)
        for statement in (
            "rq1-baselines-v2",
            "completed_development_validation",
            "development validation only",
            "h1, h2, and h3 remain undecided",
            "bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c",
            "7ae6c9af85e935c551468f590a7ba43441f58def",
            "b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799",
            "71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a",
        ):
            assert statement in compact, f"missing from {name}: {statement}"
        assert "rq1-baselines-v2 execution has status `frozen_not_run`" not in compact

        for copied_value in (
            "0.7612031186147",
            "0.3214800576645843",
            "0.006680191993666189",
            "0.007702192373035135",
            "0.2670846328466124",
            "0.9842223290084895",
            "0.008758473947251225",
            "0.009915480854183582",
            "2.1316282072803006e-14",
            "3.3306690738754696e-16",
        ):
            assert copied_value in compact, f"missing from {name}: {copied_value}"

    for name in ("README", "evidence"):
        assert "no phishvn record has been accessed" in _compact(records[name])
    assert (
        "the implementation and its tests did not open the phiusiil group-test "
        "partition or phishvn" in _compact(records["status"])
    )

    for record in records.values():
        compact = _compact(record)
        assert "group test remains analyst-exposed but model-unscored" in compact
        assert "process input boundary" in compact


def test_live_records_preserve_v14_failure_and_qualify_tolerance_observation():
    audit_sections = (
        (
            STATUS,
            "Execution Audit",
            2,
            "| Historical RQ1 baseline contract | `rq1-baselines-v1` |",
        ),
        (
            EVIDENCE_OUTLINE,
            "RQ1 Baseline Execution Note",
            3,
            "| Historical RQ1 baseline contract | `data/rq1-baseline-contract.json` "
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
        assert "| Protocol version | `1.10` |" in text
        assert contract_row in text
        assert (
            "| Historical RQ1 baseline contract SHA-256 | "
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


def test_v16_saga_diagnostic_record_is_training_only_and_reproducible():
    protocol = PROTOCOL.read_text(encoding="utf-8")
    diagnostic = _compact(_section(protocol, "SAGA convergence diagnostic", level=3))
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
        "scaling and fitting warnings remain fatal",
        "only while calling `decision_function` and `predict_proba`",
        '`sys.platform="darwin"`',
        '`platform.machine()="arm64"`',
        "numpy blas name is exactly `accelerate`",
        "module `sklearn.utils.extmath`",
        "`divide by zero encountered in matmul`",
        "`overflow encountered in matmul`",
        "`invalid value encountered in matmul`",
        "captured and recorded rather than silently discarded",
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
        "`rq1-saga-convergence-v2`",
        "`reports/rq1-saga-convergence-v1-execution.json`",
        "`7309f52f704150f85e6c17d44d96adcde917d2539c7b75264bf775ccec3aa6f4`",
        "both v1.5 fresh runs stopped",
        "`runtimewarning: divide by zero encountered in matmul`",
        "`logistic-l1` aggregate was not produced",
        'float64 `np.einsum("ij,j->i", ..., optimize=false)`',
        "`scipy.special.expit`",
        "`rtol=1e-12` and `atol=1e-12`",
        "maximum absolute decision-score and probability differences",
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
        assert "v1.5 saga diagnostic has status `stopped_platform_warning`" in text
        assert "v1.6 saga diagnostic has status `passed_training_only`" in text
        assert (
            "rq1-baselines-v2 execution has status "
            "`completed_development_validation`" in text
        )
        assert "two-model saga diagnostic" in text
        assert "both v1.6 fresh runs" in text
        assert "fitted-state sha-256" in text
        assert "h1, h2, and h3 remain undecided" in text

    readme = README.read_text(encoding="utf-8")
    assert "uv run --locked python scripts/rq1_saga_convergence_diagnostic.py" in readme


def test_baseline_v2_contract_preserves_design_and_freezes_executed_method():
    legacy_bytes = BASELINE_V1_CONTRACT.read_bytes()
    assert sha256(legacy_bytes).hexdigest() == (
        "594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4"
    )
    legacy = json.loads(legacy_bytes)
    current = json.loads(BASELINE_V2_CONTRACT.read_bytes())

    assert current["contract_id"] == "rq1-baselines-v2"
    assert current["schema_version"] == 2
    assert current["protocol_version"] == "1.7"
    for field in (
        "input",
        "output",
        "predictor_policy",
        "features",
        "partition_use",
        "score",
        "threshold_selection",
    ):
        assert current[field] == legacy[field]
    assert current["models"]["length-only"] == legacy["models"]["length-only"]
    assert current["models"]["Logistic-L1"] == legacy["models"]["Logistic-L1"]
    assert current["models"]["search"] == "none"

    classifier = current["models"]["common_pipeline"]["classifier"]
    assert classifier == {
        "class": "LogisticRegression",
        "penalty": "l1",
        "solver": "saga",
        "C": 1.0,
        "class_weight": "balanced",
        "fit_intercept": True,
        "max_iter": 5000,
        "tol": 1e-4,
        "random_state": 42,
    }
    assert "intercept_scaling" not in classifier
    integrity = current["scoring_integrity"]
    compact_integrity = _compact(json.dumps(integrity, sort_keys=True))
    for rule in (
        "decision_function",
        "predict_proba",
        "darwin",
        "arm64",
        "accelerate",
        "runtimewarning",
        "sklearn.utils.extmath",
        "divide by zero encountered in matmul",
        "overflow encountered in matmul",
        "invalid value encountered in matmul",
        "einsum",
        "expit",
        "1e-12",
    ):
        assert rule in compact_integrity

    contract_hash = sha256(BASELINE_V2_CONTRACT.read_bytes()).hexdigest()
    for record in (PROTOCOL, STATUS, EVIDENCE_OUTLINE):
        text = record.read_text(encoding="utf-8")
        assert "rq1-baselines-v2" in text
        assert contract_hash in text
    assert "data/rq1-baseline-contract-v2.json" in README.read_text(encoding="utf-8")


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
    v18_identity = decision_matrix.replace(
        "rq1-transformer-cascade-v2", "rq1-transformer-cascade-v1"
    )

    assert sha256(v18_identity.encode("utf-8")).hexdigest() == (DECISION_MATRIX_SHA256)

    critical_contracts = (
        "**RQ1:** What incremental value do structural URL features and selective character-model escalation provide under registrable-domain-disjoint and external evaluation?",
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

    rq1_row = next(
        line
        for line in decision_matrix.splitlines()
        if line.startswith("| **RQ1 / H1**")
    )
    rq1_decision_rule = rq1_row.rsplit("|", maxsplit=2)[-2]
    assert rq1_decision_rule.count("`recall(Logistic-L1) - recall(length-only)`") == 1
    assert rq1_decision_rule.count("`recall(cascade) - recall(Logistic-L1)`") == 1
    assert "each of length-only, `Logistic-L1`, and cascade" in rq1_decision_rule
    assert "transformer-only" not in rq1_decision_rule
    assert "recall(transformer-only) -" not in rq1_decision_rule
    assert "system contribution" in rq1_row
    assert "not a pure causal isolation of representation" in rq1_row


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


def test_v19_binds_transformer_contract_and_preserves_prospective_status():
    assert TRANSFORMER_CONTRACT_V2.is_file(), (
        f"missing transformer contract: {TRANSFORMER_CONTRACT_V2}"
    )
    expected_hash = sha256(TRANSFORMER_CONTRACT_V2.read_bytes()).hexdigest()
    bindings = (
        (
            PROTOCOL,
            "RQ1 transformer and cascade contract",
            3,
            True,
        ),
        (STATUS, "Current Controls", 2, True),
        (EVIDENCE_OUTLINE, "RQ1 and H1", 2, False),
    )

    for path, heading, level, binds_hash_locally in bindings:
        section = _compact(
            _section(path.read_text(encoding="utf-8"), heading, level=level)
        )
        assert "rq1-transformer-cascade-v2" in section
        if binds_hash_locally:
            assert expected_hash in section
        assert "`frozen_not_run`" in section
        assert any(
            boundary in section
            for boundary in (
                "has no completed result",
                "no completed transformer, threshold, or cascade result exists",
            )
        )

    for path in (STATUS, EVIDENCE_OUTLINE):
        text = path.read_text(encoding="utf-8")
        assert (
            "| RQ1 transformer/cascade contract | "
            "`data/rq1-transformer-cascade-contract-v2.json` "
            "(`rq1-transformer-cascade-v2`) |" in text
        )
        assert (
            f"| RQ1 transformer/cascade contract SHA-256 | `{expected_hash}` |" in text
        )


def test_live_records_bind_reviewed_transformer_code_without_claiming_completion():
    readme = _compact(README.read_text(encoding="utf-8"))
    status = _compact(STATUS.read_text(encoding="utf-8"))
    evidence = _compact(EVIDENCE_OUTLINE.read_text(encoding="utf-8"))
    basis = _compact(RESEARCH_BASIS.read_text(encoding="utf-8"))
    question = (
        "What incremental value do structural URL features and selective "
        "character-model escalation provide under registrable-domain-disjoint "
        "and external evaluation?"
    )

    for required in (
        "`frozen_implemented_not_run`",
        FINAL_TRANSFORMER_CODE_COMMIT,
        INITIAL_TRANSFORMER_CODE_COMMIT,
        TRANSFORMER_CONTRACT_V2_SHA256,
        V19_MATRIX_SHA256,
        SEPTEMBER_REPORT_SHA256,
        "procedure code, tests, cli, and private/public artifact publication code "
        "are complete",
        "no completed transformer, threshold, or cascade result exists",
        "h1, h2, and h3 remain undecided",
        "group test remains analyst-exposed but model-unscored",
        "the implementation and its tests did not open the phiusiil group-test "
        "partition or phishvn",
        "they are not performance or result runs",
        "each destination uses a temporary path in its own parent",
        "public summary is installed last and is the completion marker",
        "caught in-process `baseexception` removes only destinations created by "
        "the run",
        "not cross-destination atomic",
        "abrupt process or host failure can leave the private directory without "
        "the public summary",
        "either one-sided state is `incomplete_not_result`",
    ):
        assert required in status, f"missing from approval status: {required}"

    for required in (
        "`frozen_implemented_not_run`",
        FINAL_TRANSFORMER_CODE_COMMIT,
        "rq1-transformer-cascade-v2",
        TRANSFORMER_CONTRACT_V2_SHA256,
        "procedure code, tests, cli, and private/public artifact publication code "
        "are complete",
        "no completed transformer, threshold, or cascade result exists",
        "h1, h2, and h3 remain undecided",
        "group test remains analyst-exposed but model-unscored",
        "no phishvn record has been accessed",
    ):
        assert required in readme, f"missing from README: {required}"

    for required in (
        "`frozen_implemented_not_run`",
        FINAL_TRANSFORMER_CODE_COMMIT,
        TRANSFORMER_CONTRACT_V2_SHA256,
        "transformer/cascade procedure can produce evidence when run; it is not "
        "itself a model result",
        "no completed transformer, threshold, or cascade result exists",
        "h1, h2, and h3 remain undecided",
        "group test remains analyst-exposed but model-unscored",
        "no phishvn record has been accessed",
        "the implementation and its tests did not open the phiusiil group-test "
        "partition or phishvn",
    ):
        assert required in evidence, f"missing from evidence outline: {required}"

    for required in (
        "`frozen_implemented_not_run`",
        _compact(question),
        "`recall(logistic-l1) - recall(length-only)`",
        "`recall(cascade) - recall(logistic-l1)`",
        "transformer-only remains a comparator and operational reference",
        "not a third primary h1 gate",
        "system contribution, not a pure causal isolation",
        "current execution record",
    ):
        assert required in basis, f"missing from research basis: {required}"


def test_live_records_keep_manual_review_post_hoc_and_non_interventional():
    records = (STATUS, EVIDENCE_OUTLINE, RESEARCH_BASIS)
    boundary = (
        "manual review is permitted only as separately reported post hoc "
        "descriptive error analysis and cannot assign or override labels, change "
        "quarantine or inclusion, thresholds, features, model or procedure choices, "
        "gates, or hypothesis decisions."
    )

    for path in records:
        assert boundary in _compact(path.read_text(encoding="utf-8"))


def test_research_basis_uses_expertfusion_2027_issue_year():
    basis = RESEARCH_BASIS.read_text(encoding="utf-8")

    assert basis.count("[ExpertFusion (2027)]") == 1
    assert "[ExpertFusion (2026)]" not in basis


def test_v18_records_september_advisor_direction_and_completed_source_freeze():
    report_hash = "b72da89a4cc8a5b06f6ca88d79fe78dd54e3199a96b7450209ea53b4a4c04215"
    records = (PROTOCOL, STATUS, EVIDENCE_OUTLINE, RESEARCH_BASIS)

    for path in records:
        text = _compact(path.read_text(encoding="utf-8"))
        assert "september 3 advisor report" in text
        assert report_hash in text
        assert "complete and freeze the source-provenance release" in text
        assert "systematic hypothesis testing" in text
        assert (
            "all gates, thresholds, features, and train/validation procedures" in text
        )
        assert "locked before test results" in text
        assert "particular attention to h1 and the gmm" in text
        assert (
            "the public `phiusiil-development-v1` release completed that requested "
            "source freeze after the meeting" in text
        )


def test_v18_freezes_manual_review_boundary_in_all_method_records():
    boundary = (
        "manual review is permitted only as separately reported post hoc "
        "descriptive error analysis and cannot assign or override labels, change "
        "quarantine or inclusion, thresholds, features, model or procedure choices, "
        "gates, or hypothesis decisions."
    )

    for path in (PROTOCOL, STATUS, EVIDENCE_OUTLINE, RESEARCH_BASIS):
        assert boundary in _compact(path.read_text(encoding="utf-8"))


def test_v18_resolves_h2_complete_window_metric_before_gmm_execution():
    required_rules = (
        "every complete 256-request window of the retained external stream is a "
        "prespecified external-shift window",
        "numerator is windows with score strictly greater than the boundary",
        "denominator is all such complete windows",
        "overlapping windows count separately",
        "incomplete terminal window is excluded from this rate",
        "its requests remain routable from a prior alert",
        "independent validation-audit false-alert fraction uses the same "
        "complete-window numerator and denominator rule",
    )

    for path in (PROTOCOL, EVIDENCE_OUTLINE, RESEARCH_BASIS):
        text = _compact(path.read_text(encoding="utf-8"))
        for rule in required_rules:
            assert rule in text, f"missing from {path.name}: {rule}"

    combined = _compact(
        "\n".join(path.read_text(encoding="utf-8") for path in (PROTOCOL, STATUS))
    )
    assert "gmm execution is `not_run`" in combined


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


def test_v16_change_record_is_a_diagnostic_only_platform_amendment():
    status = STATUS.read_text(encoding="utf-8")
    change_record = _section(status, "Change Record")
    v16_rows = [line for line in change_record.splitlines() if "| 1.6 |" in line]

    assert len(v16_rows) == 1
    v16_row = v16_rows[0].lower()
    for detail in (
        "v1.5 diagnostic failure",
        "macos arm64",
        "accelerate",
        "scoring-only warning audit",
        "independent numerical reference",
        "no baseline result",
        "no rq/h decision rule changed",
    ):
        assert detail in v16_row


def test_v17_change_record_freezes_baseline_v2_without_claiming_a_result():
    status = STATUS.read_text(encoding="utf-8")
    change_record = _section(status, "Change Record")
    v17_rows = [line for line in change_record.splitlines() if "| 1.7 |" in line]

    assert len(v17_rows) == 1
    v17_row = v17_rows[0].lower()
    for detail in (
        "v1.6 diagnostic passed",
        "training feasibility only",
        "rq1-baselines-v2",
        "saga",
        "validation scoring audit",
        "frozen before validation",
        "no baseline result",
        "no rq/h decision rule changed",
    ):
        assert detail in v17_row


def test_v18_change_record_freezes_transformer_cascade_without_a_result():
    status = STATUS.read_text(encoding="utf-8")
    change_record = _section(status, "Change Record")
    v18_rows = [line for line in change_record.splitlines() if "| 1.8 |" in line]

    assert len(v18_rows) == 1
    v18_row = v18_rows[0].lower()
    for detail in (
        "september 3 advisor direction",
        "source-provenance release",
        "rq1-transformer-cascade-v1",
        "selective character-model escalation",
        "manual-review boundary",
        "complete-window h2 metric",
        "frozen_not_run",
        "no transformer, cascade, or gmm fit",
    ):
        assert detail in v18_row


def test_v19_change_record_is_publication_only_and_preserves_v1():
    status = STATUS.read_text(encoding="utf-8")
    change_record = _section(status, "Change Record")
    v19_rows = [line for line in change_record.splitlines() if "| 1.9 |" in line]

    assert len(v19_rows) == 1
    v19_row = v19_rows[0].lower()
    for detail in (
        "preserved `rq1-transformer-cascade-v1` byte-for-byte",
        "`superseded_unrun`",
        "`rq1-transformer-cascade-v2`",
        "`frozen_not_run`",
        "publication semantics",
        "does not change a research question, hypothesis, input, model, training, "
        "threshold, cascade, manual-review, or artifact-content rule",
        "`frozen_implemented_not_run`",
        "no transformer, cascade, or gmm fit",
        "corrected the still-unrun gmm allocation status to a prospective staged "
        "freeze without selecting an allocation rule",
    ):
        assert detail in v19_row


def test_v110_records_bind_exact_gmm_contract_before_execution():
    contract = json.loads(GMM_CONTRACT.read_text(encoding="utf-8"))
    expected_hash = sha256(GMM_CONTRACT.read_bytes()).hexdigest()
    for path in (PROTOCOL, STATUS, EVIDENCE_OUTLINE, RESEARCH_BASIS, README):
        text = path.read_text(encoding="utf-8")
        assert contract["contract_id"] in text
        assert expected_hash in text
    protocol = PROTOCOL.read_text(encoding="utf-8")
    assert "before any gmm execution, a separate contract will freeze" not in _compact(
        protocol
    )
    section = _section(protocol, "RQ2 GMM development contract", level=3)
    assert contract["validation_allocation"]["namespace"] in section
    assert str(contract["validation_allocation"]["seed"]) in section
    assert contract["audit"]["gate"] in section
    assert contract["execution"]["completed_status"] in section
    for path in (STATUS, EVIDENCE_OUTLINE):
        text = path.read_text(encoding="utf-8")
        assert f"| RQ2 GMM contract SHA-256 | `{expected_hash}` |" in text


def test_gmm_development_summary_binds_frozen_method_and_failed_audit():
    assert sha256(GMM_SUMMARY.read_bytes()).hexdigest() == (
        "6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523"
    )
    contract_hash = "22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393"
    assert sha256(GMM_CONTRACT.read_bytes()).hexdigest() == contract_hash
    contract = json.loads(GMM_CONTRACT.read_bytes())
    summary = json.loads(GMM_SUMMARY.read_bytes())
    assert summary["contract"] == {
        "id": "rq2-gmm-development-v1",
        "protocol_version": "1.10",
        "sha256": contract_hash,
    }
    assert summary["configuration"] == {
        key: contract[key]
        for key in (
            "features",
            "scaler",
            "validation_allocation",
            "mixture",
            "runtime",
            "windows",
            "calibration",
            "audit",
        )
    }
    expected_inputs = dict(contract["inputs"]["accepted_roles"])
    expected_inputs["baseline_contract"] = expected_inputs.pop("contract")
    assert summary["input_hashes"] == {**expected_inputs, "gmm_contract": contract_hash}
    assert summary["schema_version"] == 1
    assert summary["status"] == "completed_development_validation"
    assert summary["analysis_stage"] == "development_validation_only"
    assert summary["hypothesis_status"] == {
        "H1": "undecided",
        "H2": "undecided",
        "H3": "undecided",
    }
    assert summary["access"] == {
        "external_accessed": False,
        "group_test_accessed": False,
        "phishvn_accessed": False,
        "scope": "this_process_only",
    }
    candidates = summary["candidates"]
    assert [candidate["components"] for candidate in candidates] == list(range(1, 7))
    assert all(
        candidate["converged"] is True
        and isfinite(candidate["bic"])
        and 1 <= candidate["n_iter"] <= 500
        for candidate in candidates
    )
    selected = min(candidates, key=lambda item: (item["bic"], item["components"]))
    assert summary["selected_component_count"] == selected["components"] == 6
    assert summary["calibration_window_count"] == summary["audit_window_count"] == 252
    for stream in ("calibration", "audit"):
        assert summary["input_counts"][stream]["complete_windows"] == 252
        assert summary["input_counts"][stream]["domain_count"] == 14783
    assert summary["audit_alert_count"] == 28
    assert summary["audit_alert_fraction"] == 28 / 252 == 1 / 9
    assert 20 * summary["audit_alert_count"] > summary["audit_window_count"]
    assert summary["false_alert_gate_met"] is False
    assert summary["threshold"] == -67.45792380813624
    assert summary["warnings"] == []


def test_gmm_development_summary_contains_only_public_aggregates():
    summary = json.loads(GMM_SUMMARY.read_bytes())
    contract = json.loads(GMM_CONTRACT.read_bytes())
    wheel_url = contract["runtime"]["macos_arm64_numpy_wheel"]["url"]
    forbidden_keys = {
        "raw_url",
        "raw_urls",
        "canonical_url",
        "canonical_url_sha256",
        "registrable_domain",
        "domains",
        "record_id",
        "record_ids",
        "records",
        "feature_row",
        "feature_rows",
        "feature_vector",
        "feature_vectors",
        "weights",
        "mean",
        "means",
        "variance",
        "variances",
        "scale",
        "coefficients",
        "intercept",
        "predictions",
        "probabilities",
        "window_scores",
        "window_end_positions",
        "input_row_positions",
    }
    urls = []

    def visit(value):
        if isinstance(value, dict):
            assert forbidden_keys.isdisjoint(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            assert not any(
                marker in value
                for marker in (
                    "phiusiil-row-v1:",
                    "/Users/",
                    ".example",
                )
            )
            if "http://" in value or "https://" in value:
                urls.append(value)

    visit(summary)
    assert urls == [wheel_url]


def test_transformer_stopped_run_is_preserved_without_claiming_a_result():
    execution = json.loads(TRANSFORMER_EXECUTION.read_bytes())
    assert execution["status"] == "stopped_stage_one_integrity_check"
    assert execution["execution_commit"] == ("c3a5c815b20121f1ddd06a2f316f904077c00c4f")
    assert execution["contract_sha256"] == TRANSFORMER_CONTRACT_V2_SHA256
    assert execution["started_utc"] == "2026-09-17T18:21:12Z"
    assert execution["finished_utc"] == "2026-09-17T19:24:27Z"
    assert execution["exit_code"] == 2
    assert execution["elapsed_seconds"] == 3795.28
    assert execution["error"] == (
        "stage-one artifact threshold does not match the supplied scores"
    )
    assert execution["output_directory_present"] is False
    assert execution["public_summary_present"] is False
    assert execution["result_accepted"] is False
    assert execution["hypothesis_status"] == dict.fromkeys(
        ("H1", "H2", "H3"), "undecided"
    )
    status = STATUS.read_text(encoding="utf-8")
    assert "stopped_stage_one_integrity_check" in status
    assert "rq1-transformer-cascade-v2-execution.json" in status


def test_second_group_test_display_is_recorded_without_changing_study_status():
    status_audit = _compact(
        _section(STATUS.read_text(encoding="utf-8"), "Execution Audit")
    )
    evidence_note = _compact(
        _section(
            EVIDENCE_OUTLINE.read_text(encoding="utf-8"),
            "Internal Holdout Access Note",
            level=3,
        )
    )

    for record in (status_audit, evidence_note):
        assert "on 2026-09-09" in record
        assert "second broad local wording search displayed row content" in record
        assert (
            "displayed rows informed no model, threshold, gate, routing, or "
            "scientific-procedure change" in record
        )
        assert (
            "separate v2 transformer publication correction arose from code review and "
            "changed no scientific field" in record
        )
        assert "no fit, score, metric, or phishvn access" in record
        assert "analyst-exposed but model-unscored" in record


def test_readme_links_research_basis_and_limits_synthetic_urls_to_unit_tests():
    readme = README.read_text(encoding="utf-8")
    readme_prose = " ".join(readme.split())

    assert "[research basis](docs/research-basis.md)" in readme
    assert "Invented or synthetic URLs are unit-test fixtures only" in readme_prose
    assert "never research observations" in readme_prose


def test_public_research_records_exclude_stale_or_approval_gating_language():
    records = (PROTOCOL, STATUS, EVIDENCE_OUTLINE, RESEARCH_BASIS, README)
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
        "approval required",
        "work is stuck",
        "blocked on approval",
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
