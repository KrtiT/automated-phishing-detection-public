from __future__ import annotations

import hashlib
import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from lxml import etree

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "attachments" / "foUeB5" / "Tallam_Krti_Praxis_2026-06-15.docx"
PREVIOUS_BODY_SOURCE = HERE / "Tallam_Krti_Praxis_v3_Body_2026-09-04.md"
BODY_SOURCE = HERE / "Tallam_Krti_Praxis_v3_Body_2026-09-09.md"
BUILDER = HERE / "build_v3_working_manuscript.py"
WORKSPACE_OUTPUT = HERE / "Tallam_Krti_Praxis_v3_Working_2026-09-09.docx"
CURRENT_BODY_SOURCE = HERE / "Tallam_Krti_Praxis_v3_Body_2026-09-17.md"
CURRENT_OUTPUT = HERE / "Tallam_Krti_Praxis_v3_Working_2026-09-17.docx"

SOURCE_SHA256 = "96c056e6becf2bab7335adf4ed850707ca049749233b819fde7fb80a060daf3d"
FRONT_MATTER_SHA256 = "38cae1fa8dbbff6b091409f0c07082f37a666e2596903f85d84ab14f3cbf7147"
PREVIOUS_BODY_SHA256 = (
    "da1aea09a15c0b738e9e814fdf3721a687976e83199415a7bead456dafabf133"
)
FINAL_TRANSFORMER_CODE_COMMIT = "0793ca3dbc36e49b561cd0ac74968a4644060426"
INITIAL_TRANSFORMER_CODE_COMMIT = "a8ee067bda8fd45d19f5c4b794ba21f58d1947fc"
PUBLICATION_AMENDMENT_COMMIT = "7205e5630c5d7805d9bf6d293ead14b89cfe10a5"
PROTOCOL_RECORD_COMMIT = "0d45625527040632d444e552f9a25ccf39a3fabf"
DURABILITY_HARDENING_COMMIT = "c3a5c815b20121f1ddd06a2f316f904077c00c4f"
SUPERSEDED_TRANSFORMER_CONTRACT_SHA256 = (
    "aeaa84534c4cadf0459cf6d2f010dc802684d4801cce563ce18242f36359fb54"
)
TRANSFORMER_CONTRACT_SHA256 = (
    "686c0d86b33b8a6c2e09cd6e174003db0bd2f7c30b087faf5470e6a270524213"
)
V19_MATRIX_SHA256 = "f24eac919cb79d24d2248a94b3a74208f7b4d809ad778b963ad2e62315d78a38"
SEPTEMBER_REPORT_SHA256 = (
    "b72da89a4cc8a5b06f6ca88d79fe78dd54e3199a96b7450209ea53b4a4c04215"
)
DOCUMENT_XML = "word/document.xml"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"

EXACT_BASELINE_METRICS = {
    "Length-only": {
        "threshold": Decimal("0.7612031186147"),
        "recall": Decimal("0.3214800576645843"),
        "observed_fpr": Decimal("0.006680191993666189"),
        "fpr_upper_95": Decimal("0.007702192373035135"),
    },
    "Logistic-L1": {
        "threshold": Decimal("0.2670846328466124"),
        "recall": Decimal("0.9842223290084895"),
        "observed_fpr": Decimal("0.008758473947251225"),
        "fpr_upper_95": Decimal("0.009915480854183582"),
    },
}

ARTIFACT_PARAGRAPH = (
    "The aggregate baseline summary has SHA-256 "
    "bf5b3a6f0fc705d26852da4dd0053c6111ffc3e500d7a2e95dfba5ad859b279c. "
    "That hash-pinned JSON retains the full-precision thresholds, rates, counts, "
    "and scoring-audit values. The private length-only artifact has SHA-256 "
    "b8b92cfbe29160e769e5e7d80712becc8fc0680cdfd45a44b839ef9bada87799; "
    "it used one feature, reported n_iter=69, and evaluated 221 threshold "
    "candidates. The private Logistic-L1 artifact has SHA-256 "
    "71a3e24a0283a31ba188bc7dd60b18c1b708370b9ca275d5ab1a1004680c968a; "
    "it used 25 features, reported n_iter=4783, and evaluated 11,279 threshold "
    "candidates. Both fits stopped below the frozen 5,000-iteration limit, and "
    "both threshold records have status selected."
)
CONFUSION_PARAGRAPH = (
    "The length-only validation counts were 20,074 true negatives, 135 false "
    "positives, 4,014 true positives, and 8,472 false negatives. The Logistic-L1 "
    "counts were 20,032 true negatives, 177 false positives, 12,289 true "
    "positives, and 197 false negatives. Both models used the same 20,209 "
    "negative and 12,486 positive validation records."
)
AUDIT_PARAGRAPH = (
    "The length-only scoring audit recorded no warning and maximum decision and "
    "probability differences of zero. The Logistic-L1 audit recorded six "
    "allowlisted scoring warnings. Its finite values agreed with the independent "
    "reference: the maximum absolute decision difference was "
    "2.1316282072803006e-14, and the maximum absolute full-matrix probability "
    "difference was 3.3306690738754696e-16."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_builder():
    spec = importlib.util.spec_from_file_location("v3_manuscript_builder", BUILDER)
    if spec is None or spec.loader is None:
        raise AssertionError("unable to import manuscript builder")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    return builder


def document_root(package: ZipFile) -> etree._Element:
    return etree.fromstring(package.read(DOCUMENT_XML))


def body_children(package: ZipFile) -> list[etree._Element]:
    root = document_root(package)
    body = root.find(f".//{W}body")
    if body is None:
        raise AssertionError("word/document.xml has no body")
    return list(body)


def element_text(element: etree._Element) -> str:
    return "".join(element.itertext())


def canonical_hash(elements: list[etree._Element]) -> str:
    data = b"".join(
        etree.tostring(element, method="c14n", exclusive=False, with_comments=False)
        for element in elements
    )
    return sha256(data)


def body_text(package: ZipFile, start: int = 214) -> str:
    return "\n".join(
        element_text(element) for element in body_children(package)[start:]
    )


def direct_body_paragraphs(package: ZipFile, start: int = 214) -> list[str]:
    return [
        element_text(element).strip()
        for element in body_children(package)[start:]
        if element.tag == f"{W}p" and element_text(element).strip()
    ]


def document_tables(package: ZipFile) -> list[list[list[str]]]:
    root = document_root(package)
    tables: list[list[list[str]]] = []
    for table in root.findall(f".//{W}body/{W}tbl"):
        rows: list[list[str]] = []
        for row in table.findall(f"./{W}tr"):
            rows.append(
                [element_text(cell).strip() for cell in row.findall(f"./{W}tc")]
            )
        tables.append(rows)
    return tables


def displayed_metric_row(model: str) -> list[str]:
    values = EXACT_BASELINE_METRICS[model]
    return [
        model,
        f"{values['threshold']:.6f}",
        f"{values['recall'] * 100:.4f}%",
        f"{values['observed_fpr'] * 100:.4f}%",
        f"{values['fpr_upper_95'] * 100:.4f}%",
    ]


def markdown_section(text: str, heading: str, next_heading: str) -> str:
    return text.split(heading, maxsplit=1)[1].split(next_heading, maxsplit=1)[0]


class SourceContractTests(unittest.TestCase):
    def test_source_shell_has_fixed_digest_and_unique_chapter_boundary(self) -> None:
        self.assertEqual(sha256(SOURCE.read_bytes()), SOURCE_SHA256)
        with ZipFile(SOURCE) as package:
            children = body_children(package)
        chapter_one = [
            index
            for index, element in enumerate(children)
            if element_text(element).strip() == "Chapter 1—Introduction"
        ]
        self.assertEqual(chapter_one, [214])
        self.assertEqual(canonical_hash(children[:214]), FRONT_MATTER_SHA256)

    def test_september_4_body_source_remains_unchanged(self) -> None:
        self.assertEqual(
            sha256(PREVIOUS_BODY_SOURCE.read_bytes()), PREVIOUS_BODY_SHA256
        )

    def test_september_9_body_source_records_the_implemented_method_only(self) -> None:
        self.assertTrue(BODY_SOURCE.is_file())
        body = BODY_SOURCE.read_text(encoding="utf-8")
        required = [
            "September 9, 2026",
            "frozen_implemented_not_run",
            FINAL_TRANSFORMER_CODE_COMMIT,
            INITIAL_TRANSFORMER_CODE_COMMIT,
            PUBLICATION_AMENDMENT_COMMIT,
            PROTOCOL_RECORD_COMMIT,
            DURABILITY_HARDENING_COMMIT,
            SUPERSEDED_TRANSFORMER_CONTRACT_SHA256,
            TRANSFORMER_CONTRACT_SHA256,
            V19_MATRIX_SHA256,
            SEPTEMBER_REPORT_SHA256,
            "protocol v1.9",
            "rq1-transformer-cascade-v2",
            "rq1-transformer-cascade-v1",
            "superseded_unrun",
            (
                "Procedure code, tests, CLI, and private/public artifact publication code "
                "are complete"
            ),
            "No transformer fit, threshold calibration, or cascade result exists",
            (
                "The next milestone is one clean, noninteractive fit using only the pinned "
                "training and validation inputs"
            ),
        ]
        for phrase in required:
            self.assertIn(phrase, body, phrase)

    def test_september_9_body_records_staged_freeze_and_repeat_exposure(self) -> None:
        body = BODY_SOURCE.read_text(encoding="utf-8")
        research_design = markdown_section(body, "## 3.1", "## 3.2")
        allocation = markdown_section(body, "## 3.5", "## 3.6")
        rq_status = markdown_section(body, "## 4.5", "## 4.6")
        limitations = markdown_section(body, "## 5.5", "## 5.6")

        self.assertIn("prospective staged-freeze design", research_design)
        self.assertNotIn("prospective, frozen experimental design", research_design)
        self.assertIn("separate GMM contract will freeze", research_design)
        self.assertIn("no GMM allocation choice has been made", research_design)

        self.assertIn("On September 9, 2026", allocation)
        self.assertIn("second broad local wording search", allocation)
        self.assertIn(
            "displayed rows informed no model, threshold, gate, routing, or "
            "scientific-procedure change",
            allocation,
        )
        self.assertIn(
            "v2 transformer publication correction arose from code review",
            allocation,
        )
        self.assertIn("September 9 repeat display", rq_status)
        self.assertIn("Two documented broad-search displays", limitations)
        for section in (allocation, rq_status, limitations):
            self.assertIn("analyst-exposed but model-unscored", section)
        self.assertIn("No fit, score, metric, or PhishVN access occurred", allocation)
        self.assertIn("No PhishVN record has been accessed", rq_status)
        self.assertIn("no phishvn record has been accessed", limitations.lower())


class ManuscriptBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.output = (
            Path(cls.temporary_directory.name)
            / "Tallam_Krti_Praxis_v3_Working_2026-09-09.docx"
        )
        cls.builder = load_builder()
        cls.builder.build(SOURCE, BODY_SOURCE, cls.output)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_directory.cleanup()

    def assert_single_table(self, expected: list[list[str]]) -> None:
        with ZipFile(self.output) as package:
            matches = [
                table for table in document_tables(package) if table[0] == expected[0]
            ]
        self.assertEqual(matches, [expected])

    def test_builds_once_into_isolated_output(self) -> None:
        self.assertTrue(self.output.exists())
        self.assertNotEqual(self.output, WORKSPACE_OUTPUT)
        self.assertEqual(self.output.parent, Path(self.temporary_directory.name))

    def test_build_rejects_input_output_aliases_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.docx"
            body = Path(directory) / "body.md"

            for output_name in ("source.docx", "body.md"):
                with self.subTest(output=output_name):
                    source.write_bytes(SOURCE.read_bytes())
                    body.write_bytes(BODY_SOURCE.read_bytes())
                    output = Path(directory) / output_name
                    before = output.read_bytes()
                    with self.assertRaisesRegex(
                        ValueError, "output path must differ from input paths"
                    ):
                        self.builder.build(source, body, output)
                    self.assertEqual(output.read_bytes(), before)

    def test_main_accepts_explicit_historical_snapshot_paths_without_writing(self) -> None:
        with (
            patch.object(
                sys, "argv",
                [str(BUILDER), "--body", str(BODY_SOURCE), "--output", str(WORKSPACE_OUTPUT)],
            ),
            patch.object(self.builder, "build") as build,
            patch("builtins.print"),
        ):
            self.builder.main()
        build.assert_called_once_with(SOURCE, BODY_SOURCE, WORKSPACE_OUTPUT)

    def test_output_preserves_package_and_protected_front_matter(self) -> None:
        self.assertEqual(
            self.output.stat().st_mode & 0o777, SOURCE.stat().st_mode & 0o777
        )
        with ZipFile(SOURCE) as source, ZipFile(self.output) as output:
            self.assertEqual(set(output.namelist()), set(source.namelist()))
            for name in source.namelist():
                if name != DOCUMENT_XML:
                    self.assertEqual(output.read(name), source.read(name), name)

            source_children = body_children(source)
            output_children = body_children(output)
            self.assertEqual(canonical_hash(output_children[:214]), FRONT_MATTER_SHA256)
            self.assertEqual(
                [etree.tostring(node, method="c14n") for node in output_children[:214]],
                [etree.tostring(node, method="c14n") for node in source_children[:214]],
            )

            expected_section = etree.fromstring(etree.tostring(source_children[-1]))
            source_resets = [
                reset
                for node in source_children[214:-1]
                for reset in node.findall(f".//{W}pgNumType")
                if reset.get(f"{W}start") == "1"
            ]
            self.assertEqual(len(source_resets), 1)
            page_margin = expected_section.find(f"./{W}pgMar")
            self.assertIsNotNone(page_margin)
            expected_section.insert(
                expected_section.index(page_margin) + 1,
                etree.fromstring(etree.tostring(source_resets[0])),
            )
            self.assertEqual(
                etree.tostring(output_children[-1], method="c14n"),
                etree.tostring(expected_section, method="c14n"),
            )

    def test_output_contains_current_chapters_rules_and_no_stale_body(self) -> None:
        with ZipFile(self.output) as package:
            text = body_text(package)

        for chapter in range(1, 6):
            self.assertEqual(len(re.findall(rf"Chapter {chapter}\b", text)), 1)

        required = [
            "registrable-domain-disjoint and external evaluation",
            "RQ1",
            "RQ2",
            "RQ3",
            "H1",
            "H2",
            "H3",
            "Native label 0 maps to local is_phishing=1",
            "Native label 1 maps to local is_phishing=0",
            "publisher-provided, source-derived reference classifications",
            "not independently verified ground truth",
            "study-defined operating gates",
            "protocol v1.9",
            "The active baseline contract, frozen under protocol v1.7",
            "rq1-baselines-v2",
            "rq1-transformer-cascade-v2",
            "rq1-transformer-cascade-v1",
            "superseded_unrun",
            "frozen_implemented_not_run",
            FINAL_TRANSFORMER_CODE_COMMIT,
            PUBLICATION_AMENDMENT_COMMIT,
            PROTOCOL_RECORD_COMMIT,
            DURABILITY_HARDENING_COMMIT,
            SUPERSEDED_TRANSFORMER_CONTRACT_SHA256,
            TRANSFORMER_CONTRACT_SHA256,
            V19_MATRIX_SHA256,
            SEPTEMBER_REPORT_SHA256,
            "external source/domain shift",
            "the next 256 future requests",
            "This is a prospective, future-only policy replay",
            "does not establish a causal effect",
            "certified trusted-registry negatives",
            "Tranco reference-negative alert rate",
            "50,000 measured concurrency-64 requests",
            "completed_development_validation",
            "analyst-exposed but model-unscored",
            "No PhishVN record has been accessed",
            "H1: undecided",
            "H2: undecided",
            "H3: undecided",
            "235,795",
            "233,536",
            "2,259",
            "197,105",
            "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
            "Hussain et al., 2027",
            "Hussain, M., Abbas, J., Hussain, J., Gu, Y., & Wu, H. (2027)",
            "Sheng Siang, Y.",
        ]
        for phrase in required:
            self.assertIn(phrase, text, phrase)

        stale = [
            "AI Snapshot",
            "Public Millions",
            "security-engineer adjudication",
            "annotator adjudication",
            "300M-to-30M distillation",
            "v2025.12.10",
            "Table4-1_ModelMetrics",
            "97.79%",
            "98.71%",
            "real baseline fit has not been run",
            "group-test outcomes remain untouched",
            "untouched PhiUSIIL group-test records",
            "preregistered-style",
            "Candidate bands include all validation scores at each unique absolute distance",
            "For each of the fixed cascade and transformer-only system",
            "do not estimate generalization",
            "prospective, frozen experimental design",
            "frozen hash rule to a monitor-calibration stream",
            "final executable code commit",
            "That commit verifies",
            "The active protocol v1.7",
        ]
        for phrase in stale:
            self.assertNotIn(phrase, text, phrase)

    def test_transformer_method_is_reproducible_without_claiming_a_result(self) -> None:
        with ZipFile(self.output) as package:
            text = body_text(package)

        required = [
            "remaining non-ASCII characters are percent-encoded as UTF-8 bytes",
            "existing percent escapes and RFC 3986 reserved characters are preserved",
            "first 192 and last 64",
            "derived from training data only and ordered by ascending ASCII code point",
            "PAD=0, UNK=1, and character IDs beginning at 2",
            "right-padded, and PAD positions are masked",
            "PyTorch 2.7.1 in float32",
            "Four pre-normalization encoder layers",
            "six attention heads",
            "feed-forward width 768",
            "GELU activation",
            "dropout 0.1",
            "masked mean pooling",
            "Xavier-uniform",
            "official runtime is MPS",
            "Automatic mixed precision is disabled",
            "zero data-loader workers",
            "deterministic algorithms are enabled with warn_only=False",
            "binary cross-entropy with logits",
            "pos_weight=train_negative_count/train_positive_count",
            "AdamW",
            "learning rate 1e-4",
            "Training batch size is 256",
            "validation batch size 512",
            "maximum of 40 epochs",
            "gradient norm at 1.0",
            "strictly greater than the recorded best plus min_delta=1e-4",
            "fifth consecutive nonqualifying epoch",
            (
                "Any warning, nonfinite value, missing class, or deterministic-algorithm "
                "error stops the run"
            ),
            "mode 0700",
            "mode 0600",
            "vocabulary.json",
            "transformer-weights.npz",
            "transformer.json",
            "cascade.json",
            "SHA256SUMS",
            "Each destination is staged at a temporary path in its own parent",
            "installed with atomic no-replace semantics",
            "platform durability barrier",
            "F_FULLFSYNC on the official macOS/MPS runtime",
            "private output directory is installed first and its parent is flushed",
            (
                "public summary is installed last as the completion marker, and its parent "
                "is then flushed"
            ),
            "A completed result requires both destinations",
            "verification of their installed identities",
            (
                "A caught in-process BaseException removes only destinations created by "
                "that run"
            ),
            "attempts each removal independently",
            "attempts parent flushes without obscuring the original publication failure",
            "Publication is not cross-destination atomic",
            (
                "Abrupt process or host failure can leave private output without the public "
                "summary"
            ),
            "Either one-sided state is incomplete_not_result",
            "operator verifies it is stale and removes it before rerun",
            "both destinations already present is refused",
            "fit-transformer-cascade",
            "frozen_implemented_not_run",
            "No transformer fit, threshold calibration, or cascade result exists",
        ]
        for phrase in required:
            self.assertIn(phrase, text, phrase)

        self.assertIn(
            "The public summary may contain only counts, rates, configuration, "
            "versions, hashes, warnings, and status",
            text,
        )
        self.assertIn(
            "It excludes URLs, records, domains, sequences, predictions, "
            "coefficients, and weights",
            text,
        )

    def test_external_freeze_routing_and_stratum_boundaries_are_exact(self) -> None:
        expected = [
            (
                "H1 is supported only if each of length-only, Logistic-L1, and cascade "
                "has observed FPR <= 1% on both PhiUSIIL group-test negatives and "
                "certified trusted-registry negatives, and if the "
                "registrable-domain-clustered 95% bootstrap lower "
                "bounds exceed zero for all four prespecified recall differences. The "
                "differences are Logistic-L1 minus length-only and cascade minus "
                "Logistic-L1, each evaluated on PhiUSIIL group-test positives and NCSC "
                "gold positives. A positive difference in only one source, or an "
                "improvement accompanied by excess FPR, does not support H1."
            ),
            (
                "H2 is supported only if all four conditions hold: external-window "
                "detection is at least 80%; false alerts on the independent PhiUSIIL "
                "validation audit stream are no more than 5%; the alert policy's observed "
                "FPR on certified trusted-registry negatives is <= 1%; and the lower "
                "bound of the registrable-domain-clustered 95% bootstrap interval for "
                "false-negative-rate reduction on NCSC gold positives is greater than "
                "zero. The monitor measures departure in P(X). It does not by itself "
                "establish harmful drift, label shift, concept drift, or a causal change "
                "in model error."
            ),
            (
                "H3 requires both external-FPR safeguards for both the fixed cascade and "
                "transformer-only model. Exact one-sided 95% Clopper-Pearson upper "
                "confidence bounds will accompany the two certified-registry FPR "
                "estimates and the two Tranco control alert-rate estimates. The "
                "domain-clustered 95% bootstrap lower bound for recall(cascade) minus "
                "recall(transformer-only) on NCSC gold positives must be at least -0.02. "
                "Stage 2 must be invoked for no more than 30% of the primary reference "
                "manifest. Pooled p95 latency across the measured concurrency-64 runs "
                "must be <= 200 ms, and the request-error numerator divided by all 50,000 "
                "measured concurrency-64 requests must be < 0.1%."
            ),
            (
                "The study evaluates representation value, shift-aware routing, and "
                "inline viability under one auditable protocol. PhiUSIIL is the sole "
                "fitting and validation source. Its registrable-domain group-test "
                "partition provides internal evaluation and the operational replay "
                "reference. PhishVN v4 is reserved for one later external source/domain "
                "evaluation. Before any PhishVN record is accessed, the protocol, "
                "software environment, models, thresholds, population, ordering, and "
                "denominator rules, analysis code, and manifest specifications will be "
                "frozen. After schema verification and application of the frozen mapping "
                "and quarantine rules, the realized dataset hashes, row identifiers, "
                "counts, split manifest, and window manifest will be frozen before any "
                "model prediction. At the date of this working manuscript, PhiUSIIL "
                "preparation, the feature contract and pure raw-URL extractor, and the two "
                "RQ1 baseline fits are complete. The frozen training and validation "
                "procedure produced two development-validation operating points. These "
                "are selection-set outcomes, not group-test or external results. The "
                "group-test partition is analyst-exposed but model-unscored. No "
                "confirmatory hypothesis test has been run, and no PhishVN record has "
                "been accessed."
            ),
            (
                "Mechanical validity and conflict rules are applied before routing to "
                "define the retained external stream. Among retained records, source "
                "and confidence designations are applied only after label-blind routing "
                "to form analytic outcome strata; they do not change routing membership."
            ),
            (
                "These schema, validity, duplicate, and conflict rules define quarantine "
                "before routing. Once the retained stream is fixed, source and confidence "
                "designations are used after label-blind routing only to form analytic "
                "outcome strata; they do not remove retained records from routing."
            ),
            (
                "External records remain in published file order after only the "
                "applicable frozen quarantine. The first 256 requests use the fixed "
                "cascade. Windows then end at request 256 and every 64 requests "
                "thereafter. An alert routes exactly the next 256 future requests through "
                "the transformer. It never reroutes the records that produced the alert. "
                "Overlapping activations are unioned, a terminal activation is truncated "
                "at the stream end, and the transformer is invoked at most once for a "
                "request. This is a prospective, future-only policy replay; it does not "
                "establish a causal effect. Its paired error comparison tests whether the "
                "alert is useful for routing under the observed external source/domain "
                "shift."
            ),
            (
                "Before any record access, the protocol, software environment, source "
                "contract, model artifacts, vocabulary, feature schema, fitted "
                "preprocessing, model parameters, thresholds, cascade band, GMM, alert "
                "boundary, statistical code, full-test population and ordering rules, "
                "routing and outcome denominator rules, and manifest specifications are "
                "frozen and hashed. At first access, schema metadata, categorical values, "
                "license, provenance, and encoding are verified, after which the frozen "
                "mapping and quarantine rules are applied without model scoring. The "
                "realized dataset hashes, row identifiers, exclusion and retained counts, "
                "split manifest, and window manifest are then frozen before any prediction "
                "or inferential output is examined. A missing or inconsistent required "
                "field stops the affected analysis rather than prompting an imputed "
                "replacement."
            ),
            (
                "The primary benchmark manifest contains 10,000 URLs drawn without "
                "replacement from the PhiUSIIL group-test partition at a declared 1% "
                "phishing prevalence using seed 20260816. The declared prevalence is a "
                "measurement reference, not an estimate of production prevalence. "
                "Manifests at 0.1% and 5% are sensitivity analyses. The primary H3 replay "
                "uses the fixed cascade with no alert-triggered routing expansion. "
                "Future-only shift-period routing is evaluated separately under H2, and "
                "transformer-only worst-case load is reported separately."
            ),
            (
                "For both the fixed cascade and the transformer-only model, external "
                "observed FPR on certified trusted-registry negatives must be <= 1%. In "
                "addition, the Tranco reference-negative alert rate must be <= 1% for each "
                "system as a mandatory secondary safeguard. Exact one-sided 95% "
                "Clopper-Pearson upper bounds are reported for all four rates. H3 also "
                "uses the clustered recall-noninferiority bound, invocation, latency, and "
                "request-error gates stated in Section 1.4.3."
            ),
            (
                "The remaining work proceeds through prospective staged freezes. The "
                "character "
                "vocabulary, compact-transformer procedure, fixed-cascade procedure, CLI, "
                "tests, and private/public artifact publication code are implemented and "
                "frozen. First, one clean, noninteractive run will fit the transformer using "
                "the pinned training input and use validation only to select its threshold "
                "and the cascade band. Second, a separate GMM contract will freeze the "
                "validation-stream allocation, component-selection and window procedures, "
                "alert boundary, and audit rules before any GMM execution. Third, once all "
                "four RQ1 models, thresholds, evaluators, manifest "
                "specifications, selection rules, environment, and hashes are bound, the "
                "group-test partition will receive its single noninteractive pass. Fourth, "
                "before "
                "any PhishVN record is opened, the external population, ordering, routing- "
                "and outcome-denominator rules, analysis code, and manifest specifications "
                "will be frozen. Fifth, schema verification and frozen mapping and "
                "quarantine will establish the realized dataset hashes, row identifiers, "
                "counts, split manifest, and window manifest before prediction. The one-pass "
                "external "
                "evaluation and internal operational replay will then generate the evidence "
                "used "
                "to decide H1 through H3."
            ),
        ]
        with ZipFile(self.output) as package:
            paragraphs = direct_body_paragraphs(package)
        for paragraph in expected:
            self.assertEqual(paragraphs.count(paragraph), 1, paragraph)

    def test_active_baseline_contract_is_round_tripped(self) -> None:
        with ZipFile(self.output) as package:
            text = body_text(package)

        required = [
            "protocol v1.7",
            "ebd5b9f90157d6d21e8f22b7e1937c16dbaf60a577ec4f55cefc4709b604caef",
            "rq1-baselines-v2",
            "05d6d0831def7d26448c8dbdc8117800ea2448cdfc2aca2ad95489f22d2d11ba",
            "SAGA",
            "tol=1e-4",
            "unpenalized intercept",
            "rq1-scoring-integrity-v1",
            "historical protocol v1.4",
            "rq1-baselines-v1",
            "594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4",
            "ordered 25-element float64 vector",
            "StandardScaler(with_mean=True, with_std=True)",
            "A convergence warning is an error",
            "There is no hyperparameter or feature search",
            "Training data alone fit the scaler and classifier",
            "validation data alone select the threshold",
            "P(is_phishing=1)",
            "score >= threshold",
            "nextafter(maximum validation score, +infinity)",
            "target_not_met",
            "access.group_test_accessed=false",
            "process input boundary",
            "does not negate analyst access",
        ]
        for phrase in required:
            self.assertIn(phrase, text, phrase)

    def test_method_wording_and_closing_are_exact(self) -> None:
        baseline_record = (
            "The historical v1 contract and the active v2 contract are frozen at the "
            "identifiers and digests reported in Section 3.6. The v1.4 full-feature fit "
            "stopped as nonconverged. A provenance-incomplete tolerance observation was "
            "rejected, the v1.5 diagnostic stopped on a platform scoring warning, and the "
            "v1.6 diagnostic passed as training-only evidence after the scoring audit was "
            "frozen. The baseline command under protocol v1.7 then ran once from the clean "
            "freeze commit using only the pinned training and validation inputs. Its "
            "execution status is completed_development_validation."
        )
        next_milestone = (
            "The next milestone is one clean, noninteractive fit using only the pinned "
            "training and validation inputs. It will run the implemented transformer "
            "procedure once, select the transformer threshold and cascade band on "
            "validation only, and publish through the private/public artifact publication "
            "code. The two baseline operating points remain fixed. The group-test partition "
            "remains excluded until all four "
            "RQ1 models, thresholds, evaluator, manifest specifications, selection rules, "
            "environment, and hashes are frozen for one noninteractive pass. A separate GMM "
            "contract will then freeze its validation-stream allocation, "
            "component-selection and window procedures, alert boundary, and audit rules "
            "before any GMM execution. This preserves the systematic H1-then-GMM progression "
            "directed in the September 3 advisor report."
        )
        closing = (
            "This working manuscript moves the project from exploratory claims to an "
            "evidence-first prospective evaluation. The completed work to date includes "
            "the development-data preparation record, the active RQ1 feature contract and "
            "pure raw-URL extractor, and the validation-selected length-only and "
            "Logistic-L1 operating points with their integrity checks. The frozen "
            "transformer and cascade procedure is now implemented, tested, exposed through "
            "a constrained CLI, and connected to private/public artifact publication code, "
            "but it has not been fitted or calibrated. Whether the joint system meets the combined "
            "detection and service gates under domain-disjoint and external source/domain "
            "evaluation remains to be tested. Positive, mixed, or null findings can "
            "contribute by showing which constraints hold or fail under the frozen "
            "evaluation."
        )
        with ZipFile(self.output) as package:
            paragraphs = direct_body_paragraphs(package)
            text = body_text(package)
        self.assertEqual(paragraphs.count(baseline_record), 1)
        self.assertEqual(paragraphs.count(next_milestone), 1)
        self.assertEqual(paragraphs.count(closing), 1)
        self.assertIn("prospective staged-freeze design", text)
        self.assertNotIn("prospective, frozen experimental design", text)
        self.assertIn(
            "Candidate half-widths are the sorted unique absolute distances between "
            "the Logistic-L1 validation probabilities and its threshold.",
            text,
        )
        self.assertIn(
            "For both the fixed cascade and the transformer-only model, external "
            "observed FPR",
            text,
        )
        self.assertNotIn("The v1.7 contract then ran", text)
        self.assertNotIn("The scientific contribution remains a question", text)

    def test_evidence_tables_have_exact_rows_and_mappings(self) -> None:
        preparation = [
            [
                "Partition",
                "Registrable domains",
                "Rows",
                "Legitimate, is_phishing=0",
                "Phishing, is_phishing=1",
            ],
            ["Train", "137,973", "166,248", "94,373", "71,875"],
            ["Validation", "29,566", "32,695", "20,209", "12,486"],
            ["Group test", "29,566", "34,593", "20,267", "14,326"],
        ]
        metrics = [
            [
                "Model",
                "Validation threshold",
                "Validation recall",
                "Observed validation FPR",
                "One-sided 95% FPR upper bound",
            ],
            displayed_metric_row("Length-only"),
            displayed_metric_row("Logistic-L1"),
        ]
        status = [
            ["Item", "Evidence required for a decision", "Current status"],
            [
                "RQ1 / H1",
                (
                    "Frozen models and thresholds; paired internal and external predictions; "
                    "four clustered recall-difference intervals; both FPR strata"
                ),
                "Transformer/cascade procedure implemented but not run; H1: undecided",
            ],
            [
                "RQ2 / H2",
                (
                    "Fitted and calibrated GMM; independent false-alert audit; external "
                    "window trace; paired future-only policy outcomes"
                ),
                "H2: undecided",
            ],
            [
                "RQ3 / H3",
                (
                    "External detection and control rates; noninferiority interval; invocation "
                    "trace; five measured concurrency-64 HTTP runs"
                ),
                "H3: undecided",
            ],
        ]
        for expected in (preparation, metrics, status):
            self.assert_single_table(expected)

    def test_evidence_paragraphs_are_exact_and_unique(self) -> None:
        with ZipFile(self.output) as package:
            paragraphs = direct_body_paragraphs(package)
        prefixes = (
            "The aggregate baseline summary has SHA-256",
            "The length-only validation counts were",
            "The length-only scoring audit recorded",
        )
        records = [
            paragraph for paragraph in paragraphs if paragraph.startswith(prefixes)
        ]
        self.assertEqual(
            records, [ARTIFACT_PARAGRAPH, CONFUSION_PARAGRAPH, AUDIT_PARAGRAPH]
        )

    def test_validation_limit_and_chapter_4_date_are_exact(self) -> None:
        limit = (
            "These are validation-set operating points and selection-set outcomes. They "
            "are not independent confirmatory estimates of group-test or external "
            "generalization, and they do not decide whether one representation improves "
            "on another under H1."
        )
        date_record = (
            "This chapter records only evidence generated and verified as of September "
            "9, 2026. Development-data preparation, the feature contract and pure raw-URL "
            "extractor, and the frozen baseline-v2 training and validation run are "
            "complete. The transformer and fixed-cascade procedure code, tests, constrained "
            "CLI, and private/public artifact publication code are also complete at status "
            "frozen_implemented_not_run. The reported baseline values are "
            "development-validation operating points used for later locked evaluation. No "
            "transformer fit, threshold calibration, cascade result, group-test prediction, "
            "external evaluation, monitoring result, HTTP replay, or confirmatory "
            "hypothesis test is reported."
        )
        with ZipFile(self.output) as package:
            paragraphs = direct_body_paragraphs(package)
            text = body_text(package)
        self.assertEqual(paragraphs.count(limit), 1)
        dated = [
            paragraph
            for paragraph in paragraphs
            if paragraph.startswith(
                "This chapter records only evidence generated and verified as of"
            )
        ]
        self.assertEqual(dated, [date_record])
        self.assertEqual(text.count("September 9, 2026"), 2)
        self.assertNotIn("as of September 4, 2026", text)
        self.assertNotIn("as of September 3, 2026", text)

    def test_pandoc_round_trip_contains_frozen_questions_rules_and_status(self) -> None:
        completed = subprocess.run(
            ["pandoc", str(self.output), "--to=plain"],
            capture_output=True,
            check=True,
            text=True,
        )
        text = " ".join(completed.stdout.split())
        required = [
            (
                "What incremental value do structural URL features and selective "
                "character-model escalation provide under registrable-domain-disjoint and "
                "external evaluation?"
            ),
            "Logistic-L1 minus length-only and cascade minus Logistic-L1",
            (
                "Transformer-only remains a comparator and operational reference, not a "
                "third primary H1 gate"
            ),
            "system contribution, not a pure causal isolation",
            (
                "Every complete 256-request window of the retained external stream is a "
                "prespecified external-shift window"
            ),
            "score strictly greater than the frozen boundary",
            "denominator is all complete windows",
            "Overlapping windows count separately",
            "terminal incomplete window is excluded from the rate",
            "prior complete window",
            "same complete-window numerator and denominator rule",
            "An alert routes exactly the next 256 future requests through the transformer",
            "It never reroutes the records that produced the alert",
            "Overlapping activations are unioned",
            "terminal activation is truncated at the stream end",
            "does not establish a causal effect",
            (
                "For both the fixed cascade and the transformer-only model, external observed "
                "FPR on certified trusted-registry negatives must be <= 1%"
            ),
            "Tranco reference-negative alert rate must be <= 1% for each system",
            (
                "Exact one-sided 95% Clopper-Pearson upper bounds are reported for all four "
                "rates"
            ),
            (
                "publisher-provided, source-derived reference classifications, not "
                "independently verified ground truth"
            ),
            "group test remains analyst-exposed but model-unscored",
            "No PhishVN record has been accessed",
            (
                "Manual review is permitted only as separately reported post hoc descriptive "
                "error analysis"
            ),
            "cannot assign or override labels",
            "frozen_implemented_not_run",
            PUBLICATION_AMENDMENT_COMMIT,
            PROTOCOL_RECORD_COMMIT,
            DURABILITY_HARDENING_COMMIT,
            FINAL_TRANSFORMER_CODE_COMMIT,
            SUPERSEDED_TRANSFORMER_CONTRACT_SHA256,
            TRANSFORMER_CONTRACT_SHA256,
            V19_MATRIX_SHA256,
            "No transformer fit, threshold calibration, or cascade result exists",
            "H1: undecided. H2: undecided. H3: undecided.",
        ]
        for phrase in required:
            self.assertIn(phrase, text, phrase)

    def test_output_has_no_comments_or_tracked_revisions(self) -> None:
        with ZipFile(self.output) as package:
            lower_names = [name.lower() for name in package.namelist()]
            self.assertFalse(
                any("comment" in name or "people" in name for name in lower_names)
            )
            forbidden = {
                f"{W}ins",
                f"{W}del",
                f"{W}moveFrom",
                f"{W}moveTo",
                f"{W}commentRangeStart",
                f"{W}commentRangeEnd",
                f"{W}commentReference",
                f"{W}trackRevisions",
            }
            for name in package.namelist():
                if not name.endswith(".xml"):
                    continue
                root = etree.fromstring(package.read(name))
                present = {element.tag for element in root.iter()}
                self.assertFalse(forbidden & present, name)
                revision_authors = [
                    value
                    for element in root.iter()
                    for attribute, value in element.attrib.items()
                    if etree.QName(attribute).localname == "author"
                ]
                self.assertEqual(revision_authors, [], name)

    def test_status_table_and_next_milestone_have_deliberate_pagination(self) -> None:
        with ZipFile(self.output) as package:
            root = document_root(package)
        tables = root.findall(f".//{W}tbl")
        self.assertGreaterEqual(len(tables), 3)
        for table in tables:
            rows = table.findall(f"./{W}tr")
            for row in rows:
                self.assertIsNotNone(row.find(f"./{W}trPr/{W}cantSplit"))
            for row in rows[:-1]:
                cell_paragraphs = row.findall(f"./{W}tc/{W}p")
                self.assertTrue(cell_paragraphs)
                for paragraph in cell_paragraphs:
                    self.assertIsNotNone(paragraph.find(f"./{W}pPr/{W}keepNext"))

        paragraphs = root.findall(f".//{W}body/{W}p")
        status_heading = [
            paragraph
            for paragraph in paragraphs
            if element_text(paragraph).strip() == "4.5 Research Question Status"
        ]
        self.assertEqual(len(status_heading), 1)
        self.assertIsNone(status_heading[0].find(f"./{W}pPr/{W}pageBreakBefore"))

        milestone_heading = [
            paragraph
            for paragraph in paragraphs
            if element_text(paragraph).strip() == "4.6 Next Executable Milestone"
        ]
        self.assertEqual(len(milestone_heading), 1)
        self.assertIsNone(milestone_heading[0].find(f"./{W}pPr/{W}pageBreakBefore"))
        self.assertIsNotNone(milestone_heading[0].find(f"./{W}pPr/{W}keepNext"))

    def test_reference_entries_do_not_split_across_pages(self) -> None:
        with ZipFile(self.output) as package:
            children = body_children(package)
        reference_index = next(
            index
            for index, element in enumerate(children)
            if element_text(element).strip() == "References"
        )
        references = [
            element
            for element in children[reference_index + 1 :]
            if element.tag == f"{W}p"
        ]
        self.assertGreaterEqual(len(references), 10)
        for paragraph in references:
            self.assertIsNotNone(paragraph.find(f"./{W}pPr/{W}keepLines"))

    def test_metadata_names_only_krti(self) -> None:
        with ZipFile(self.output) as package:
            core = etree.fromstring(package.read("docProps/core.xml"))
        names = [
            element.text.strip()
            for element in core
            if etree.QName(element).localname in {"creator", "lastModifiedBy"}
            and element.text
        ]
        self.assertEqual(names, ["Tallam, Krti", "Tallam, Krti"])

    def test_body_source_is_current_and_builder_is_narrow(self) -> None:
        body = BODY_SOURCE.read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"^# Chapter [1-5]\b", body, re.MULTILINE)), 5)
        self.assertIn("## References", body)
        self.assertNotIn("AI Snapshot", body)
        self.assertNotIn("Public Millions", body)

        code = BUILDER.read_text(encoding="utf-8")
        self.assertIn("zipfile", code)
        self.assertIn("lxml", code)
        self.assertNotIn("python-docx", code)
        self.assertNotIn("pandoc", code.lower())


class September17SnapshotTests(unittest.TestCase):
    def current_body(self) -> str:
        self.assertTrue(CURRENT_BODY_SOURCE.is_file())
        return CURRENT_BODY_SOURCE.read_text(encoding="utf-8")

    def test_september_9_snapshot_bytes_are_unchanged(self) -> None:
        self.assertEqual(
            sha256(BODY_SOURCE.read_bytes()),
            "da4df76d4c86e26ad959ea227cd336810c24fd2de9372218b9f09e54eaf4b251",
        )
        self.assertEqual(
            sha256(WORKSPACE_OUTPUT.read_bytes()),
            "9b7a5301352d7cc42a08cf1c2a1afbda83dc3ef7df92d77aa8968480774521c1",
        )

    def test_default_build_targets_the_new_dated_snapshot(self) -> None:
        builder = load_builder()
        with (
            patch.object(sys, "argv", [str(BUILDER)]),
            patch.object(builder, "build") as build,
            patch("builtins.print"),
        ):
            builder.main()
        build.assert_called_once_with(SOURCE, CURRENT_BODY_SOURCE, CURRENT_OUTPUT)

    def test_hypotheses_and_reference_list_are_preserved_exactly(self) -> None:
        old = BODY_SOURCE.read_text(encoding="utf-8")
        current = self.current_body()
        self.assertEqual(
            markdown_section(current, "## 1.4", "## 1.5"),
            markdown_section(old, "## 1.4", "## 1.5"),
        )
        self.assertEqual(current.split("## References", 1)[1], old.split("## References", 1)[1])
        for heading, following in (
            ("## 1.5", "# Chapter 2"),
            ("# Chapter 2", "# Chapter 3"),
            ("## 3.7", "## 3.9"),
            ("## 3.10", "## 3.13"),
            ("## 3.14", "# Chapter 4"),
            ("## 4.3", "## 4.4"),
            ("## 5.5", "## 5.6"),
        ):
            self.assertEqual(
                markdown_section(current, heading, following),
                markdown_section(old, heading, following),
                heading,
            )
        self.assertEqual(
            markdown_section(current, "Every complete 256-request", "## 3.10"),
            markdown_section(old, "Every complete 256-request", "## 3.10"),
        )

    def test_current_status_is_dated_and_does_not_infer_results(self) -> None:
        body = self.current_body()
        status = markdown_section(body, "## 4.1", "## 4.2")
        self.assertIn("September 17, 2026", status)
        self.assertIn("2026-09-17T18:21:12Z", body)
        self.assertIn(DURABILITY_HARDENING_COMMIT, body)
        self.assertIn("35257955477", body)
        for hypothesis in ("H1", "H2", "H3"):
            self.assertIn(f"{hypothesis}: undecided", body)
        self.assertIn("analyst-exposed but model-unscored", body)
        self.assertIn("No PhishVN record has been accessed", body)
        self.assertNotIn("frozen_implemented_not_run", body)

    def test_execution_cutoff_is_consistent_and_after_the_stopped_attempt(self) -> None:
        body = self.current_body()
        cutoffs = re.findall(r"execution cutoff,? `([^`]+)`", body)
        self.assertEqual(len(cutoffs), 2)
        self.assertEqual(len(set(cutoffs)), 1)
        self.assertGreaterEqual(
            datetime.fromisoformat(cutoffs[0].replace("Z", "+00:00")),
            datetime.fromisoformat("2026-09-17T19:24:27+00:00"),
        )

    def test_gmm_allocation_is_frozen_and_audit_failure_is_reported(self) -> None:
        body = self.current_body()
        method = markdown_section(body, "## 3.9", "## 3.10")
        self.assertIn("rq2-gmm-development-v1", body)
        self.assertIn("protocol v1.10", body)
        for fact in (
            "rq2-gmm-validation-v1", "20260816", "SHA-256", "NUL",
            "floor(D/2)", "source order", "26", "training", "BIC",
            "smaller", "256", "64", "linear", "20 * alert_windows <= complete_windows",
        ):
            self.assertIn(fact, method)
        self.assertNotIn("no GMM allocation choice has been made", body)
        self.assertNotIn("separate GMM contract will freeze", body)
        self.assertNotIn("no GMM result", body)
        self.assertIn("28 of 252", body)
        self.assertIn("11.1111%", body)
        self.assertIn("-67.45792380813624", body)
        self.assertIn("false_alert_gate_met=false", body)
        self.assertIn("was not retuned", body)
        self.assertIn(
            "6f695138a302e854e1e5af590152e289486affe8ccdf75510ca9a5dcaad3b523",
            body,
        )

    def test_stopped_transformer_attempt_and_diagnosis_are_not_model_results(self) -> None:
        body = self.current_body()
        for fact in (
            "2026-09-17T19:24:27Z", "3,795.28 seconds", "exit code 2",
            "stage-one artifact threshold does not match the supplied scores",
            "no accepted transformer or cascade artifacts",
            "no-fit development-validation diagnosis", "einsum", "C-order", "F-order",
            "zero changed decisions at the accepted threshold",
            "guard was not loosened", "threshold was not changed",
            "Any warning during transformer training or validation",
        ):
            self.assertIn(fact, body)
        self.assertNotIn("official development run is still active", body)
        self.assertNotIn("finish the running transformer/cascade", body)

    def test_controlled_retry_is_running_not_accepted_at_exact_cutoff(self) -> None:
        body = self.current_body()
        for fact in (
            "2026-09-17T20:43:08Z", "2026-09-17T20:42:45Z",
            "e866441f2ff858472d031b8d358fd469897c6a65", "35272134401",
            "controlled retry remains active without a completed result",
            "601 tests", "124 tests", "No scientific contract changed",
        ):
            self.assertIn(fact, body)
        self.assertNotIn("2026-09-17T18:55:44Z", body)

    def test_gmm_description_is_post_hoc_and_preserves_failed_gate(self) -> None:
        body = self.current_body()
        for fact in (
            "post hoc descriptive", "saved audit", "nine consecutive alert runs",
            "not 28 independent events", "482,882.37", "uniform shift",
            "GMM's portable stage-one probability path is unchanged",
            "dc3def7dddfeb2c161d567d89b9cd5fcf0457016",
            "fe0e8c9fdae32fc48118102b71e0cda7f771b9ab77e49c4146d2f3479c052712",
        ):
            self.assertIn(fact, body)

    def test_remaining_execution_dependencies_and_decision_rationales_are_explicit(self) -> None:
        body = self.current_body()
        for fact in (
            "not an optimal feature set", "not causal feature importance",
            "does not guarantee the audit fraction", "portable inference loaders",
            "domain-clustered", "RNG", "cluster weighting", "percentile interpolation",
            "empty-stratum", "composed H2", "selective inference service",
            "logical invocation mask", "not measured transformer work saved",
            "single noninteractive raw-partition pass", "replay manifests",
        ):
            self.assertIn(fact, body)

    def test_new_snapshot_preserves_shell_and_roundtrips_current_method(self) -> None:
        body = self.current_body()
        builder = load_builder()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / CURRENT_OUTPUT.name
            builder.build(SOURCE, CURRENT_BODY_SOURCE, output)
            with ZipFile(SOURCE) as original, ZipFile(output) as generated:
                self.assertEqual(set(original.namelist()), set(generated.namelist()))
                for name in original.namelist():
                    if name != DOCUMENT_XML:
                        self.assertEqual(original.read(name), generated.read(name), name)
                self.assertEqual(
                    canonical_hash(body_children(generated)[:214]), FRONT_MATTER_SHA256
                )
                self.assertEqual(len(document_tables(generated)), 3)
            result = subprocess.run(
                ["pandoc", str(output), "-t", "plain"],
                check=True, text=True, capture_output=True,
            )
        normalized = " ".join(result.stdout.split())
        for section, following in (("## 1.4", "## 1.5"), ("## 3.9", "## 3.10")):
            for paragraph in markdown_section(body, section, following).split("\n\n"):
                if not paragraph.strip() or paragraph.startswith("###"):
                    continue
                expected = " ".join(builder._clean_inline(paragraph).split())
                self.assertIn(expected, normalized)


if __name__ == "__main__":
    unittest.main(verbosity=2)
