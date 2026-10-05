"""Seal an explicit aggregate-only final delivery; preserve previous packages."""

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, is_zipfile

import build_complete_revision_20261002 as final
from check_complete_package_20261002 import (
    CONTEXT, FROZEN, HERE, ORIGINAL_PACKAGE, REVISION, ROOT, digest, read_json,
    require, verify_hashes,
)

DESTINATION = CONTEXT / "deliverables/Tallam_Dissertation_Complete_2026-10-02"
FORBIDDEN = {"model.json", "predictions.jsonl", "execution-manifest-v1.json",
             "service-recovery-authorization-v2.json"}


def add_member(selected, source, target):
    name = Path(target)
    require(not name.is_absolute() and ".." not in name.parts and target not in selected,
            f"Unsafe or duplicate member: {target}")
    require(name.name not in FORBIDDEN and "private-authorization" not in name.name
            and name.suffix != ".jsonl", f"Sensitive member: {target}")
    require(source.is_file() and not source.is_symlink(), f"Invalid source: {source}")
    selected[target] = source


def select_files(qa):
    selected = {}

    def add(source, target):
        add_member(selected, source, target)

    for kind, target in [("README", "README.txt"), ("COVERAGE", "COVERAGE_AND_VERIFICATION.txt"),
                         ("DATA_DICTIONARY", "DATA_DICTIONARY.txt")]:
        add(HERE / f"complete_package_{kind}_20261002.txt", target)
    add(ORIGINAL_PACKAGE / "DATA_DICTIONARY.txt", "ORIGINAL_DATA_DICTIONARY.txt")
    for name in qa["artifact_sha256"]:
        add(final.FINAL / name, name)
    add(final.BODY.parent / "Current_Abstract_2026-10-01.txt", "manuscript/Current_Abstract_2026-10-02.txt")
    for extension in ("png", "pdf"):
        add(HERE / f"gwu-system-dataflow-20261001.{extension}", f"manuscript/gwu-system-dataflow-20261001.{extension}")
    figure_manifest = read_json(ROOT / "figures/detection-figure-manifest.json")
    for name in figure_manifest["outputs"]:
        add(ROOT / "figures" / name, f"manuscript/{name}")
    add(ROOT / "figures/detection-figure-manifest.json", "verification/final/detection-figure-manifest.json")
    for line in (ORIGINAL_PACKAGE / "SHA256SUMS.txt").read_text().splitlines():
        name = line.split("  ", 1)[1]
        section, _, remainder = name.partition("/")
        if section in {"aggregate-data", "provenance", "verification"}:
            add(ORIGINAL_PACKAGE / name, f"{section}/original/{remainder}")
        elif section == "analysis-scripts":
            add(ORIGINAL_PACKAGE / name, name)
    add(ORIGINAL_PACKAGE / "SHA256SUMS.txt", "verification/original/SHA256SUMS-original-package.txt")
    add(CONTEXT / "reviews/final-package-receipt-20261001.json", "verification/original/archive-receipt.json")
    for source, target, key in [("verified-detection-v1", "detection-D", "export_hashes"),
                                ("verified-service-v2", "service-S", "aggregate_sha256")]:
        verification = read_json(ROOT / source / "verification.json")
        for name in [*verification[key], "verification.json"]:
            add(ROOT / source / name, f"aggregate-data/{target}/{name}")
    for name in ("comparison-specification-v1.md", "diagnosis-and-design.md", "methodological-source-review.md",
                 "implementation-clarifications-v1.md", "service-implementation-clarifications-v1.md",
                 "implementation-review-v1.md", "retrieval-correction-1.md", "service-recovery-amendment-v2.md",
                 "service-v1-preservation.json", "regression-verification-v1.md"):
        add(ROOT / name, f"provenance/followup/{name}")
    for name in ("followup-full-suite.txt", "interrupt-fixture-reproduction.txt", "interrupt-fixture-correction.txt",
                 "interrupt-fixture-correction-2.txt", "service-recovery-tests-v2.txt", "final-focused-verification.txt"):
        add(ROOT / name, f"verification/followup/{name}")
    add(ROOT / "verification-tests/test_operational_child_interruptions.py",
        "analysis-scripts/supplemental-fixture/test_operational_child_interruptions.py")
    for name in ("final-content-preservation-qa.json", "document-package-tests.txt", "document-package-lint.txt"):
        add(final.FINAL / name, f"verification/final/{name}")
    for folder in (final.BODY.parent, final.PPTX.parent):
        add(folder / "render-check.json", f"verification/final/{folder.name}-render-check.json")
    for name in ("gwu-navigation-20261001.json", "gwu-pagination-map-20261001.json", "navigation-targets.json"):
        add(final.BODY.parent / name, f"verification/final/{name}")
    for name in ("build_complete_revision_20261002.py", "check_complete_package_20261002.py",
                 "package_complete_20261002.py", "test_complete_package_20261002.py", "test_complete_revision_20261002.py",
                 "build_followup_working_20261001.py", "integrate_followup_manuscript_20261001.py",
                 "check_followup_render_20261001.py", "draw_followup_detection_20261001.py",
                 "verify_followup_detection_20261001.py", "verify_followup_service_20261002.py",
                 "test_verify_followup_detection_20261001.py", "test_verify_followup_service_20261002.py",
                 "run_followup_service_recovery_20261002.py", "test_followup_service_recovery_20261002.py",
                 "test_followup_manuscript_20261001.py", "test_followup_gwu_format_20261001.py",
                 "test_followup_render_20261001.py", "test_followup_figures_20261001.py"):
        add(HERE / name, f"analysis-scripts/dissertation/{name}")
    for name in ("build_followup_deck_20261001.py", "check_followup_deck_render_20261001.py",
                 "test_followup_deck_20261001.py", "test_followup_deck_render_20261001.py"):
        add(CONTEXT / "gwu_advisor_update_source" / name, f"analysis-scripts/advisor/{name}")
    changed = subprocess.check_output(["git", "-C", str(FROZEN), "diff-tree", "--no-commit-id", "--name-only", "-r", REVISION], text=True)
    for name in [*changed.splitlines(), "pyproject.toml", "uv.lock", "LICENSE", "CITATION.cff"]:
        require(Path(name).suffix in {".py", ".md", ".toml", ".lock", ".cff", ""}, "Unexpected frozen source member")
        add(FROZEN / name, f"provenance/followup-source/{name}")
    return selected


def check_credential_signatures(selected):
    signatures = [rb"gh[pousr]_[A-Za-z0-9]{30,}", rb"github_pat_[A-Za-z0-9_]{30,}",
                  rb"sk-(?:proj|svcacct)-[A-Za-z0-9_-]{30,}",
                  rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
    for target, source in selected.items():
        payloads = [source.read_bytes()]
        if is_zipfile(source):
            with ZipFile(source) as archive:
                payloads.extend(archive.read(name) for name in archive.namelist() if name.endswith(".xml"))
        require(not any(re.search(pattern, payload) for pattern in signatures for payload in payloads),
                f"Potential credential in {target}")


def verify_archive(path, expected):
    with ZipFile(path) as archive:
        require(archive.testzip() is None, "Archive CRC failed")
        require(len(archive.namelist()) == len(expected) and set(archive.namelist()) == set(expected), "Archive inventory mismatch")
        for name, expected_hash in expected.items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected_hash, f"Archive content mismatch: {name}")


def main():
    require(not DESTINATION.exists() and not DESTINATION.with_suffix(".zip").exists(), "Final destination already exists")
    qa = read_json(final.FINAL / "final-content-preservation-qa.json")
    require(qa["status"] == "verified" and qa["frozen_revision"] == REVISION, "Final QA missing")
    verify_hashes(final.FINAL, qa["artifact_sha256"])
    selected = select_files(qa)
    tests = (final.FINAL / "document-package-tests.txt").read_text()
    require(re.search(r"Ran \d+ tests", tests) and "\nOK\n" in tests, "Final tests not verified")
    require("All checks passed!" in (final.FINAL / "document-package-lint.txt").read_text(), "Final lint not verified")
    check_credential_signatures(selected)
    with TemporaryDirectory(prefix="final-dissertation-", dir=DESTINATION.parent) as temporary:
        staging = Path(temporary) / DESTINATION.name
        staging.mkdir()
        for name, source in selected.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            require(digest(target) == digest(source), f"Copy mismatch: {name}")
        pins = {
            "original_measured_revision": "77d128377ce5b401437d7179f5cd78fb4294b72c",
            "followup_measured_revision": REVISION,
            "repository": "https://github.com/KrtiT/automated-phishing-detection-public",
            "followup_source_scope": "The 18 changed files at the local frozen revision plus environment/license metadata; not a complete repository snapshot.",
            "source_dependencies": "Scripts retain original controlled-workspace paths, dependencies and input bindings; audit source, not a standalone rerun bundle.",
            "authorization": "Operator approval and prospective/disclosed amendments; no advisor or institutional approval asserted. Private execution bindings remain outside this package.",
            "excluded": ["original datasets", "row-level URL and prediction records", "models/weights", "private authorizations/capabilities", "credentials", "full host/process logs"],
            "historical_records": "Earlier pending statuses in original/freeze documents are preserved history, not the current completion state.",
        }
        (staging / "provenance/CODE_AND_ENVIRONMENT.json").write_text(json.dumps(pins, indent=2) + "\n")
        report = {
            "status": "assembled_and_checked", "created_at": datetime.now(timezone.utc).isoformat(),
            "manuscript_pages": 132, "deck_slides": 25, "primary_checks": 22, "operational_groups": 25,
            "original_cells": 125, "followup_detection_rows": 8622, "followup_service_arms": 80,
            "followup_service_measured_requests": 800000, "followup_service_errors": 1,
            "original_decisions": "H1/H2/H3 not supported", "followup_decisions": "D and strict S not supported",
            "focused_tests_passed": int(re.search(r"Ran (\d+) tests", tests).group(1)),
            "broad_research_regression": qa["broad_research_regression"],
            "qa_record": "verification/final/final-content-preservation-qa.json",
            "source_sha256": {name: digest(source) for name, source in sorted(selected.items())},
        }
        (staging / "package-verification.json").write_text(json.dumps(report, indent=2) + "\n")
        members = sorted(path for path in staging.rglob("*") if path.is_file())
        (staging / "SHA256SUMS.txt").write_text("".join(f"{digest(path)}  {path.relative_to(staging).as_posix()}\n" for path in members))
        subprocess.run(["shasum", "-a", "256", "-c", "SHA256SUMS.txt"], cwd=staging, check=True, capture_output=True)
        members = sorted(path for path in staging.rglob("*") if path.is_file())
        archive_path = Path(temporary) / (DESTINATION.name + ".zip")
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
            for path in members:
                archive.write(path, (Path(staging.name) / path.relative_to(staging)).as_posix())
        verify_archive(archive_path, {(Path(staging.name) / path.relative_to(staging)).as_posix(): digest(path) for path in members})
        archive_hash = digest(archive_path)
        staging.rename(DESTINATION)
        archive_path.rename(DESTINATION.with_suffix(".zip"))
    DESTINATION.with_suffix(".zip.sha256").write_text(f"{archive_hash}  {DESTINATION.name}.zip\n")
    receipt = {"status": "verified", "created_at": datetime.now(timezone.utc).isoformat(),
               "directory": str(DESTINATION), "zip": str(DESTINATION.with_suffix(".zip")),
               "zip_sha256": archive_hash, "files_in_zip": len(members),
               "zip_bytes": DESTINATION.with_suffix(".zip").stat().st_size,
               "manifest_check": "passed", "zip_crc_inventory_and_byte_checks": "passed"}
    (final.FINAL / "package-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
