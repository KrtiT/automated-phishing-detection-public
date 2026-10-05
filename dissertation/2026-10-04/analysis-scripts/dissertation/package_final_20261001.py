"""Assemble the final, explicitly selected dissertation files and verify the archive."""

import csv
import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

HERE = Path(__file__).resolve().parent
CONTEXT = HERE.parent
DESTINATION = CONTEXT / "deliverables/Tallam_Dissertation_Results_2026-10-01"
REVIEW = CONTEXT / "reviews"
RENDER = REVIEW / "gwu-native-render-20261001"
MEASURED = CONTEXT / "gwu_working/study-series-development"
REVISION = "77d128377ce5b401437d7179f5cd78fb4294b72c"
CSV_COUNTS = {
    "primary-gates.csv": 22, "paired-contrasts.csv": 6,
    "operational-runs.csv": 125, "operational-groups.csv": 25,
    "secondary-metrics.csv": 153, "calibration-bins.csv": 430,
    "low-fpr-score-curves.csv": 43, "prevalence-projections.csv": 129,
    "source-contingency.csv": 5, "seed-logical-invocations.csv": 35,
    "external-monitor-windows.csv": 396, "external-psi-features.csv": 3432,
    "probe-decisions-and-scores.csv": 20, "probe-monitors-and-scores.csv": 12,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_files():
    selected = {}

    def add(source, target):
        if target in selected or source.is_symlink() or not source.is_file():
            raise ValueError(f"Invalid package member: {source} -> {target}")
        selected[target] = source

    for kind in ("README", "DATA_DICTIONARY", "COVERAGE"):
        name = "COVERAGE_AND_VERIFICATION" if kind == "COVERAGE" else kind
        add(HERE / f"final_package_{kind}_20261001.txt", f"{name}.txt")
    for name in ("Tallam_Krti_Praxis_GWU_2026-10-01.docx",
                 "Tallam_Krti_Praxis_v3_Body_Results_2026-10-01.md",
                 "Current_Abstract_2026-10-01.txt", "gwu-system-dataflow-20261001.png",
                 "gwu-system-dataflow-20261001.pdf"):
        add(HERE / name, f"manuscript/{name}")
    add(RENDER / "Tallam_Krti_Praxis_GWU_2026-10-01.pdf",
        "manuscript/Tallam_Krti_Praxis_GWU_2026-10-01.pdf")
    for extension in ("pptx", "pdf"):
        name = f"Tallam_Praxis_Advisor_Results_2026-10-01.{extension}"
        source = CONTEXT / "deliverables" / name if extension == "pptx" else RENDER / name
        add(source, f"advisor/{name}")
    name = "Tallam_Praxis_Advisor_Results_2026-10-01_Speaker_Notes.txt"
    add(CONTEXT / "deliverables" / name, f"advisor/{name}")
    for name in (*CSV_COUNTS, "primary-results.json", "complete-secondary-results.json",
                 "verification.json", "secondary-verification.json"):
        add(HERE / "final-evidence-20261001" / name, f"aggregate-data/{name}")
    for name in ("Historical_Evidence_Supplement_2026-10-01.md",
                 "advisor-deck-scope-crosswalk-20261001.md"):
        add(HERE / name, f"provenance/{name}")
    add(CONTEXT / "plans/2026-09-29-final-rqh-synthesis-checklist.md",
        "provenance/historical-synthesis-requirements-checklist.md")
    for name in ("pyproject.toml", "uv.lock", "CITATION.cff", "LICENSE"):
        add(MEASURED / name, f"provenance/measured-source/{name}")
    tracked = subprocess.check_output(["git", "-C", str(MEASURED), "ls-files", "data", "docs"], text=True)
    for name in tracked.splitlines():
        add(MEASURED / name, f"provenance/measured-source/{name}")
    for name in ("verify_final_evidence_20261001.py", "export_final_secondary_20261001.py",
                 "synthesize_final_20261001.py", "build_results_reading_edition_20261001.py",
                 "build_v3_working_manuscript.py", "draw_system_dataflow_20261001.py",
                 "check_gwu_render_20261001.py", "check_final_deliverables_20261001.py",
                 "package_final_20261001.py", "test_gwu_manuscript_format_20261001.py",
                 "test_final_package_20261001.py", "test_build_v3_working_manuscript.py"):
        add(HERE / name, f"analysis-scripts/dissertation/{name}")
    for name in ("gwu-navigation-20261001.json", "gwu-pagination-map-20261001.json"):
        add(HERE / name, f"verification/{name}")
    for name in ("build_advisor_results_20261001.py", "build_advisor_update_2026_09_03.py"):
        add(CONTEXT / "gwu_advisor_update_source" / name, f"analysis-scripts/advisor/{name}")
    for name in ("final-evidence-verification-20261001.log",
                 "final-secondary-verification-20261001.log", "gwu-format-final-20261001.log",
                 "series-publication-full-suite-20261001.log",
                 "series-publication-final-focused-20261001.log",
                 "series-publication-final-identity-20261001.txt",
                 "series-publication-remote-ci-20261001.json", "final-delivery-qa-20261001.txt",
                 "final-reference-metadata-20261001.json",
                 "statistical-reference-verification-20261001.json",
                 "holm-reference-verification-20261001.json"):
        add(REVIEW / name, f"verification/{name}")
    for name in ("render-check.json", "deliverables-check.json", "navigation-targets.json"):
        add(RENDER / name, f"verification/{name}")
    return selected


def validate_sources(selected):
    observed = subprocess.check_output(["git", "-C", str(MEASURED), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(MEASURED), "status", "--porcelain"], text=True)
    if observed != REVISION or dirty:
        raise ValueError("Measured checkout no longer clean and pinned")
    verification = json.loads(selected["aggregate-data/verification.json"].read_text())
    if verification["status"] != "verified" or verification["measurement_revision"] != REVISION:
        raise ValueError("Unverified evidence or revision mismatch")
    for name, count in CSV_COUNTS.items():
        with selected[f"aggregate-data/{name}"].open() as source:
            if len(list(csv.DictReader(source))) != count:
                raise ValueError(f"Incomplete table: {name}")
    primary = json.loads(selected["aggregate-data/primary-results.json"].read_text())["primary"]
    for hypothesis in primary["hypotheses"].values():
        if not hypothesis["complete"] or hypothesis["decision"] != "not_supported":
            raise ValueError("Primary decision changed or incomplete")
    with selected["aggregate-data/operational-runs.csv"].open() as source:
        runs = list(csv.DictReader(source))
    if sum(int(row["request_count"]) for row in runs) != 1243505:
        raise ValueError("Request total mismatch")
    if sum(int(row["request_errors"]) for row in runs) != 1901:
        raise ValueError("Error total mismatch")
    tests = selected["verification/gwu-format-final-20261001.log"].read_text()
    if "Ran 44 tests" not in tests or "\nOK\n" not in tests:
        raise ValueError("Document tests not verified")
    suite = selected["verification/series-publication-full-suite-20261001.log"].read_text()
    if "12989 passed, 2 skipped" not in suite:
        raise ValueError("Publication test suite not verified")
    report = json.loads(selected["verification/deliverables-check.json"].read_text())
    for name, expected in report["artifact_sha256"].items():
        candidates = [source for target, source in selected.items() if Path(target).name == name]
        if len(candidates) != 1 or digest(candidates[0]) != expected:
            raise ValueError(f"Unchecked or changed artifact: {name}")
    patterns = [rb"gh[pousr]_[A-Za-z0-9]{30,}", rb"github_pat_[A-Za-z0-9_]{30,}",
                rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", rb"sk-proj-[A-Za-z0-9_-]{30,}"]
    for target, source in selected.items():
        if any(re.search(pattern, source.read_bytes()) for pattern in patterns):
            raise ValueError(f"Potential credential in package member: {target}")
    return verification, report


def main():
    if DESTINATION.exists() or DESTINATION.with_suffix(".zip").exists():
        raise FileExistsError("Final package already exists; inspect before any replacement")
    selected = select_files()
    verification, rendering = validate_sources(selected)
    with TemporaryDirectory(prefix="dissertation-package-", dir=DESTINATION.parent) as temporary:
        staging = Path(temporary) / DESTINATION.name
        staging.mkdir()
        for target, source in selected.items():
            output = staging / target
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
            if digest(output) != digest(source):
                raise ValueError(f"Copy mismatch: {target}")
        pins = {
            "measured_revision": REVISION,
            "test_only_publication_revision": "6ddc39274de88ed8c5a9d4ab923ca3886b069243",
            "repository": "https://github.com/KrtiT/automated-phishing-detection-public",
            "publication_branch": "praxis-realignment-v3",
            "measured_source_paths": "provenance/measured-source; exact frozen public contracts and environment pins",
            "historical_status_boundary": "Frozen docs and the historical checklist retain earlier pending statuses. Final research status is in primary-results.json, the current manuscript and COVERAGE_AND_VERIFICATION.txt.",
            "source_dependencies": "Analysis and builder scripts preserve original workspace paths and require the controlled evidence, original manuscript/template/decks and specified dependencies. They are audit source, not a standalone measurement rerun.",
            "excluded": ["original datasets", "row-level research URLs or predictions", "models and weights",
                         "private execution profiles and capabilities", "credentials"],
        }
        (staging / "provenance/CODE_AND_ENVIRONMENT.json").write_text(json.dumps(pins, indent=2) + "\n")
        report = {
            "status": "assembled_and_checked", "created_at": datetime.now(timezone.utc).isoformat(),
            "measured_revision": REVISION, "source_files": len(selected), "csv_row_counts": CSV_COUNTS,
            "primary": {key: verification[key] for key in ("retained_cells", "new_cells", "operational_cells", "operational_groups", "hypothesis_gates")},
            "measured_requests": 1243505, "measured_errors": 1901,
            "manuscript_pages": rendering["manuscript_pages"], "deck_slides": rendering["deck_slides"],
            "document_tests_passed": 44, "public_code_tests_passed": 12989, "public_code_tests_skipped": 2,
            "visual_review_record": "verification/final-delivery-qa-20261001.txt",
            "privacy_scope": "Explicit selected-file inventory; raw research inputs, predictions, weights, private capabilities and credentials excluded; common credential signatures checked.",
            "source_sha256": {name: digest(path) for name, path in sorted(selected.items())},
        }
        (staging / "package-verification.json").write_text(json.dumps(report, indent=2) + "\n")
        members = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = "".join(f"{digest(path)}  {path.relative_to(staging).as_posix()}\n" for path in members)
        (staging / "SHA256SUMS.txt").write_text(manifest)
        subprocess.run(["shasum", "-a", "256", "-c", "SHA256SUMS.txt"], cwd=staging, check=True, capture_output=True)
        archive_path = Path(temporary) / (DESTINATION.name + ".zip")
        members = sorted(path for path in staging.rglob("*") if path.is_file())
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
            for path in members:
                archive.write(path, (Path(staging.name) / path.relative_to(staging)).as_posix())
        with ZipFile(archive_path) as archive:
            expected = {(Path(staging.name) / path.relative_to(staging)).as_posix(): digest(path) for path in members}
            if archive.testzip() is not None or set(archive.namelist()) != set(expected):
                raise ValueError("Archive integrity or inventory mismatch")
            for name, expected_hash in expected.items():
                if hashlib.sha256(archive.read(name)).hexdigest() != expected_hash:
                    raise ValueError(f"Archive content mismatch: {name}")
        archive_hash = digest(archive_path)
        staging.rename(DESTINATION)
        archive_path.rename(DESTINATION.with_suffix(".zip"))
    checksum_path = DESTINATION.with_suffix(".zip.sha256")
    checksum_path.write_text(f"{archive_hash}  {DESTINATION.name}.zip\n")
    receipt = {"status": "verified", "directory": str(DESTINATION), "zip": str(DESTINATION.with_suffix(".zip")),
               "zip_sha256": archive_hash, "files_in_zip": len(members),
               "zip_bytes": DESTINATION.with_suffix(".zip").stat().st_size,
               "manifest_check": "passed", "zip_crc_inventory_and_byte_checks": "passed"}
    (REVIEW / "final-package-receipt-20261001.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
