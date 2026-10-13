"""Check the October 12 reading edition without opening private research rows."""

import csv
import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dissertation/2026-10-12"
WORD = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def data(name):
    return json.loads((PACKAGE / name).read_text())


def rows(name):
    with (PACKAGE / "aggregate-data" / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_revision_manifest_covers_every_file():
    manifest = PACKAGE / "SHA256SUMS.txt"
    assert manifest.is_file()
    entries = dict(
        line.split("  ", 1)[::-1] for line in manifest.read_text().splitlines()
    )
    files = {
        path.relative_to(PACKAGE).as_posix()
        for path in PACKAGE.rglob("*")
        if path.is_file() and path != manifest
    }
    assert files == set(entries)
    for name, digest in entries.items():
        assert sha256((PACKAGE / name).read_bytes()).hexdigest() == digest, name


def test_correction_preserves_all_original_requirements_and_changed_decision():
    ledger = rows("corrected-primary-gates.csv")
    assert len(ledger) == 22
    assert Counter(row["original_status"] for row in ledger) == {"pass": 9, "fail": 13}
    assert Counter(row["corrected_status"] for row in ledger)["pass"] == 8
    assert all(
        row["corrected_status"] in {"pass", "fail", "not_established"} for row in ledger
    )
    assert all(
        row["original_threshold"] == row["corrected_threshold"] for row in ledger
    )
    assert {
        hypothesis: sum(
            row["corrected_status"] == "pass"
            for row in ledger
            if row["hypothesis"] == hypothesis
        )
        for hypothesis in ("H1", "H2", "H3")
    } == {"H1": 5, "H2": 1, "H3": 2}
    changed = [
        row
        for row in ledger
        if row["original_status"] == "pass" and row["corrected_status"] != "pass"
    ]
    assert len(changed) == 1
    assert changed[0]["name"] == "gold.cascade_minus_transformer"
    assert float(changed[0]["corrected_lower"]) < -0.02


def test_seven_responses_are_verified_without_claiming_advisor_acceptance():
    with (PACKAGE / "RESPONSE_MATRIX.csv").open(newline="") as stream:
        responses = list(csv.DictReader(stream))
    assert [int(row["comment"]) for row in responses] == list(range(1, 8))
    assert {row["state"] for row in responses} == {
        "evidence_integrated_package_verified"
    }
    assert all(row["manuscript"] for row in responses)
    assert "not advisor acceptance" in (PACKAGE / "VERIFICATION.txt").read_text()


def test_frozen_snapshot_matches_profile_hashes_without_private_locators():
    freeze = data("provenance/freeze-projection.json")
    assert freeze["bound_files_checked"] == len(freeze["bindings"]) == 506
    public = [item for item in freeze["bindings"] if "public_path" in item]
    assert len(public) == 35
    assert len({item["binding_id"] for item in freeze["bindings"]}) == 506
    for item in public:
        assert (
            sha256((PACKAGE / item["public_path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )
    assert "executable" not in freeze["runtime"]
    assert freeze["projection_is_execution_authorization"] is False
    assert not freeze["raw_revision_records_published"]


def test_document_evidence_and_preservation_are_explicit():
    document = PACKAGE / "manuscript/Tallam_Krti_Praxis_Revision_2026-10-12.docx"
    with ZipFile(document) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
        assert len(root.findall(".//w:tbl", WORD)) == 31
        assert len(root.findall(".//w:drawing", WORD)) == 4
        text = " ".join(node.text or "" for node in root.findall(".//w:t", WORD))
        assert "Cross-Dataset Reliability and Computational Trade-offs" in text
        assert "Amir Etemadi" in text and "Mazen Mheish" in text
        assert not root.findall(".//w:ins", WORD) and not root.findall(".//w:del", WORD)
    render = data("document-checks/manuscript-render.json")
    assert render["status"] == "verified"
    assert render["pdf_pages"] == 150
    assert render["navigation_entries"] == 127
    assert (
        render["detached_table_captions"] == render["figure_prose_intersections"] == 0
    )
    preservation = data("provenance/preservation.json")
    assert (
        preservation["submitted_original_sha256"]
        == "b0dad1835252ba6a1a95d29e5eb2695f9437a868b4ee7cb4f62a991c3114f7b5"
    )
    assert preservation["submitted_original_unchanged"] is True
    with ZipFile(
        PACKAGE / "advisor/Tallam_Advisor_Revision_2026-10-12.pptx"
    ) as archive:
        assert (
            sum(
                bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name))
                for name in archive.namelist()
            )
            == 16
        )


def test_all_cost_arms_and_control_seeds_are_retained():
    arms = rows("C-all-arms.csv")
    assert len(arms) == 30
    assert Counter(row["path"] for row in arms) == dict.fromkeys(
        [
            "structural",
            "original",
            "symmetric",
            "directional",
            "symmetric_control",
            "transformer",
        ],
        5,
    )
    assert sum(int(row["requests_completed"]) for row in arms) == 30000
    assert sum(int(row["warmup_requests_excluded"]) for row in arms) == 3000
    assert sum(int(row["requests_failed"]) for row in arms) == 0
    controls = rows("P-all-seeds.csv")
    assert [int(row["seed"]) for row in controls] == list(range(42, 47))
    assert all(row["validation_labels_select_threshold"] == "True" for row in controls)
    assert all(
        int(row["positive"]) == 12486 and int(row["negative"]) == 20209
        for row in controls
    )
    assert all(0 < int(row["iterations"]) < 5000 for row in controls)


def test_routing_and_formatting_keep_adverse_outcomes():
    routing = {row["system"]: row for row in rows("R-routing.csv")}
    assert all(int(row["true_positive"]) == 1392 for row in routing.values())
    assert int(routing["structural"]["false_positive"]) == 1028
    assert int(routing["symmetric"]["false_positive"]) == 1046
    assert int(routing["symmetric"]["logical_calls"]) == 21
    formatting = data("aggregate-data/F-formatting.json")
    assert formatting["gold"]["https"]["true_positive"] == 22
    assert formatting["certified"]["https"]["false_positive"] == 50
    assert formatting["gold"]["original"]["true_positive"] == 69
    assert formatting["certified"]["original"]["false_positive"] == 2261


def test_public_projection_contains_no_record_files_or_personal_locators():
    forbidden = re.compile(
        r"/Users/|/private/var/|/var/folders/|file:///|mailto:|@gmail\.com|@gwu\.edu"
    )
    for path in PACKAGE.rglob("*"):
        assert not path.is_symlink(), path
        if not path.is_file():
            continue
        assert path.suffix not in {
            ".jsonl",
            ".pkl",
            ".joblib",
            ".pt",
            ".pth",
            ".safetensors",
        }, path
        if path.suffix in {".docx", ".pptx"}:
            with ZipFile(path) as archive:
                assert not any("comment" in name.lower() for name in archive.namelist())
                for name in archive.namelist():
                    if name.endswith((".xml", ".rels")):
                        assert not forbidden.search(archive.read(name).decode()), (
                            path,
                            name,
                        )
        elif path.suffix in {".md", ".txt", ".json", ".csv", ".py"}:
            assert not forbidden.search(path.read_text()), path


def test_current_reading_route_and_local_links_resolve():
    for path in [ROOT / "README.md", ROOT / "REVIEWER_GUIDE.md", *PACKAGE.glob("*.md")]:
        content = path.read_text().split("<!-- HISTORICAL_SNAPSHOT_BEGIN -->")[0]
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            assert (path.parent / target.split("#", 1)[0]).is_file(), (path, target)
    assert "dissertation/2026-10-12" in (ROOT / "README.md").read_text()
