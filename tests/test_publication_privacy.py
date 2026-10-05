"""Privacy checks on public derivatives, without changing scientific identities."""

import json
import re
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dissertation/2026-10-04"
WORD = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
LOCAL_PATH = re.compile(r"/(?:Users|private/var|var/folders)/[^\s\"']+")


def test_public_reports_do_not_expose_personal_machine_paths():
    affected = [
        path.name
        for path in (ROOT / "reports").glob("*.json")
        if LOCAL_PATH.search(path.read_text())
    ]
    assert not affected, affected


def test_author_degree_history_is_confined_to_manuscript_front_matter():
    document = next((PACKAGE / "manuscript").glob("Tallam*.docx"))
    with ZipFile(document) as archive:
        tree = ElementTree.fromstring(archive.read("word/document.xml"))
    paragraphs = tree.findall("./w:body/w:p", WORD)
    degree_lines = " ".join(paragraphs[2].itertext())
    institutions = set(
        re.findall(r"(?:University of [A-Za-z]+|[A-Za-z]+ University)", degree_lines)
    )
    assert institutions, "The supplied GWU template includes prior-degree lines"
    for path in (PACKAGE / "analysis-scripts").rglob("*.py"):
        assert not any(name in path.read_text() for name in institutions), path
    for path in (PACKAGE / "advisor").iterdir():
        if path.suffix == ".txt":
            assert not any(name in path.read_text() for name in institutions), path
        elif path.suffix == ".pptx":
            with ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith((".xml", ".rels")):
                        assert not any(
                            institution in archive.read(name).decode()
                            for institution in institutions
                        ), (path, name)


def test_public_privacy_receipt_authenticates_each_derivative():
    receipt_path = ROOT / "privacy/2026-10-04/publication.json"
    assert receipt_path.is_file(), "Disclose privacy derivatives separately"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["base_commit"] == "4d6c78883d9bc39dec397ab709064952ba726cc3"
    assert receipt["scientific_results_changed"] is False
    assert receipt["frozen_execution_rules_changed"] is False
    assert receipt["history_rewritten"] is False
    reports = receipt["report_projections"]
    assert len(reports) == 3
    for name, record in reports.items():
        content = (ROOT / name).read_bytes()
        assert sha256(content).hexdigest() == record["public_sha256"]
        assert record["original_sha256"] != record["public_sha256"]
        assert record["changed_json_pointers"]
        assert record["unchanged_scientific_values"] is True
        assert not LOCAL_PATH.search(json.dumps(record))


def test_current_edition_is_bound_by_privacy_record_not_reissued_old_receipt():
    receipt_path = ROOT / "privacy/2026-10-04/publication.json"
    assert receipt_path.is_file(), "A new edition needs a new identity record"
    receipt = json.loads(receipt_path.read_text())
    for name, digest in receipt["document_outputs"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    assert (ROOT / "privacy/README.md").is_file()


def test_frozen_execution_pins_are_not_replaced_with_public_copy_hashes():
    from automated_phishing_detection.development_correction import HISTORY_PINS

    receipt = json.loads((ROOT / "privacy/2026-10-04/publication.json").read_text())
    for version in (1, 2, 3):
        contract = json.loads(
            (ROOT / f"data/execution-binding-contract-v{version}.json").read_text()
        )
        pins = contract["public_file_sha256"]
        for name, projection in receipt["report_projections"].items():
            if name in pins:
                assert pins[name] == projection["original_sha256"]
                assert pins[name] != projection["public_sha256"]
    for name, pin in HISTORY_PINS.items():
        projection = receipt["report_projections"].get(name)
        if projection:
            assert pin == projection["original_sha256"]
            assert pin != projection["public_sha256"]


def test_office_files_do_not_retain_personal_editing_metadata():
    private_fields = {
        "lastModifiedBy",
        "created",
        "modified",
        "revision",
        "TotalTime",
        "Company",
        "Manager",
        "Template",
    }
    for path in PACKAGE.rglob("*"):
        if path.suffix not in {".docx", ".pptx"}:
            continue
        with ZipFile(path) as archive:
            for name in ("docProps/core.xml", "docProps/app.xml"):
                tree = ElementTree.fromstring(archive.read(name))
                assert not any(
                    child.tag.rsplit("}", 1)[-1] in private_fields for child in tree
                ), (path, name)
