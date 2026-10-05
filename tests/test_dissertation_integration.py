import json
import posixpath
import re
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dissertation/2026-10-04"
RELEASE = (
    "https://github.com/KrtiT/automated-phishing-detection-public/"
    "releases/tag/research-record-2026-10-04"
)
NAMESPACES = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def test_manuscript_cites_versioned_record_and_current_publication_scope():
    manuscript = next((PACKAGE / "manuscript").glob("Tallam*.md")).read_text()
    body, tail = manuscript.split("## References\n\n")
    references = tail.split("# Appendix A", 1)[0].strip().split("\n\n")
    assert "Tallam (2026)" in body
    assert len(references) == 59
    assert sum(RELEASE in entry for entry in references) == 1
    assert "without redistributing row-level URLs" not in manuscript
    assert "rather than redistributed processed row-level outputs" not in manuscript
    assert "Sections 1.4–1.5" in manuscript
    assert "EVIDENCE_MAP.md" in manuscript
    for phrase in ("private execution", "licensed", "retained", "release"):
        assert phrase in manuscript


def test_evidence_map_covers_all_tables_and_figures_with_existing_sources():
    mapping = json.loads((PACKAGE / "provenance/evidence-map.json").read_text())
    expected_tables = (
        ["2.1", "3.1", "4.1", "4.2a", "4.2b"]
        + [f"4.{number}" for number in range(3, 21)]
        + ["5.1", "5.2", "A.1", "B.1"]
    )
    assert [entry["id"] for entry in mapping["tables"]] == expected_tables
    assert [entry["id"] for entry in mapping["figures"]] == ["3.1", "3.2", "4.1", "4.2"]
    assert [entry["number"] for entry in mapping["slides"]] == list(range(1, 26))
    for entry in mapping["tables"] + mapping["figures"] + mapping["slides"]:
        assert entry["sources"]
        for source in entry["sources"]:
            assert (PACKAGE / source).is_file(), source


def test_appendix_evidence_locators_resolve_to_public_files():
    manuscript = next((PACKAGE / "manuscript").glob("Tallam*.md")).read_text()
    appendix = manuscript.split("# Appendix A", 1)[1].split("# Appendix B", 1)[0]
    table_rows = [line for line in appendix.splitlines() if line.startswith("| ")]
    assert len(table_rows) == 9
    for row in table_rows[1:]:
        artifacts = row.split("|")[3].strip()
        for artifact in artifacts.split("; "):
            relative = artifact.split(" (", 1)[0]
            if "/" not in relative:
                relative = "aggregate-data/" + relative
            assert (PACKAGE / relative).is_file(), relative


def test_deck_notes_follow_actual_slide_order_and_public_evidence():
    deck = next((PACKAGE / "advisor").glob("*.pptx"))
    notes = next((PACKAGE / "advisor").glob("*.txt")).read_text()
    sections = re.split(r"\nSLIDE \d+\n", "\n" + notes)[1:]
    with ZipFile(deck) as archive:
        relationships = {
            entry.get("Id"): entry.get("Target")
            for entry in ElementTree.fromstring(
                archive.read("ppt/_rels/presentation.xml.rels")
            )
        }
        slides = ElementTree.fromstring(archive.read("ppt/presentation.xml")).findall(
            ".//p:sldId", NAMESPACES
        )
        assert len(slides) == len(sections) == 25
        for slide, section in zip(slides, sections):
            slide_path = (
                "ppt/" + relationships[slide.get("{" + NAMESPACES["r"] + "}id")]
            )
            relations_path = posixpath.join(
                posixpath.dirname(slide_path),
                "_rels",
                posixpath.basename(slide_path) + ".rels",
            )
            relation = next(
                entry
                for entry in ElementTree.fromstring(archive.read(relations_path))
                if entry.get("Type").endswith("/notesSlide")
            )
            note_path = posixpath.normpath(
                posixpath.join(posixpath.dirname(slide_path), relation.get("Target"))
            )
            note = ElementTree.fromstring(archive.read(note_path))
            body = next(
                shape
                for shape in note.findall(".//p:sp", NAMESPACES)
                if any(
                    placeholder.get("type") == "body"
                    for placeholder in shape.findall(".//p:ph", NAMESPACES)
                )
            )
            paragraphs = [
                "".join(element.itertext())
                for element in body.findall(".//a:t", NAMESPACES)
            ]
            assert section.strip() == "\n".join(paragraphs).strip()
            assert RELEASE in section
            assert "final-evidence-20261001/" not in section
            assert "verified-service-v2/" not in section
            assert "followup-20261001/verified-detection-v1/" not in section
    readme = (PACKAGE / "README.md").read_text()
    assert "25 slides" in readme and "23 slides" not in readme


def test_current_package_links_resolve_after_editorial_revision():
    for relative in (
        "README.md",
        "REVIEWER_GUIDE.md",
        "dissertation/2026-10-04/EVIDENCE_MAP.md",
    ):
        page = ROOT / relative
        text = page.read_text().split("<!-- HISTORICAL_SNAPSHOT_BEGIN -->")[0]
        for target in re.findall(r"\]\(([^)]+)\)", text):
            if not re.match(r"(?:https?://|mailto:|#)", target):
                assert (page.parent / target.split("#", 1)[0]).exists(), target
