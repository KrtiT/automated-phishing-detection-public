import csv
import json
import re
from collections import Counter
from hashlib import sha256
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dissertation" / "2026-10-04"
WORD = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def read_rows(name):
    with (PACKAGE / "aggregate-data" / name).open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def test_publication_manifest_covers_every_package_file():
    manifest = PACKAGE / "SHA256SUMS.txt"
    assert manifest.is_file(), "The publication package must have a hash manifest"
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


def test_all_primary_checks_are_decided_without_reclassifying_failures():
    rows = read_rows("primary-gates.csv")
    assert len(rows) == 22
    assert Counter(row["status"] for row in rows) == {"pass": 9, "fail": 13}
    expected = {"H1": (5, 5), "H2": (1, 3), "H3": (3, 5)}
    for hypothesis, (passed, failed) in expected.items():
        statuses = Counter(
            row["status"] for row in rows if row["hypothesis"] == hypothesis
        )
        assert statuses == {"pass": passed, "fail": failed}
    primary = json.loads((PACKAGE / "aggregate-data/primary-results.json").read_text())
    for result in primary["primary"]["hypotheses"].values():
        assert result["complete"] is True
        assert result["decision"] == "not_supported"


def test_original_and_followup_schedules_remain_separate_and_complete():
    runs = read_rows("operational-runs.csv")
    assert len(runs) == 125
    assert [int(row["cell_ordinal"]) for row in runs] == list(range(1, 126))
    assert len(read_rows("operational-groups.csv")) == 25
    assert sum(int(row["request_count"]) for row in runs) == 1243505
    assert sum(int(row["request_errors"]) for row in runs) == 1901
    assert len(read_rows("service-S/arm-metrics.csv")) == 80
    assert len(read_rows("service-S/primary-pairs.csv")) == 10
    assert Counter(
        row["passed"] for row in read_rows("service-S/requirements.csv")
    ) == {
        "True": 4,
        "False": 1,
    }
    service = json.loads(
        (PACKAGE / "aggregate-data/service-S/verification.json").read_text()
    )
    assert service["interrupted_v1_pooled"] is False
    assert service["original_hypothesis_decisions_changed"] is False
    assert service["primary"]["S_requirement_met"] is False
    detection = json.loads(
        (PACKAGE / "aggregate-data/detection-D/verification.json").read_text()
    )
    assert detection["records"] == 8622
    assert detection["D_requirement_met"] is False


def test_current_pages_distinguish_final_results_from_frozen_history():
    for name in (
        "README.md",
        "docs/research-basis.md",
        "docs/research-evidence-outline.md",
        "docs/advisor-approval/approval-status.md",
    ):
        content = (ROOT / name).read_text()
        current, historical = content.split("<!-- HISTORICAL_SNAPSHOT_BEGIN -->", 1)
        assert "H1, H2 and H3 are not supported" in current
        assert "22" in current and "25" in current and "125" in current
        assert "undecided" not in current.lower()
        assert "Historical snapshot" in current
        assert historical.rstrip().endswith("</details>")
    readme = (
        (ROOT / "README.md").read_text().split("<!-- HISTORICAL_SNAPSHOT_BEGIN -->")[0]
    )
    for term in (
        "8,622",
        "79.68%",
        "99,999",
        "799/100,000",
        "synthetic",
        "author review",
    ):
        assert term in readme


def test_public_word_file_has_no_comments_or_local_file_links():
    documents = list((PACKAGE / "manuscript").glob("*.docx"))
    assert len(documents) == 1
    with ZipFile(documents[0]) as archive:
        assert not any("comment" in name.lower() for name in archive.namelist())
        document = ElementTree.fromstring(archive.read("word/document.xml"))
        assert len(document.findall(".//w:tbl", WORD)) == 27
        assert len(document.findall(".//w:drawing", WORD)) == 4
        for name in archive.namelist():
            if name.endswith((".xml", ".rels")):
                data = archive.read(name).decode("utf-8")
                assert "file:///" not in data, name
                assert "/Users/ktallam/" not in data, name
    record = json.loads((PACKAGE / "provenance/publication-copy.json").read_text())
    assert record["visible_text_unchanged"] is True
    assert record["tables_unchanged"] is True
    assert record["figure_bytes_unchanged"] is True


def test_package_excludes_private_inputs_and_copyrighted_paper_copies():
    assert PACKAGE.is_dir()
    forbidden = {".jsonl", ".pkl", ".joblib", ".pt", ".pth", ".safetensors"}
    for path in PACKAGE.rglob("*"):
        assert not path.is_symlink()
        assert path.suffix not in forbidden, path
        assert "email_draft" not in path.name.lower()
        assert "raw" not in path.relative_to(PACKAGE).parts
    for path in (PACKAGE / "literature").rglob("*"):
        assert path.suffix != ".pdf"


def test_current_markdown_local_file_links_resolve():
    paths = [
        ROOT / "README.md",
        ROOT / "REVIEWER_GUIDE.md",
        ROOT / "research-archive/2026-10-04/README.md",
        ROOT / "research-archive/2026-10-04/PROVENANCE.md",
        ROOT / "research-archive/2026-10-04/LICENSES.md",
        *(
            PACKAGE / name
            for name in (
                "README.md",
                "RESEARCH_STORY.md",
                "REPRODUCTION.md",
                "VERIFICATION.md",
                "literature/README.md",
            )
        ),
    ]
    for path in paths:
        content = path.read_text().split("<!-- HISTORICAL_SNAPSHOT_BEGIN -->")[0]
        for target in re.findall(r"\]\(([^)]+)\)", content):
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            local = target.split("#", 1)[0].split(' "', 1)[0]
            assert (path.parent / local).exists(), (path, target)


def test_research_release_catalog_matches_plan_and_checksums():
    archive = ROOT / "research-archive/2026-10-04"
    catalog = json.loads((archive / "catalog.json").read_text())["families"]
    plan = json.loads((archive / "build-plan.json").read_text())
    hashes = dict(
        line.split("  ", 1)[::-1]
        for line in (archive / "SHA256SUMS.txt").read_text().splitlines()
    )
    assert set(catalog) == {name + ".tar.gz" for name in plan} == set(hashes)
    assert len(catalog) == 9
    statuses = Counter()
    for name, family in catalog.items():
        assert family["roots"] == plan[name.removesuffix(".tar.gz")]
        assert hashes[name] == family["sha256"]
        assert sum(family["statuses"].values()) == family["files"]
        statuses.update(family["statuses"])
    assert sum(family["files"] for family in catalog.values()) == 73234
    assert statuses == {
        "exact": 72754,
        "administrative_projection": 340,
        "hash_only": 140,
    }
    verified = {
        row["archive"]: row
        for row in map(
            json.loads,
            (archive / "archive-verification.jsonl").read_text().splitlines(),
        )
    }
    assert verified.keys() == catalog.keys()
    for name, family in catalog.items():
        assert verified[name]["files"] == family["files"]
        assert verified[name]["blobs"] == family["blobs"]
        assert verified[name]["withheld"] == family["statuses"].get("hash_only", 0)
    audit = json.loads((archive / "content-audit.json").read_text())
    assert audit["files"] == sum(statuses.values())
    assert (
        audit["catalog_sha256"]
        == sha256((archive / "catalog.json").read_bytes()).hexdigest()
    )
    assert (
        audit["builder_sha256"]
        == sha256((ROOT / "scripts/research_archive.py").read_bytes()).hexdigest()
    )
    recomputation = json.loads((archive / "recomputation.json").read_text())
    assert recomputation["status"] == "verified"
    assert recomputation["authentication"]["catalog_sha256"] == audit["catalog_sha256"]
    assert recomputation["authentication"]["inventory_sha256"] == {
        name + ".jsonl": family["inventory_sha256"] for name, family in catalog.items()
    }
    for path in (ROOT / "README.md", archive / "README.md"):
        for count in (*statuses.values(), sum(statuses.values())):
            assert f"{count:,}" in path.read_text()


def test_current_docs_link_the_retained_record_release():
    for path in (
        ROOT / "README.md",
        PACKAGE / "README.md",
        PACKAGE / "REPRODUCTION.md",
    ):
        current = path.read_text().split("<!-- HISTORICAL_SNAPSHOT_BEGIN -->")[0]
        assert "research-archive/2026-10-04/README.md" in current
        assert "Scientific datasets and row-level results are not" not in current
        assert "They are intentionally absent" not in current
