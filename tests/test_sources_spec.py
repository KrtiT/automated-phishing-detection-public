import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_official_development_sources_are_exactly_pinned():
    source_path = ROOT / "data" / "sources.json"
    assert source_path.is_file()
    sources = json.loads(source_path.read_text(encoding="utf-8"))

    assert sources == {
        "contract_id": "phiusiil-development-v1",
        "schema_version": 2,
        "phiusiil": {
            "uci_dataset_id": 967,
            "paper_doi": "10.1016/j.cose.2023.103545",
            "archive_url": "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip",
            "archive_sha256": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a",
            "csv_filename": "PhiUSIIL_Phishing_URL_Dataset.csv",
            "csv_sha256": "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6",
            "license": "CC BY 4.0",
            "page_url": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
            "native_label_semantics": {
                "0": "phishing",
                "1": "legitimate",
            },
            "publisher_reported_class_sources": {
                "legitimate": ["Open PageRank"],
                "phishing": ["PhishTank", "OpenPhish", "MalwareWorld"],
            },
            "publisher_reported_phishing_retrieval_window": {
                "start": "2022-10-01",
                "end": "2023-05-21",
            },
            "publisher_reported_legitimate_collection_window": None,
            "reference_classification_basis": "publisher-provided/source-derived",
            "public_per_row_provenance_available": {
                "source": False,
                "timestamp": False,
                "snapshot": False,
                "independent_adjudication": False,
            },
        },
        "public_suffix_list": {
            "url": "https://raw.githubusercontent.com/publicsuffix/list/0f1fa47ec45056a19c2fdcd32a08442de9715d12/public_suffix_list.dat",
            "upstream_url": "https://publicsuffix.org/list/public_suffix_list.dat",
            "version": "commit-pinned snapshot",
            "commit": "0f1fa47ec45056a19c2fdcd32a08442de9715d12",
            "sha256": "65365c4c9a4a6f746d53aadc758ab6b08aa10bb1379fea8ac353e381bca4b62e",
            "license": "MPL-2.0",
        },
    }
    assert "phishvn" not in source_path.read_text(encoding="utf-8").lower()


def test_published_preparation_summary_uses_pinned_sources():
    source_path = ROOT / "data" / "sources.json"
    source_bytes = source_path.read_bytes()
    summary = json.loads(
        (ROOT / "reports" / "phiusiil-preparation-summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert summary["declared_sources"] == json.loads(source_bytes)
    assert summary["source_spec_sha256"] == sha256(source_bytes).hexdigest()
    assert summary["label_mapping"] == {
        "version": "phiusiil-native-label-map-v1",
        "native_label_meanings": {"0": "phishing", "1": "legitimate"},
        "native_to_is_phishing": {"0": 1, "1": 0},
        "is_phishing_meanings": {"0": "legitimate", "1": "phishing"},
    }


def test_research_basis_records_sources_threshold_ownership_and_prior_work():
    research_basis = ROOT / "docs" / "research-basis.md"
    assert research_basis.is_file()
    text = research_basis.read_text(encoding="utf-8")

    for heading in (
        "## Source basis and limitations",
        "## Study-defined decision gates",
        "## Closest prior work and contribution boundary",
    ):
        assert heading in text

    for primary_source in (
        "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
        "https://doi.org/10.1016/j.cose.2023.103545",
        "https://doi.org/10.3389/fcomp.2026.1834407",
        "https://www.sciencedirect.com/science/article/pii/S0957417426029957",
        "https://doi.org/10.3390/electronics15143051",
        "https://doi.org/10.1016/j.dib.2026.113195",
        "https://doi.org/10.1016/j.comnet.2024.110398",
        "https://doi.org/10.1609/aaai.v38i19.30161",
        "https://aclanthology.org/2021.findings-emnlp.43/",
        "https://www.usenix.org/conference/usenixsecurity21/presentation/yang-limin",
    ):
        assert primary_source in text

    assert "Every gate below is study-defined" in text
    assert "not literature-prescribed" in text
    assert "not an achieved result" in text
    assert "native `0` -> `is_phishing=1`; native `1` -> `is_phishing=0`" in text
