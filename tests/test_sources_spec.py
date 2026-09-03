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
        "schema_version": 1,
        "phiusiil": {
            "uci_dataset_id": 967,
            "archive_url": "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip",
            "archive_sha256": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a",
            "csv_filename": "PhiUSIIL_Phishing_URL_Dataset.csv",
            "csv_sha256": "a236549cd369cd80bd478ff8e1779cbf44c58d5c3f79f7a51a1adbed7d06d1c6",
            "license": "CC BY 4.0",
            "page_url": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
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
