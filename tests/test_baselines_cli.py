import json
import os
import stat
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from automated_phishing_detection import baselines

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "data" / "rq1-baseline-contract.json"
OFFICIAL_TRAIN_SHA256 = (
    "575f2fb13a0766020e29d78bf8e633a185b381abde7060bdd1ed04cc4a5e38a0"
)
OFFICIAL_VALIDATION_SHA256 = (
    "970c6568a6400a1fc265b7809ef7bd9d1c297632799cbb88801313bc34ac415a"
)
OFFICIAL_SUMMARY_SHA256 = (
    "1a85a7eecc0f5baa7c59e03a0cbde63fd4595409feb918dc5ff916ead7cd5c9e"
)
OFFICIAL_CONTRACT_SHA256 = (
    "594a66769dee3bf23c4133020dcf9b7d57c105590e5007832ac4249def6a33d4"
)
SOURCE_CSV_SHA256 = "0" * 64


def _jsonl_bytes(records):
    return b"".join(
        (
            json.dumps(record, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
            + "\n"
        ).encode("ascii")
        for record in records
    )


def _records(split, *, negative_count, positive_count, ordinal_start):
    records = []
    partition_marker = {"train": "tr", "validation": "va"}[split]
    for offset in range(negative_count + positive_count):
        ordinal = ordinal_start + offset
        is_phishing = int(offset >= negative_count)
        marker = "phish" if is_phishing else "safe"
        records.append(
            {
                "canonical_url_sha256": f"{ordinal:064x}",
                "is_phishing": is_phishing,
                "raw_url": (
                    f"https://{marker}-{ordinal:04d}."
                    f"{partition_marker}-{ordinal:04d}.example/path/{marker}"
                ),
                "record_id": (f"phiusiil-row-v1:{SOURCE_CSV_SHA256}:{ordinal:016x}"),
                "registrable_domain": f"{partition_marker}-{ordinal:04d}.example",
                "split": split,
            }
        )
    return records


def _preparation_summary(train_bytes, validation_bytes, train_rows, validation_rows):
    split_counts = {
        "train": {
            "class_counts": {
                "0": sum(row["is_phishing"] == 0 for row in train_rows),
                "1": sum(row["is_phishing"] == 1 for row in train_rows),
            },
            "domain_count": len(train_rows),
            "row_count": len(train_rows),
        },
        "validation": {
            "class_counts": {
                "0": sum(row["is_phishing"] == 0 for row in validation_rows),
                "1": sum(row["is_phishing"] == 1 for row in validation_rows),
            },
            "domain_count": len(validation_rows),
            "row_count": len(validation_rows),
        },
        "group_test": {
            "class_counts": {"0": 1, "1": 1},
            "domain_count": 2,
            "row_count": 2,
        },
    }
    return {
        "algorithms": {
            "allocation_basis": "unique_ascii_domain_groups",
            "allocation_version": "hamilton-largest-remainder-v1",
            "canonicalization_version": "canonical-url-v1",
            "domain_split_version": "phiusiil-domain-split-v1",
            "record_identifier_version": "phiusiil-row-v1",
            "seed": "20260816",
            "split_percentages": {"group_test": 15, "train": 70, "validation": 15},
        },
        "declared_sources": {
            "contract_id": "phiusiil-development-v1",
            "phiusiil": {
                "archive_sha256": "1" * 64,
                "archive_url": "https://example.invalid/archive.zip",
                "csv_filename": "fixture.csv",
                "csv_sha256": SOURCE_CSV_SHA256,
                "license": "CC BY 4.0",
                "native_label_semantics": {"0": "phishing", "1": "legitimate"},
                "page_url": "https://example.invalid/dataset",
                "paper_doi": "10.0000/fixture",
                "public_per_row_provenance_available": {
                    "independent_adjudication": False,
                    "snapshot": False,
                    "source": False,
                    "timestamp": False,
                },
                "publisher_reported_class_sources": {
                    "legitimate": ["fixture legitimate"],
                    "phishing": ["fixture phishing"],
                },
                "publisher_reported_legitimate_collection_window": None,
                "publisher_reported_phishing_retrieval_window": {
                    "end": "2023-05-21",
                    "start": "2022-10-01",
                },
                "reference_classification_basis": "publisher-provided/source-derived",
                "uci_dataset_id": 967,
            },
            "public_suffix_list": {
                "commit": "2" * 40,
                "license": "MPL-2.0",
                "sha256": "3" * 64,
                "upstream_url": "https://example.invalid/psl",
                "url": "https://example.invalid/psl-at-commit",
                "version": "commit-pinned snapshot",
            },
            "schema_version": 2,
        },
        "label_mapping": {
            "is_phishing_meanings": {"0": "legitimate", "1": "phishing"},
            "native_label_meanings": {"0": "phishing", "1": "legitimate"},
            "native_to_is_phishing": {"0": 1, "1": 0},
            "version": "phiusiil-native-label-map-v1",
        },
        "local_label_counts": {
            "0": split_counts["train"]["class_counts"]["0"]
            + split_counts["validation"]["class_counts"]["0"]
            + 1,
            "1": split_counts["train"]["class_counts"]["1"]
            + split_counts["validation"]["class_counts"]["1"]
            + 1,
        },
        "native_label_counts": {"0": 1, "1": 1, "invalid": 0},
        "output_hashes": {
            "SHA256SUMS": "4" * 64,
            "group_test.jsonl": "5" * 64,
            "quarantine.jsonl": "6" * 64,
            "train.jsonl": sha256(train_bytes).hexdigest(),
            "validation.jsonl": sha256(validation_bytes).hexdigest(),
        },
        "overall_counts": {
            "canonicalized_url_groups": len(train_rows) + len(validation_rows) + 2,
            "input_rows": len(train_rows) + len(validation_rows) + 2,
            "quarantined_rows": 0,
            "retained_domains": len(train_rows) + len(validation_rows) + 2,
            "retained_rows": len(train_rows) + len(validation_rows) + 2,
        },
        "quarantine_reason_counts": {
            "canonical_url_conflicting_mapping": 0,
            "canonical_url_duplicate_same_mapping": 0,
            "invalid_or_missing_url": 0,
            "invalid_phiusiil_native_label": 0,
        },
        "schema_version": 1,
        "source_spec_sha256": "7" * 64,
        "splits": split_counts,
    }


def _write_fixture(tmp_path, *, mutate_summary=None, mutate_train=None):
    train_rows = _records(
        "train", negative_count=24, positive_count=24, ordinal_start=1
    )
    validation_rows = _records(
        "validation", negative_count=400, positive_count=20, ordinal_start=1000
    )
    if mutate_train is not None:
        mutate_train(train_rows)
    train_bytes = _jsonl_bytes(train_rows)
    validation_bytes = _jsonl_bytes(validation_rows)
    summary = _preparation_summary(
        train_bytes, validation_bytes, train_rows, validation_rows
    )
    if mutate_summary is not None:
        mutate_summary(summary)
    summary_bytes = (
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")

    train_path = tmp_path / "train.jsonl"
    validation_path = tmp_path / "validation.jsonl"
    prep_summary_path = tmp_path / "preparation-summary.json"
    contract_path = tmp_path / "rq1-baseline-contract.json"
    train_path.write_bytes(train_bytes)
    validation_path.write_bytes(validation_bytes)
    prep_summary_path.write_bytes(summary_bytes)
    contract_path.write_bytes(CONTRACT.read_bytes())
    return {
        "train": train_path,
        "validation": validation_path,
        "preparation_summary": prep_summary_path,
        "contract": contract_path,
        "output_dir": tmp_path / "models",
        "summary": tmp_path / "baseline-summary.json",
        "policy": baselines._InputHashPolicy(
            train_sha256=sha256(train_bytes).hexdigest(),
            validation_sha256=sha256(validation_bytes).hexdigest(),
            preparation_summary_sha256=sha256(summary_bytes).hexdigest(),
            contract_sha256=sha256(CONTRACT.read_bytes()).hexdigest(),
        ),
    }


def _run_fixture(paths):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_private-fixture-fit",
            *(
                str(paths[key])
                for key in (
                    "train",
                    "validation",
                    "preparation_summary",
                    "contract",
                    "output_dir",
                    "summary",
                )
            ),
            paths["policy"].train_sha256,
            paths["policy"].validation_sha256,
            paths["policy"].preparation_summary_sha256,
            paths["policy"].contract_sha256,
        ],
        capture_output=True,
        check=False,
        env=environment,
    )


def _refresh_summary_hashes(paths):
    summary = json.loads(paths["preparation_summary"].read_bytes())
    summary["output_hashes"]["train.jsonl"] = sha256(
        paths["train"].read_bytes()
    ).hexdigest()
    summary["output_hashes"]["validation.jsonl"] = sha256(
        paths["validation"].read_bytes()
    ).hexdigest()
    summary_bytes = (
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    paths["preparation_summary"].write_bytes(summary_bytes)
    paths["policy"] = baselines._InputHashPolicy(
        train_sha256=sha256(paths["train"].read_bytes()).hexdigest(),
        validation_sha256=sha256(paths["validation"].read_bytes()).hexdigest(),
        preparation_summary_sha256=sha256(summary_bytes).hexdigest(),
        contract_sha256=paths["policy"].contract_sha256,
    )


def _private_fixture_main(arguments):
    if len(arguments) != 11 or arguments[0] != "_private-fixture-fit":
        raise AssertionError("tests-only fixture invocation is invalid")
    paths = list(map(Path, arguments[1:7]))
    policy = baselines._InputHashPolicy(*arguments[7:11])
    try:
        result = baselines._fit_baselines(
            train_path=paths[0],
            validation_path=paths[1],
            preparation_summary_path=paths[2],
            contract_path=paths[3],
            output_dir=paths[4],
            summary_path=paths[5],
            _input_hash_policy=policy,
        )
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        baselines.BaselineError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def _successful_run(tmp_path, suffix=""):
    run_dir = tmp_path / f"run{suffix}"
    run_dir.mkdir()
    paths = _write_fixture(run_dir)
    completed = _run_fixture(paths)
    return completed, paths


def _assert_no_sensitive_keys(value):
    forbidden = {
        "raw_url",
        "canonical_url_sha256",
        "registrable_domain",
        "record_id",
        "row_score",
    }
    if isinstance(value, dict):
        assert not (set(value) & forbidden)
        for child in value.values():
            _assert_no_sensitive_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_sensitive_keys(child)


def test_fixture_fit_writes_portable_private_models_and_public_summary(tmp_path):
    completed, paths = _successful_run(tmp_path)

    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stderr == b""
    assert stat.S_IMODE(paths["output_dir"].stat().st_mode) == 0o700
    assert {path.name for path in paths["output_dir"].iterdir()} == {
        "length-only.json",
        "logistic-l1.json",
        "SHA256SUMS",
    }
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in paths["output_dir"].iterdir()
    )
    assert stat.S_IMODE(paths["summary"].stat().st_mode) == 0o644

    summary = json.loads(paths["summary"].read_bytes())
    assert json.loads(completed.stdout) == summary
    assert summary["analysis_stage"] == "development_validation_only"
    assert summary["hypothesis_status"] == {
        "H1": "undecided",
        "H2": "undecided",
        "H3": "undecided",
    }
    assert summary["access"] == {
        "group_test_accessed": False,
        "phishvn_accessed": False,
    }
    assert summary["input_counts"] == {
        "train": {"0": 24, "1": 24, "rows": 48},
        "validation": {"0": 400, "1": 20, "rows": 420},
    }
    assert summary["input_hashes"] == {
        "contract": paths["policy"].contract_sha256,
        "preparation_summary": paths["policy"].preparation_summary_sha256,
        "train": paths["policy"].train_sha256,
        "validation": paths["policy"].validation_sha256,
    }
    assert set(summary["models"]) == {"Logistic-L1", "length-only"}
    assert summary["models"]["length-only"]["feature_count"] == 1
    assert summary["models"]["Logistic-L1"]["feature_count"] == 25
    _assert_no_sensitive_keys(summary)
    assert str(tmp_path) not in paths["summary"].read_text(encoding="ascii")
    assert "tr-0001.example" not in paths["summary"].read_text(encoding="ascii")

    checksums = {}
    for line in (paths["output_dir"] / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        checksums[filename] = digest
    assert set(checksums) == {"length-only.json", "logistic-l1.json"}
    for filename, digest in checksums.items():
        artifact_path = paths["output_dir"] / filename
        assert sha256(artifact_path.read_bytes()).hexdigest() == digest
        artifact = json.loads(artifact_path.read_bytes())
        assert artifact["artifact_type"] == "rq1-baseline-model"
        assert artifact["analysis_stage"] == "development_validation_only"
        assert artifact["classes"] == [0, 1]
        assert artifact["validation_threshold"]["status"] in {
            "selected",
            "target_not_met",
        }
        assert "scaler" in artifact and "classifier" in artifact
        _assert_no_sensitive_keys(artifact)


def test_repeated_fixture_runs_are_byte_identical(tmp_path):
    first, first_paths = _successful_run(tmp_path, "-first")
    second, second_paths = _successful_run(tmp_path, "-second")

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert first_paths["summary"].read_bytes() == second_paths["summary"].read_bytes()
    for filename in ("length-only.json", "logistic-l1.json", "SHA256SUMS"):
        assert (first_paths["output_dir"] / filename).read_bytes() == (
            second_paths["output_dir"] / filename
        ).read_bytes()


def test_production_cli_hard_pins_each_input_before_parsing(tmp_path):
    assert baselines._OFFICIAL_INPUT_HASH_POLICY == baselines._InputHashPolicy(
        train_sha256=OFFICIAL_TRAIN_SHA256,
        validation_sha256=OFFICIAL_VALIDATION_SHA256,
        preparation_summary_sha256=OFFICIAL_SUMMARY_SHA256,
        contract_sha256=OFFICIAL_CONTRACT_SHA256,
    )
    paths = _write_fixture(tmp_path)
    paths["train"].write_bytes(b"not parseable")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "automated_phishing_detection.cli",
            "fit-baselines",
            "--train",
            str(paths["train"]),
            "--validation",
            str(paths["validation"]),
            "--preparation-summary",
            str(paths["preparation_summary"]),
            "--contract",
            str(paths["contract"]),
            "--output-dir",
            str(paths["output_dir"]),
            "--summary",
            str(paths["summary"]),
        ],
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert f"expected {OFFICIAL_TRAIN_SHA256}".encode() in completed.stderr
    assert b"JSON" not in completed.stderr
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


@pytest.mark.parametrize(
    "mutation",
    (
        lambda rows: rows[0].__setitem__("extra", "forbidden"),
        lambda rows: rows[0].__setitem__("is_phishing", True),
        lambda rows: rows[0].__setitem__("raw_url", 7),
        lambda rows: rows.__setitem__(slice(0, 2), list(reversed(rows[:2]))),
        lambda rows: rows[1].__setitem__("record_id", rows[0]["record_id"]),
        lambda rows: rows[0].__setitem__("split", "validation"),
    ),
)
def test_invalid_train_schema_order_and_types_leave_no_artifacts(tmp_path, mutation):
    paths = _write_fixture(tmp_path, mutate_train=mutation)
    completed = _run_fixture(paths)

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


def test_duplicate_json_keys_are_rejected(tmp_path):
    paths = _write_fixture(tmp_path)
    original = paths["train"].read_bytes()
    paths["train"].write_bytes(original.replace(b"{", b'{"split":"train",', 1))
    paths["policy"] = baselines._InputHashPolicy(
        train_sha256=sha256(paths["train"].read_bytes()).hexdigest(),
        validation_sha256=paths["policy"].validation_sha256,
        preparation_summary_sha256=paths["policy"].preparation_summary_sha256,
        contract_sha256=paths["policy"].contract_sha256,
    )
    _refresh_summary_hashes(paths)
    completed = _run_fixture(paths)

    assert completed.returncode == 2
    assert b"duplicate JSON key" in completed.stderr
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


def test_declared_count_mismatch_is_an_error(tmp_path):
    def mutate(summary):
        summary["splits"]["train"]["row_count"] += 1

    paths = _write_fixture(tmp_path, mutate_summary=mutate)
    completed = _run_fixture(paths)

    assert completed.returncode == 2
    assert b"declared" in completed.stderr
    assert not paths["output_dir"].exists()
    assert not paths["summary"].exists()


def test_train_validation_domains_must_be_disjoint(tmp_path):
    paths = _write_fixture(tmp_path)
    validation_lines = paths["validation"].read_text().splitlines()
    first = json.loads(validation_lines[0])
    first["registrable_domain"] = "tr-0001.example"
    validation_lines[0] = json.dumps(first, separators=(",", ":"), sort_keys=True)
    paths["validation"].write_text("\n".join(validation_lines) + "\n")
    paths["policy"] = baselines._InputHashPolicy(
        train_sha256=paths["policy"].train_sha256,
        validation_sha256=sha256(paths["validation"].read_bytes()).hexdigest(),
        preparation_summary_sha256=paths["policy"].preparation_summary_sha256,
        contract_sha256=paths["policy"].contract_sha256,
    )
    _refresh_summary_hashes(paths)
    completed = _run_fixture(paths)

    assert completed.returncode == 2
    assert b"registrable domain crosses train and validation" in completed.stderr
    assert not paths["output_dir"].exists()


def test_existing_or_aliased_destinations_are_not_replaced(tmp_path):
    paths = _write_fixture(tmp_path)
    paths["summary"].write_text("keep\n")

    completed = _run_fixture(paths)

    assert completed.returncode == 2
    assert paths["summary"].read_text() == "keep\n"
    assert not paths["output_dir"].exists()


def test_summary_publish_collision_preserves_published_output_and_competitor_summary(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path)
    original_publish = baselines._publish_path_without_replace

    def collide_on_summary(source, destination):
        if destination == paths["summary"]:
            paths["summary"].write_text("competitor summary\n")
        original_publish(source, destination)

    monkeypatch.setattr(baselines, "_publish_path_without_replace", collide_on_summary)

    with pytest.raises(baselines.BaselineError, match="already exists"):
        baselines._fit_baselines(
            train_path=paths["train"],
            validation_path=paths["validation"],
            preparation_summary_path=paths["preparation_summary"],
            contract_path=paths["contract"],
            output_dir=paths["output_dir"],
            summary_path=paths["summary"],
            _input_hash_policy=paths["policy"],
        )

    assert paths["summary"].read_text() == "competitor summary\n"
    assert {path.name for path in paths["output_dir"].iterdir()} == {
        "length-only.json",
        "logistic-l1.json",
        "SHA256SUMS",
    }


def test_summary_interruption_cannot_delete_competitor_swapped_output(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path)
    original_publish = baselines._publish_path_without_replace
    preserved_output = tmp_path / "preserved-published-output"
    sentinel = paths["output_dir"] / "competitor.txt"

    def interrupt_after_swapping_output(source, destination):
        if destination == paths["summary"]:
            os.rename(paths["output_dir"], preserved_output)
            paths["output_dir"].mkdir()
            sentinel.write_text("competitor output\n")
            raise RuntimeError("simulated summary interruption")
        original_publish(source, destination)

    monkeypatch.setattr(
        baselines,
        "_publish_path_without_replace",
        interrupt_after_swapping_output,
    )

    with pytest.raises(RuntimeError, match="simulated summary interruption"):
        baselines._fit_baselines(
            train_path=paths["train"],
            validation_path=paths["validation"],
            preparation_summary_path=paths["preparation_summary"],
            contract_path=paths["contract"],
            output_dir=paths["output_dir"],
            summary_path=paths["summary"],
            _input_hash_policy=paths["policy"],
        )

    assert sentinel.read_text() == "competitor output\n"
    assert {path.name for path in preserved_output.iterdir()} == {
        "length-only.json",
        "logistic-l1.json",
        "SHA256SUMS",
    }
    assert not paths["summary"].exists()


def test_open_verified_descriptor_is_used_if_input_path_is_replaced(
    tmp_path, monkeypatch
):
    paths = _write_fixture(tmp_path)
    original_hash = paths["policy"].train_sha256
    replacement_url = "https://replacement.tr-0001.example/path/safe"
    observed_urls = []
    original_verify = baselines._verify_input_hashes
    original_extract = baselines.extract_url_features

    def replace_path_after_hash(inputs, policy):
        observed = original_verify(inputs, policy)
        lines = paths["train"].read_text().splitlines()
        first = json.loads(lines[0])
        first["raw_url"] = replacement_url
        lines[0] = json.dumps(first, separators=(",", ":"), sort_keys=True)
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_text("\n".join(lines) + "\n")
        os.replace(replacement, paths["train"])
        return observed

    def record_extraction(raw_url):
        observed_urls.append(raw_url)
        return original_extract(raw_url)

    monkeypatch.setattr(baselines, "_verify_input_hashes", replace_path_after_hash)
    monkeypatch.setattr(baselines, "extract_url_features", record_extraction)

    summary = baselines._fit_baselines(
        train_path=paths["train"],
        validation_path=paths["validation"],
        preparation_summary_path=paths["preparation_summary"],
        contract_path=paths["contract"],
        output_dir=paths["output_dir"],
        summary_path=paths["summary"],
        _input_hash_policy=paths["policy"],
    )

    assert replacement_url not in observed_urls
    assert summary["input_hashes"]["train"] == original_hash
    assert sha256(paths["train"].read_bytes()).hexdigest() != original_hash


def test_cli_exposes_only_explicit_development_inputs():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "automated_phishing_detection.cli",
            "fit-baselines",
            "--help",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    for option in (
        "--train",
        "--validation",
        "--preparation-summary",
        "--contract",
        "--output-dir",
        "--summary",
    ):
        assert option in completed.stdout
    assert "group-test" not in completed.stdout
    assert "glob" not in completed.stdout


if __name__ == "__main__":
    raise SystemExit(_private_fixture_main(sys.argv[1:]))
