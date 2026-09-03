import csv
import io
import json
import os
import stat
import subprocess
import sys
from hashlib import sha256

import pytest

from automated_phishing_detection import phiusiil

PSL_BYTES = b"// fixture PSL\nexample\n"


def _csv_bytes(rows, headers=("URL", "label", "unused"), bom=False):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(headers)
    writer.writerows(rows)
    prefix = b"\xef\xbb\xbf" if bom else b""
    return prefix + stream.getvalue().encode("utf-8")


def _source_spec(csv_path, csv_bytes, psl_bytes=PSL_BYTES):
    return {
        "contract_id": "phiusiil-development-v1",
        "schema_version": 1,
        "phiusiil": {
            "uci_dataset_id": 967,
            "archive_url": "https://archive.ics.uci.edu/static/public/967/phiusiil%2Bphishing%2Burl%2Bdataset.zip",
            "archive_sha256": "0a639fd03aea6308c5b1c10c92aa23c2ce1505447a9137271865cd0badc9a59a",
            "csv_filename": csv_path.name,
            "csv_sha256": sha256(csv_bytes).hexdigest(),
            "license": "CC BY 4.0",
            "page_url": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
        },
        "public_suffix_list": {
            "url": "https://raw.githubusercontent.com/publicsuffix/list/0f1fa47ec45056a19c2fdcd32a08442de9715d12/public_suffix_list.dat",
            "upstream_url": "https://publicsuffix.org/list/public_suffix_list.dat",
            "version": "commit-pinned snapshot",
            "commit": "0f1fa47ec45056a19c2fdcd32a08442de9715d12",
            "sha256": sha256(psl_bytes).hexdigest(),
            "license": "MPL-2.0",
        },
    }


def _write_inputs(tmp_path, csv_bytes, psl_bytes=PSL_BYTES, spec_mutator=None):
    csv_path = tmp_path / "PhiUSIIL_Phishing_URL_Dataset.csv"
    psl_path = tmp_path / "public_suffix_list.dat"
    spec_path = tmp_path / "sources.json"
    csv_path.write_bytes(csv_bytes)
    psl_path.write_bytes(psl_bytes)
    spec = _source_spec(csv_path, csv_bytes, psl_bytes)
    if spec_mutator is not None:
        spec_mutator(spec)
    spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return csv_path, psl_path, spec_path


def _two_class_rows():
    rows = []
    for index in range(20):
        domain = f"domain-{index:02d}.example"
        rows.extend(
            [
                (f"https://left.{domain}/", "0", "ignored"),
                (f"https://right.{domain}/", "1", "ignored"),
            ]
        )
    rows.append(("HTTPS://LEFT.DOMAIN-00.EXAMPLE:443", "0", "duplicate"))
    rows.append(("not an absolute URL", "0", "invalid"))
    return rows


def _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "automated_phishing_detection.cli",
            "prepare-phiusiil",
            "--csv",
            str(csv_path),
            "--suffix-rules",
            str(psl_path),
            "--source-spec",
            str(spec_path),
            "--output-dir",
            str(output_dir),
            "--summary",
            str(summary_path),
        ],
        capture_output=True,
        check=False,
        env=environment,
    )


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _successful_run(tmp_path, suffix=""):
    csv_bytes = _csv_bytes(_two_class_rows(), bom=True)
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / f"prepared{suffix}"
    summary_path = tmp_path / f"summary{suffix}.json"
    completed = _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path)
    return completed, csv_path, psl_path, spec_path, output_dir, summary_path


def test_prepare_cli_writes_private_validated_artifacts_and_public_summary(tmp_path):
    summary_path = tmp_path / "summary.json"
    csv_bytes = _csv_bytes(_two_class_rows(), bom=True)
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"

    completed = _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path)

    assert completed.returncode == 0, completed.stderr.decode()
    assert completed.stderr == b""
    summary = json.loads(summary_path.read_bytes())
    assert json.loads(completed.stdout) == summary
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700

    expected_names = {
        "train.jsonl",
        "validation.jsonl",
        "group_test.jsonl",
        "quarantine.jsonl",
        "SHA256SUMS",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_names
    for path in output_dir.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    split_rows = {
        split: _read_jsonl(output_dir / f"{split}.jsonl")
        for split in ("train", "validation", "group_test")
    }
    for split, rows in split_rows.items():
        assert rows == sorted(rows, key=lambda row: row["record_id"])
        assert all(
            set(row)
            == {
                "record_id",
                "raw_url",
                "canonical_url_sha256",
                "registrable_domain",
                "is_phishing",
                "split",
            }
            for row in rows
        )
        assert {row["split"] for row in rows} == {split}
        assert {row["is_phishing"] for row in rows} == {0, 1}

    assert {split: len(rows) for split, rows in split_rows.items()} == {
        "train": 28,
        "validation": 6,
        "group_test": 6,
    }
    all_retained = sum(split_rows.values(), [])
    assert len({row["canonical_url_sha256"] for row in all_retained}) == 40
    domain_splits = {}
    for row in all_retained:
        previous = domain_splits.setdefault(row["registrable_domain"], row["split"])
        assert previous == row["split"]

    quarantine = _read_jsonl(output_dir / "quarantine.jsonl")
    assert quarantine == sorted(quarantine, key=lambda row: row["record_id"])
    assert all(
        set(row) == {"record_id", "canonical_url_sha256", "reason_code"}
        for row in quarantine
    )
    assert [row["reason_code"] for row in quarantine] == [
        "canonical_url_duplicate_same_mapping",
        "invalid_or_missing_url",
    ]
    assert quarantine[0]["canonical_url_sha256"] is not None
    assert quarantine[1]["canonical_url_sha256"] is None

    checksum_entries = {}
    for line in (output_dir / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, filename = line.split("  ", 1)
        checksum_entries[filename] = digest
    assert set(checksum_entries) == expected_names - {"SHA256SUMS"}
    for filename, digest in checksum_entries.items():
        assert sha256((output_dir / filename).read_bytes()).hexdigest() == digest

    source_spec_bytes = spec_path.read_bytes()
    assert summary["schema_version"] == 1
    assert summary["source_spec_sha256"] == sha256(source_spec_bytes).hexdigest()
    assert summary["declared_sources"] == json.loads(source_spec_bytes)
    assert summary["algorithms"] == {
        "allocation_basis": "unique_ascii_domain_groups",
        "allocation_version": "hamilton-largest-remainder-v1",
        "canonicalization_version": "canonical-url-v1",
        "domain_split_version": "phiusiil-domain-split-v1",
        "record_identifier_version": "phiusiil-row-v1",
        "seed": "20260816",
        "split_percentages": {
            "group_test": 15,
            "train": 70,
            "validation": 15,
        },
    }
    assert summary["overall_counts"] == {
        "input_rows": 42,
        "quarantined_rows": 2,
        "retained_domains": 20,
        "retained_rows": 40,
        "canonicalized_url_groups": 40,
    }
    assert summary["native_label_counts"] == {"0": 22, "1": 20, "invalid": 0}
    assert summary["local_label_counts"] == {"0": 20, "1": 20}
    assert summary["quarantine_reason_counts"] == {
        "canonical_url_conflicting_mapping": 0,
        "canonical_url_duplicate_same_mapping": 1,
        "invalid_or_missing_url": 1,
        "invalid_phiusiil_native_label": 0,
    }
    assert {
        split: (counts["row_count"], counts["domain_count"], counts["class_counts"])
        for split, counts in summary["splits"].items()
    } == {
        "train": (28, 14, {"0": 14, "1": 14}),
        "validation": (6, 3, {"0": 3, "1": 3}),
        "group_test": (6, 3, {"0": 3, "1": 3}),
    }
    assert set(summary["output_hashes"]) == expected_names
    for filename, digest in summary["output_hashes"].items():
        assert sha256((output_dir / filename).read_bytes()).hexdigest() == digest


def test_public_summary_contains_no_record_or_location_data(tmp_path):
    completed, _, _, _, _, summary_path = _successful_run(tmp_path)
    assert completed.returncode == 0, completed.stderr.decode()

    summary = json.loads(summary_path.read_bytes())
    forbidden_keys = {
        "raw_url",
        "canonical_url_sha256",
        "hostname",
        "registrable_domain",
        "record_id",
    }

    def visit(value):
        if isinstance(value, dict):
            assert not (set(value) & forbidden_keys)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(summary)
    serialized = summary_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert "https://left.domain-00.example/" not in serialized
    assert "domain-00.example" not in serialized
    assert "phiusiil-row-v1:" not in serialized


def test_repeated_cli_runs_are_byte_identical(tmp_path):
    csv_bytes = _csv_bytes(_two_class_rows(), bom=True)
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"
    first_summary = tmp_path / "first-summary.json"
    second_summary = tmp_path / "second-summary.json"

    first = _run_prepare(csv_path, psl_path, spec_path, first_output, first_summary)
    second = _run_prepare(csv_path, psl_path, spec_path, second_output, second_summary)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == b""
    assert first_summary.read_bytes() == second_summary.read_bytes()
    for filename in (
        "train.jsonl",
        "validation.jsonl",
        "group_test.jsonl",
        "quarantine.jsonl",
        "SHA256SUMS",
    ):
        assert (first_output / filename).read_bytes() == (
            second_output / filename
        ).read_bytes()


@pytest.mark.parametrize("mismatched_input", ["csv", "psl"])
def test_hash_mismatch_precedes_parsing_and_leaves_no_artifacts(
    tmp_path, mismatched_input
):
    csv_bytes = b"\xff" if mismatched_input == "csv" else _csv_bytes([])
    psl_bytes = b"\xff" if mismatched_input == "psl" else PSL_BYTES

    def mutate(spec):
        if mismatched_input == "csv":
            spec["phiusiil"]["csv_sha256"] = "0" * 64
        else:
            spec["public_suffix_list"]["sha256"] = "0" * 64

    csv_path, psl_path, spec_path = _write_inputs(
        tmp_path, csv_bytes, psl_bytes, mutate
    )
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"

    completed = _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path)

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert f"{mismatched_input.upper()} SHA-256 mismatch".encode() in completed.stderr
    assert b"utf-8" not in completed.stderr
    assert not output_dir.exists()
    assert not summary_path.exists()
    assert not any(
        path.name.startswith(".prepared.tmp-") for path in tmp_path.iterdir()
    )


@pytest.mark.parametrize(
    "csv_bytes",
    [
        pytest.param(_csv_bytes([], ("URL", "label", "URL")), id="duplicate-header"),
        pytest.param(_csv_bytes([], ("URL", "target")), id="missing-label"),
        pytest.param(b"", id="missing-header"),
    ],
)
def test_csv_schema_failures_leave_no_artifacts(tmp_path, csv_bytes):
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"

    completed = _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path)

    assert completed.returncode == 2
    assert b"CSV header" in completed.stderr
    assert not output_dir.exists()
    assert not summary_path.exists()


def test_source_spec_rejects_unexpected_extra_source(tmp_path):
    csv_bytes = _csv_bytes(_two_class_rows())

    def add_unexpected_source(spec):
        spec["phishvn"] = {"url": "https://unexpected.example"}

    csv_path, psl_path, spec_path = _write_inputs(
        tmp_path, csv_bytes, spec_mutator=add_unexpected_source
    )
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"

    completed = _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path)

    assert completed.returncode == 2
    assert b"source spec fields" in completed.stderr
    assert not output_dir.exists()
    assert not summary_path.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda spec: spec.update(contract_id="other-contract"),
            id="contract-id",
        ),
        pytest.param(
            lambda spec: spec["phiusiil"].update(
                archive_url="https://unexpected.example/archive.zip"
            ),
            id="phiusiil-identity",
        ),
        pytest.param(
            lambda spec: spec["public_suffix_list"].update(version="other-version"),
            id="psl-version",
        ),
        pytest.param(
            lambda spec: spec["public_suffix_list"].update(
                url="https://publicsuffix.org/list/public_suffix_list.dat"
            ),
            id="psl-mutable-url",
        ),
        pytest.param(
            lambda spec: spec["public_suffix_list"].update(
                upstream_url="https://unexpected.example/public_suffix_list.dat"
            ),
            id="psl-upstream-url",
        ),
    ],
)
def test_source_spec_rejects_unrecognized_contract_metadata(tmp_path, mutate):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(
        tmp_path, csv_bytes, spec_mutator=mutate
    )
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"

    completed = _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path)

    assert completed.returncode == 2
    assert b"source spec" in completed.stderr
    assert not output_dir.exists()
    assert not summary_path.exists()


def test_single_class_failure_does_not_write_any_artifact(tmp_path):
    rows = [
        (f"https://domain-{index:02d}.example/", "0", "unused") for index in range(20)
    ]
    csv_bytes = _csv_bytes(rows)
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"

    completed = _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path)

    assert completed.returncode == 2
    assert b"both classes" in completed.stderr
    assert not output_dir.exists()
    assert not summary_path.exists()


def test_injected_write_failure_removes_all_temporary_artifacts(tmp_path, monkeypatch):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"
    original_writer = phiusiil._write_private_file
    calls = 0

    def fail_second_write(path, content):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        original_writer(path, content)

    monkeypatch.setattr(phiusiil, "_write_private_file", fail_second_write)

    with pytest.raises(OSError, match="injected write failure"):
        phiusiil.prepare_phiusiil(
            csv_path=csv_path,
            suffix_rules_path=psl_path,
            source_spec_path=spec_path,
            output_dir=output_dir,
            summary_path=summary_path,
        )

    assert not output_dir.exists()
    assert not summary_path.exists()
    assert not any(
        path.name.startswith(".prepared.tmp-") for path in tmp_path.iterdir()
    )


def test_summary_temp_permission_failure_leaves_no_partial_artifact(
    tmp_path, monkeypatch
):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"

    def fail_permission_change(descriptor, mode):
        raise OSError("injected summary permission failure")

    monkeypatch.setattr(phiusiil.os, "fchmod", fail_permission_change)

    with pytest.raises(OSError, match="injected summary permission failure"):
        phiusiil.prepare_phiusiil(
            csv_path=csv_path,
            suffix_rules_path=psl_path,
            source_spec_path=spec_path,
            output_dir=output_dir,
            summary_path=summary_path,
        )

    assert not output_dir.exists()
    assert not summary_path.exists()
    assert not any(
        path.name.startswith(f".{summary_path.name}.") for path in tmp_path.iterdir()
    )


def test_existing_summary_is_rejected_without_publishing_output(tmp_path):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"
    summary_path.write_bytes(b"preserve me\n")

    completed = _run_prepare(csv_path, psl_path, spec_path, output_dir, summary_path)

    assert completed.returncode == 2
    assert b"summary path already exists" in completed.stderr
    assert not output_dir.exists()
    assert summary_path.read_bytes() == b"preserve me\n"


def test_summary_publish_race_leaves_output_and_competitor_summary(
    tmp_path, monkeypatch
):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"
    original_publisher = getattr(phiusiil, "_publish_path_without_replace", None)
    assert callable(original_publisher)
    calls = 0

    def race_at_summary_publish(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            destination.write_bytes(b"competitor summary\n")
        return original_publisher(source, destination)

    monkeypatch.setattr(
        phiusiil, "_publish_path_without_replace", race_at_summary_publish
    )

    with pytest.raises(phiusiil.PreparationError, match="already exists"):
        phiusiil.prepare_phiusiil(
            csv_path=csv_path,
            suffix_rules_path=psl_path,
            source_spec_path=spec_path,
            output_dir=output_dir,
            summary_path=summary_path,
        )

    assert output_dir.is_dir()
    assert (output_dir / "SHA256SUMS").is_file()
    assert summary_path.read_bytes() == b"competitor summary\n"
    assert not any(".tmp-" in path.name for path in tmp_path.iterdir())


def test_interruption_after_summary_publish_leaves_both_published_artifacts(
    tmp_path, monkeypatch
):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"
    original_publisher = getattr(phiusiil, "_publish_path_without_replace", None)
    assert callable(original_publisher)
    calls = 0

    def publish_then_interrupt(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            os.rename(source, destination)
            raise KeyboardInterrupt("injected post-summary interruption")
        return original_publisher(source, destination)

    monkeypatch.setattr(
        phiusiil,
        "_publish_path_without_replace",
        publish_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="post-summary interruption"):
        phiusiil.prepare_phiusiil(
            csv_path=csv_path,
            suffix_rules_path=psl_path,
            source_spec_path=spec_path,
            output_dir=output_dir,
            summary_path=summary_path,
        )

    assert output_dir.is_dir()
    summary = json.loads(summary_path.read_bytes())
    assert (
        summary["output_hashes"]["SHA256SUMS"]
        == sha256((output_dir / "SHA256SUMS").read_bytes()).hexdigest()
    )
    assert not any(".tmp-" in path.name for path in tmp_path.iterdir())


def test_interruption_after_output_publish_leaves_completed_output_without_summary(
    tmp_path, monkeypatch
):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"
    original_publisher = getattr(phiusiil, "_publish_path_without_replace", None)
    assert callable(original_publisher)

    def publish_then_interrupt(source, destination):
        os.rename(source, destination)
        raise KeyboardInterrupt("injected post-output interruption")

    monkeypatch.setattr(
        phiusiil, "_publish_path_without_replace", publish_then_interrupt
    )

    with pytest.raises(KeyboardInterrupt, match="post-output interruption"):
        phiusiil.prepare_phiusiil(
            csv_path=csv_path,
            suffix_rules_path=psl_path,
            source_spec_path=spec_path,
            output_dir=output_dir,
            summary_path=summary_path,
        )

    assert output_dir.is_dir()
    assert (output_dir / "SHA256SUMS").is_file()
    assert not summary_path.exists()
    assert not any(".tmp-" in path.name for path in tmp_path.iterdir())


def test_output_publish_race_does_not_replace_competing_empty_directory(
    tmp_path, monkeypatch
):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    output_dir = tmp_path / "prepared"
    summary_path = tmp_path / "summary.json"
    original_publisher = getattr(phiusiil, "_publish_path_without_replace", None)
    assert callable(original_publisher)
    competing_identity = []

    def race_at_publish(source, destination):
        destination.mkdir(mode=0o700)
        competing_identity.append(
            (destination.stat().st_dev, destination.stat().st_ino)
        )
        return original_publisher(source, destination)

    monkeypatch.setattr(phiusiil, "_publish_path_without_replace", race_at_publish)

    with pytest.raises(phiusiil.PreparationError, match="already exists"):
        phiusiil.prepare_phiusiil(
            csv_path=csv_path,
            suffix_rules_path=psl_path,
            source_spec_path=spec_path,
            output_dir=output_dir,
            summary_path=summary_path,
        )

    assert (output_dir.stat().st_dev, output_dir.stat().st_ino) == competing_identity[0]
    assert list(output_dir.iterdir()) == []
    assert not summary_path.exists()
    assert not any(".tmp-" in path.name for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    ("protected_name", "alias_kind"),
    [
        ("csv", "same-path"),
        ("psl", "same-path"),
        ("source-spec", "same-path"),
        ("csv", "symlink"),
        ("psl", "hardlink"),
    ],
)
def test_existing_summary_alias_is_rejected_without_touching_protected_input(
    tmp_path, protected_name, alias_kind
):
    csv_bytes = _csv_bytes(_two_class_rows())
    csv_path, psl_path, spec_path = _write_inputs(tmp_path, csv_bytes)
    protected_paths = {
        "csv": csv_path,
        "psl": psl_path,
        "source-spec": spec_path,
    }
    protected = protected_paths[protected_name]
    if alias_kind == "same-path":
        summary_path = protected
    else:
        summary_path = tmp_path / "summary-alias.json"
        if alias_kind == "symlink":
            summary_path.symlink_to(protected)
        else:
            os.link(protected, summary_path)
    originals = {path: path.read_bytes() for path in protected_paths.values()}
    output_dir = tmp_path / "prepared"

    with pytest.raises(phiusiil.PreparationError, match="summary path already exists"):
        phiusiil.prepare_phiusiil(
            csv_path=csv_path,
            suffix_rules_path=psl_path,
            source_spec_path=spec_path,
            output_dir=output_dir,
            summary_path=summary_path,
        )

    assert not output_dir.exists()
    assert all(path.read_bytes() == content for path, content in originals.items())
