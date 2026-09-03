import json
import os
import subprocess
import sys
from hashlib import sha256

import pytest

SUFFIX_RULES_BYTES = b"// synthetic rules\ncom\nco.uk\n*.ck\n!www.ck\n"


def _valid_manifest_bytes(schema_version=1):
    manifest = {
        "schema_version": schema_version,
        "records": [
            {
                "record_id": "one",
                "url": "https://login.example.com",
                "label": 1,
                "split": "train",
            },
            {
                "record_id": "two",
                "url": "https://sample.co.uk",
                "label": 0,
                "split": "validation",
            },
            {
                "record_id": "three",
                "url": "https://shop.foo.ck",
                "label": 1,
                "split": "group_test",
            },
        ],
    }
    return (json.dumps(manifest, indent=2) + "\n").encode("utf-8")


def _write_inputs(tmp_path, manifest_bytes=None, suffix_rules_bytes=SUFFIX_RULES_BYTES):
    if manifest_bytes is None:
        manifest_bytes = _valid_manifest_bytes()
    manifest_path = tmp_path / "manifest.json"
    suffix_rules_path = tmp_path / "suffix-rules.dat"
    manifest_path.write_bytes(manifest_bytes)
    suffix_rules_path.write_bytes(suffix_rules_bytes)
    return manifest_path, suffix_rules_path


def _run_preflight(tmp_path, manifest_path, suffix_rules_path):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "automated_phishing_detection.cli",
            "validate-manifest",
            "--manifest",
            str(manifest_path),
            "--suffix-rules",
            str(suffix_rules_path),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        check=False,
    )


def _expected_stdout(manifest_bytes, suffix_rules_bytes=SUFFIX_RULES_BYTES):
    summary = {
        "status": "valid",
        "record_count": 3,
        "registrable_domain_count": 3,
        "splits": {
            "train": {"record_count": 1, "registrable_domain_count": 1},
            "validation": {"record_count": 1, "registrable_domain_count": 1},
            "group_test": {"record_count": 1, "registrable_domain_count": 1},
        },
        "manifest_sha256": sha256(manifest_bytes).hexdigest(),
        "suffix_rules_sha256": sha256(suffix_rules_bytes).hexdigest(),
    }
    return (json.dumps(summary, sort_keys=True) + "\n").encode("utf-8")


def test_cli_prints_sorted_summary_with_exact_input_hashes(tmp_path):
    manifest_bytes = _valid_manifest_bytes()
    manifest_path, suffix_rules_path = _write_inputs(tmp_path, manifest_bytes)

    completed = _run_preflight(tmp_path, manifest_path, suffix_rules_path)

    assert completed.returncode == 0
    assert completed.stdout == _expected_stdout(manifest_bytes)
    assert completed.stderr == b""


def test_cli_repeated_runs_are_byte_identical(tmp_path):
    manifest_bytes = _valid_manifest_bytes()
    manifest_path, suffix_rules_path = _write_inputs(tmp_path, manifest_bytes)

    first = _run_preflight(tmp_path, manifest_path, suffix_rules_path)
    second = _run_preflight(tmp_path, manifest_path, suffix_rules_path)

    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout == _expected_stdout(manifest_bytes)
    assert first.stderr == second.stderr == b""


@pytest.mark.parametrize(
    ("manifest_bytes", "remove_manifest", "error_fragment"),
    [
        pytest.param(
            b"{not json\n",
            False,
            b"Expecting property name enclosed in double quotes",
            id="malformed-json",
        ),
        pytest.param(
            _valid_manifest_bytes(),
            True,
            b"No such file or directory",
            id="missing-file",
        ),
        pytest.param(b"\xff", False, b"utf-8", id="invalid-utf8"),
        pytest.param(
            _valid_manifest_bytes(schema_version=2),
            False,
            b"schema_version must be 1",
            id="manifest-validation",
        ),
    ],
)
def test_cli_reports_input_errors_on_one_stderr_line(
    tmp_path, manifest_bytes, remove_manifest, error_fragment
):
    manifest_path, suffix_rules_path = _write_inputs(tmp_path, manifest_bytes)
    if remove_manifest:
        manifest_path.unlink()

    completed = _run_preflight(tmp_path, manifest_path, suffix_rules_path)

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert completed.stderr.startswith(b"error: ")
    assert completed.stderr.endswith(b"\n")
    assert completed.stderr.count(b"\n") == 1
    assert error_fragment in completed.stderr


@pytest.mark.parametrize(
    "manifest_bytes",
    [
        pytest.param(
            b'{"schema_version":1,"schema_version":1,"records":[]}',
            id="top-level",
        ),
        pytest.param(
            b'{"schema_version":1,"records":[{"record_id":"one",'
            b'"url":"https://one.example.com","label":0,"label":0,'
            b'"split":"train"}]}',
            id="record",
        ),
    ],
)
def test_cli_rejects_duplicate_json_keys(tmp_path, manifest_bytes):
    manifest_path, suffix_rules_path = _write_inputs(tmp_path, manifest_bytes)

    completed = _run_preflight(tmp_path, manifest_path, suffix_rules_path)

    assert completed.returncode == 2
    assert completed.stdout == b""
    assert b"duplicate JSON key" in completed.stderr


def test_cli_creates_no_output_artifact(tmp_path):
    manifest_bytes = _valid_manifest_bytes()
    manifest_path, suffix_rules_path = _write_inputs(tmp_path, manifest_bytes)
    original_paths = set(tmp_path.iterdir())

    completed = _run_preflight(tmp_path, manifest_path, suffix_rules_path)

    assert completed.returncode == 0
    assert completed.stdout == _expected_stdout(manifest_bytes)
    assert set(tmp_path.iterdir()) == original_paths
