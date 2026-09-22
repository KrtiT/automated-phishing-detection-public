"""CLI dispatch uses invented paths and substituted entry points only."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_secondary_correction.py"


@pytest.fixture
def cli():
    assert SCRIPT.exists(), "missing correction CLI"
    spec = importlib.util.spec_from_file_location("correction_cli_fixture", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binding_args():
    return [
        "--repo-root",
        "/invented/repository",
        "--expected-revision",
        "a" * 40,
        "--expected-profile-sha256",
        "b" * 64,
    ]


def execution_args():
    result = binding_args()
    for name in (
        "train",
        "validation",
        "suffix-rules",
        "original-attempt",
        "attempt",
        "public-summary",
    ):
        result.extend((f"--{name}", f"/invented/{name}"))
    return result


def test_metadata_check_never_dispatches_execution(cli, monkeypatch, capsys):
    observed = []

    def bind(root, **kwargs):
        observed.append((root, kwargs))

    def forbidden(*args, **kwargs):
        pytest.fail("metadata check attempted research execution")

    monkeypatch.setattr(cli, "bind_correction", bind)
    monkeypatch.setattr(cli, "run_correction", forbidden)
    monkeypatch.setattr(cli, "run_correction_worker", forbidden)
    assert cli.main(binding_args() + ["--check"]) == 0
    assert observed == [
        (
            Path("/invented/repository"),
            {
                "expected_revision": "a" * 40,
                "expected_profile_sha256": "b" * 64,
            },
        )
    ]
    assert "Research inputs were not read" in capsys.readouterr().out


@pytest.mark.parametrize("stage", [None, "retained_audit", "random_forest"])
def test_execution_dispatch_preserves_exact_pins_and_narrow_paths(
    cli, monkeypatch, stage
):
    observed = []

    def run(root, **kwargs):
        observed.append((root, kwargs))

    def forbidden(*args, **kwargs):
        pytest.fail("wrong execution mode")

    monkeypatch.setattr(cli, "bind_correction", forbidden)
    monkeypatch.setattr(cli, "run_correction", run if stage is None else forbidden)
    monkeypatch.setattr(
        cli, "run_correction_worker", forbidden if stage is None else run
    )
    arguments = execution_args()
    if stage is not None:
        arguments.extend(("--worker", stage))
    assert cli.main(arguments) == 0
    root, options = observed[0]
    assert root == Path("/invented/repository")
    assert options.pop("expected_revision") == "a" * 40
    assert options.pop("expected_profile_sha256") == "b" * 64
    paths = options.pop("paths")
    assert paths == cli.CorrectionPaths(
        train=Path("/invented/train"),
        validation=Path("/invented/validation"),
        suffix_rules=Path("/invented/suffix-rules"),
        original_attempt=Path("/invented/original-attempt"),
        attempt=Path("/invented/attempt"),
        public_summary=Path("/invented/public-summary"),
    )
    assert options == ({} if stage is None else {"stage": stage})


@pytest.mark.parametrize(
    "arguments",
    [
        binding_args(),
        execution_args() + ["--check"],
        binding_args() + ["--check", "--worker", "random_forest"],
        execution_args() + ["--worker", "permutation_42"],
        execution_args() + ["--resume"],
        execution_args() + ["--external", "/invented/external"],
        [
            "--repo-root",
            "/invented/repository",
            "--expected-revision",
            "a" * 40,
            "--check",
        ],
    ],
)
def test_invalid_modes_and_paths_fail_before_dispatch(cli, monkeypatch, arguments):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid CLI arguments reached execution")

    monkeypatch.setattr(cli, "bind_correction", forbidden)
    monkeypatch.setattr(cli, "run_correction", forbidden)
    monkeypatch.setattr(cli, "run_correction_worker", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["check", "parent", "worker"])
def test_failure_does_not_print_private_exception_text_or_type(
    cli, monkeypatch, capsys, mode
):
    failure = type("https://private.example/class", (Exception,), {})

    def fail(*args, **kwargs):
        raise failure("/private/model/path: https://private.example/record")

    monkeypatch.setattr(cli, "bind_correction", fail)
    monkeypatch.setattr(cli, "run_correction", fail)
    monkeypatch.setattr(cli, "run_correction_worker", fail)
    arguments = binding_args() + ["--check"] if mode == "check" else execution_args()
    if mode == "worker":
        arguments.extend(("--worker", "random_forest"))
    assert cli.main(arguments) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "stopped" in output.err
    assert "private" not in output.err
    assert "Traceback" not in output.err


def test_real_cli_help_hides_private_worker_modes():
    assert SCRIPT.exists(), "missing correction CLI"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], capture_output=True, check=False
    )
    assert result.returncode == 0
    assert b"--check" in result.stdout
    assert b"--expected-profile-sha256" in result.stdout
    assert b"--worker" not in result.stdout
    assert b"PhishVN" not in result.stdout
