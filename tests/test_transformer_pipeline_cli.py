import json
import subprocess
import sys

import pytest

from automated_phishing_detection import cli


def test_cli_exposes_only_the_frozen_development_paths():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "automated_phishing_detection.cli",
            "fit-transformer-cascade",
            "--help",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    expected = {
        "--train",
        "--validation",
        "--preparation-summary",
        "--baseline-contract",
        "--logistic-l1-artifact",
        "--transformer-contract",
        "--output-dir",
        "--summary",
    }
    assert expected <= set(completed.stdout.split())
    for forbidden in (
        "group-test",
        "test-path",
        "external",
        "phishvn",
        "epoch",
        "batch-size",
        "learning-rate",
        "device",
    ):
        assert forbidden not in completed.stdout.lower()


def test_subcommand_has_exact_required_options_and_rejects_abbreviations():
    parser = cli._parser()
    subparsers = next(
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    transformer_parser = subparsers.choices["fit-transformer-cascade"]
    expected = {
        "--train",
        "--validation",
        "--preparation-summary",
        "--baseline-contract",
        "--logistic-l1-artifact",
        "--transformer-contract",
        "--output-dir",
        "--summary",
    }
    option_actions = {
        option: action
        for action in transformer_parser._actions
        for option in action.option_strings
        if option.startswith("--") and option != "--help"
    }

    assert set(option_actions) == expected
    assert all(action.required for action in option_actions.values())
    complete = ["fit-transformer-cascade"]
    for option in sorted(expected):
        complete.extend((option, "fixture"))
    transformer_index = complete.index("--transformer-contract")
    abbreviated = complete.copy()
    abbreviated[transformer_index] = "--transformer"
    with pytest.raises(SystemExit):
        parser.parse_args(abbreviated)


@pytest.mark.parametrize(
    "forbidden",
    (
        "--group-test",
        "--test-path",
        "--external",
        "--phishvn",
        "--epochs",
        "--batch-size",
        "--learning-rate",
        "--device",
    ),
)
def test_cli_rejects_held_out_and_runtime_tuning_options(forbidden):
    parser = cli._parser()
    allowed = (
        "--train",
        "--validation",
        "--preparation-summary",
        "--baseline-contract",
        "--logistic-l1-artifact",
        "--transformer-contract",
        "--output-dir",
        "--summary",
    )
    argv = ["fit-transformer-cascade"]
    for option in allowed:
        argv.extend((option, "fixture"))
    argv.extend((forbidden, "forbidden"))

    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_cli_passes_exact_paths_to_the_public_pipeline(tmp_path, monkeypatch, capsys):
    option_names = (
        "train",
        "validation",
        "preparation-summary",
        "baseline-contract",
        "logistic-l1-artifact",
        "transformer-contract",
        "output-dir",
        "summary",
    )
    argv = ["fit-transformer-cascade"]
    expected = {}
    for option in option_names:
        path = tmp_path / option
        argv.extend((f"--{option}", str(path)))
        expected[option.replace("-", "_")] = path
    observed = {}

    def fake_fit(**kwargs):
        observed.update(kwargs)
        return {"status": "completed_development_validation"}

    monkeypatch.setattr(cli.transformer_pipeline, "fit_transformer_cascade", fake_fit)

    assert cli.main(argv) == 0
    assert observed == {
        "train_path": expected["train"],
        "validation_path": expected["validation"],
        "preparation_summary_path": expected["preparation_summary"],
        "baseline_contract_path": expected["baseline_contract"],
        "logistic_l1_artifact_path": expected["logistic_l1_artifact"],
        "transformer_contract_path": expected["transformer_contract"],
        "output_dir": expected["output_dir"],
        "summary_path": expected["summary"],
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "completed_development_validation"
    }


def test_cli_reports_pipeline_errors_without_a_traceback(monkeypatch, capsys):
    def fail(**_kwargs):
        raise cli.transformer_pipeline.TransformerPipelineError("input mismatch")

    monkeypatch.setattr(cli.transformer_pipeline, "fit_transformer_cascade", fail)
    argv = ["fit-transformer-cascade"]
    for option in (
        "train",
        "validation",
        "preparation-summary",
        "baseline-contract",
        "logistic-l1-artifact",
        "transformer-contract",
        "output-dir",
        "summary",
    ):
        argv.extend((f"--{option}", option))

    assert cli.main(argv) == 2
    assert capsys.readouterr().err == "error: input mismatch\n"
