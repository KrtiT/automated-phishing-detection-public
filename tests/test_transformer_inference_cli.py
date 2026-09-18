import json
from pathlib import Path

import pytest
import torch

from automated_phishing_detection import cli, transformer_inference


def arguments():
    return [
        "verify-transformer-bundle",
        "--bundle-dir",
        "private-bundle",
        "--summary",
        "summary.json",
        "--summary-sha256",
        "a" * 64,
        "--logistic-l1-artifact",
        "stage1.json",
    ]


def test_artifact_verification_has_only_artifact_inputs():
    parser = cli._parser()
    parsed = parser.parse_args(arguments())
    assert vars(parsed) == {
        "command": "verify-transformer-bundle",
        "bundle_dir": Path("private-bundle"),
        "summary": Path("summary.json"),
        "summary_sha256": "a" * 64,
        "logistic_l1_artifact": Path("stage1.json"),
    }
    subparser = next(
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    ).choices["verify-transformer-bundle"]
    assert {action.dest for action in subparser._actions} == {
        "help",
        "bundle_dir",
        "summary",
        "summary_sha256",
        "logistic_l1_artifact",
    }
    assert all(
        action.required for action in subparser._actions if action.dest != "help"
    )


@pytest.mark.parametrize(
    "option",
    [
        "--train",
        "--validation",
        "--group-test",
        "--phishvn",
        "--threshold",
        "--half-width",
        "--seed",
        "--device",
        "--output-dir",
        "--summary-sha",
    ],
)
def test_artifact_verification_rejects_data_tuning_and_abbreviations(option):
    with pytest.raises(SystemExit) as error:
        cli._parser().parse_args(arguments() + [option, "1"])
    assert error.value.code == 2


def test_artifact_verification_dispatches_without_scoring_or_fitting(
    monkeypatch, capsys
):
    calls = []

    def load(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(transformer_inference, "load_transformer_cascade_bundle", load)
    assert cli.main(arguments()) == 0
    assert calls == [
        {
            "bundle_dir": Path("private-bundle"),
            "public_summary_path": Path("summary.json"),
            "logistic_l1_artifact_path": Path("stage1.json"),
            "expected_public_summary_sha256": "a" * 64,
            "device": torch.device("mps"),
        }
    ]
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {
        "status": "verified_artifact_bundle",
        "analysis_stage": "development_validation_only",
        "summary_sha256": "a" * 64,
        "device": "mps",
        "fit_performed_during_verification": False,
        "research_rows_scored_during_verification": False,
    }


@pytest.mark.parametrize(
    "error_type", [OSError, transformer_inference.TransformerInferenceError]
)
def test_artifact_verification_failure_emits_no_success_record(
    monkeypatch, capsys, error_type
):
    def fail(**_):
        raise error_type("fixture unreadable artifact")

    monkeypatch.setattr(transformer_inference, "load_transformer_cascade_bundle", fail)
    assert cli.main(arguments()) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "error: fixture unreadable artifact\n"
