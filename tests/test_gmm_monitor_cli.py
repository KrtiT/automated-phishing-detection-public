from pathlib import Path

import pytest

from automated_phishing_detection import cli

OPTIONS = (
    "train",
    "validation",
    "preparation-summary",
    "baseline-contract",
    "logistic-l1-artifact",
    "gmm-contract",
    "output-dir",
    "summary",
)


def arguments():
    return [
        "fit-gmm-monitor",
        *(value for option in OPTIONS for value in ("--" + option, option)),
    ]


def test_gmm_cli_has_exact_path_only_surface():
    parsed = cli._parser().parse_args(arguments())
    assert vars(parsed) == {
        "command": "fit-gmm-monitor",
        **{key.replace("-", "_"): Path(key) for key in OPTIONS},
    }
    parser = next(
        action
        for action in cli._parser()._actions
        if hasattr(action, "choices") and action.choices
    ).choices["fit-gmm-monitor"]
    assert {action.dest for action in parser._actions} == {
        "help",
        *(key.replace("-", "_") for key in OPTIONS),
    }
    assert all(action.required for action in parser._actions if action.dest != "help")


@pytest.mark.parametrize(
    "forbidden",
    [
        "--test",
        "--group-test",
        "--phishvn",
        "--components",
        "--seed",
        "--threads",
        "--window",
        "--quantile",
        "--device",
        "--gmm-cont",
    ],
)
def test_gmm_cli_rejects_held_out_tuning_and_abbreviations(forbidden):
    with pytest.raises(SystemExit) as exc:
        cli._parser().parse_args(arguments() + [forbidden, "1"])
    assert exc.value.code == 2


def test_gmm_cli_dispatch_and_errors(monkeypatch, capsys):
    import automated_phishing_detection.gmm_monitor as gm

    observed = {}

    def fit(**kwargs):
        observed.update(kwargs)
        return {"status": "completed_development_validation"}

    monkeypatch.setattr(gm, "fit_gmm_monitor", fit)
    assert cli.main(arguments()) == 0
    assert set(observed) == {
        key.replace("-", "_") + ("" if key == "output-dir" else "_path")
        for key in OPTIONS
    }
    assert "completed_development_validation" in capsys.readouterr().out

    def fail(**kwargs):
        raise gm.GMMMonitorError("fixture failure")

    monkeypatch.setattr(gm, "fit_gmm_monitor", fail)
    assert cli.main(arguments()) == 2
    assert capsys.readouterr().err == "error: fixture failure\n"
