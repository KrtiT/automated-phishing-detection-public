import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/run_followup_detection.py"
    spec = importlib.util.spec_from_file_location("followup_detection_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scheme,expected", [("http", "https"), ("HTTPS", "http")])
def test_counterfactual_only_changes_scheme(runner, scheme, expected):
    remainder = "://User@MiXeD.example:443/Path%2f?Query=1#Frag"
    assert runner.swapped_scheme(scheme + remainder) == expected + remainder


def execution_fixture(runner, tmp_path, monkeypatch, *, admitted=True):
    directory = tmp_path / "dissertation/followup-20261001"
    population = directory / "population"
    population.mkdir(parents=True)
    specification = directory / "comparison-specification-v1.md"
    specification.write_text("synthetic test specification")
    (population / "retained.jsonl").write_text("synthetic hash input only")
    summary = {
        "admitted": admitted,
        "output_hashes": {
            "retained.jsonl": runner.digest(population / "retained.jsonl")
        },
    }
    (population / "summary.json").write_text(json.dumps(summary))
    manifest = {
        "code_revision": "synthetic-revision",
        "protocol": "bounded-followup-comparison-v1",
        "software_versions": runner.baselines._software_versions(),
        "specification_sha256": runner.digest(specification),
        "population_directory": "population",
        "population_summary_sha256": runner.digest(population / "summary.json"),
        "runtime_identity": runner.execution_versions(),
        "clarification_hashes": {},
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        runner.subprocess,
        "check_output",
        lambda args, **kwargs: "synthetic-revision" if "rev-parse" in args else "",
    )
    return path, population


def test_admitted_population_and_all_hashes_are_checked_before_execution(
    runner, tmp_path, monkeypatch
):
    manifest, population = execution_fixture(runner, tmp_path, monkeypatch)
    result = runner.verify_execution(tmp_path, manifest)
    assert result[2] == population
    (population / "retained.jsonl").write_text("changed synthetic observations")
    with pytest.raises(ValueError, match="population changed"):
        runner.verify_execution(tmp_path, manifest)


def test_missing_population_holds_the_whole_extension(runner, tmp_path, monkeypatch):
    manifest, _ = execution_fixture(runner, tmp_path, monkeypatch, admitted=False)
    with pytest.raises(ValueError, match="whole-study hold"):
        runner.verify_execution(tmp_path, manifest)


def test_dirty_or_changed_code_never_opens_research_inputs(
    runner, tmp_path, monkeypatch
):
    manifest, _ = execution_fixture(runner, tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner.subprocess, "check_output", lambda *args, **kwargs: "dirty"
    )
    with pytest.raises(ValueError, match="frozen commit"):
        runner.verify_execution(tmp_path, manifest)


def test_repeated_output_cannot_replace_a_preserved_result(runner, tmp_path):
    path = tmp_path / "result.json"
    runner.write_json(path, {"status": "original"})
    with pytest.raises(FileExistsError):
        runner.write_json(path, {"status": "replacement"})
    assert json.loads(path.read_text()) == {"status": "original"}


def test_changed_http_runtime_is_rejected_before_measurement(
    runner, tmp_path, monkeypatch
):
    manifest, _ = execution_fixture(runner, tmp_path, monkeypatch)
    changed = runner.execution_versions()
    changed["packages"]["httpx"] = "unexpected"
    monkeypatch.setattr(runner, "execution_versions", lambda: changed)
    with pytest.raises(ValueError, match="execution runtime"):
        runner.verify_execution(tmp_path, manifest)


def test_clarifications_are_bound_to_the_manifest(runner, tmp_path, monkeypatch):
    manifest, population = execution_fixture(runner, tmp_path, monkeypatch)
    clarification = population.parent / "clarification.md"
    clarification.write_text("original")
    value = json.loads(manifest.read_text())
    value["clarification_hashes"] = {clarification.name: runner.digest(clarification)}
    manifest.write_text(json.dumps(value))
    runner.verify_execution(tmp_path, manifest)
    clarification.write_text("changed")
    with pytest.raises(ValueError, match="clarification"):
        runner.verify_execution(tmp_path, manifest)


def test_failed_fit_preserves_intent_failure_and_cannot_be_repeated(
    runner, tmp_path, monkeypatch
):
    followup = tmp_path / "followup"
    followup.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(runner, "verify_execution", lambda *args: ({}, followup, None))
    monkeypatch.setattr(
        runner,
        "fit",
        lambda *args: (_ for _ in ()).throw(ValueError("synthetic fit failure")),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--context-root",
            str(tmp_path),
            "--manifest",
            str(manifest),
            "--phase",
            "fit",
        ],
    )
    with pytest.raises(ValueError, match="synthetic fit"):
        runner.main()
    attempt = followup / "detection-fit-v1"
    assert (attempt / "intent.json").exists()
    assert json.loads((attempt / "failure.json").read_text())["type"] == "ValueError"
    assert not (attempt / "completion.json").exists()
    with pytest.raises(FileExistsError):
        runner.main()


@pytest.mark.parametrize("name", ["baseline", "candidate"])
def test_evaluation_thresholds_must_match_frozen_artifacts(runner, name):
    assert hasattr(runner, "verify_thresholds"), "threshold binding check is absent"
    baseline = SimpleNamespace(validation_threshold_record={"threshold": 0.7})
    candidate = {"validation_threshold": {"status": "selected", "threshold": 0.8}}
    models = {"baseline_threshold": 0.7, "candidate_threshold": 0.8}
    runner.verify_thresholds(models, baseline, candidate)
    models[f"{name}_threshold"] = 0.9
    with pytest.raises(ValueError, match="threshold binding"):
        runner.verify_thresholds(models, baseline, candidate)
