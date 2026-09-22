"""Correction orchestration never uses research inputs in tests."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_development_audit import material, retained  # noqa: F401

from automated_phishing_detection import development_correction, execution_receipt

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def api():
    assert (
        ROOT / "src/automated_phishing_detection/development_correction_runner.py"
    ).exists(), "missing correction supervisor"
    return importlib.import_module(
        "automated_phishing_detection.development_correction_runner"
    )


def test_checkpoint_is_immutable_and_survives_failure(api, tmp_path):
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"fixture": True}
    )
    api._record(attempt, "fit-state.json", b'{"fixture":true}')
    with pytest.raises(execution_receipt.ExecutionReceiptError):
        api._record(attempt, "fit-state.json", b"replacement")
    execution_receipt.record_failure(
        attempt, stage="random_forest", error_type="ValueError"
    )
    assert (attempt.directory / "fit-state.json").read_bytes() == b'{"fixture":true}'
    with pytest.raises(execution_receipt.ExecutionReceiptError):
        api._record(attempt, "fit-input.json", b"late")


@pytest.mark.parametrize("code", [0, 2, -9])
def test_observed_exit_is_persisted_before_verification(
    api, tmp_path, monkeypatch, code
):
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"fixture": True}
    )
    monkeypatch.setattr(
        api.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=code, stdout=b"", stderr=b""),
    )
    observed = api._launch(attempt, "retained_audit", ["invented-command"])
    assert observed == code
    record = json.loads(
        (attempt.directory / "retained_audit-process.json").read_bytes()
    )
    assert record["exit_code"] == code
    assert record["status"] == "worker_exited"


def test_launch_failure_has_no_invented_worker_exit(api, tmp_path, monkeypatch):
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"fixture": True}
    )

    def fail(*args, **kwargs):
        raise OSError("private path must not be retained")

    monkeypatch.setattr(api.subprocess, "run", fail)
    with pytest.raises(api.CorrectionError, match="worker_launch_failed"):
        api._launch(attempt, "random_forest", ["invented-command"])
    content = (attempt.directory / "random_forest-process.json").read_text()
    record = json.loads(content)
    assert record["exit_code"] is None
    assert record["status"] == "worker_launch_failed"
    assert "private" not in content


def test_record_rejects_unknown_names_and_link_aliases(api, tmp_path):
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"fixture": True}
    )
    with pytest.raises(api.CorrectionError):
        api._record(attempt, "../escape", b"x")
    (attempt.directory / "fit-state.json").symlink_to(tmp_path / "target")
    with pytest.raises(execution_receipt.ExecutionReceiptError):
        api._record(attempt, "fit-state.json", b"x")
    assert not (tmp_path / "target").exists()


@pytest.fixture
def execution(api, retained, tmp_path, monkeypatch):  # noqa: F811
    binding = development_correction.CorrectionBinding(
        retained.binding,
        development_correction.PROFILE_SHA256,
        api._json_bytes(retained.accounting),
    )
    paths = api.CorrectionPaths(
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
        tmp_path / "suffix.dat",
        retained.original,
        tmp_path / "correction",
        tmp_path / "summary.json",
    )
    paths.train.write_bytes(retained.data["arguments"]["train_content"])
    paths.validation.write_bytes(retained.data["validation_content"])
    paths.suffix_rules.write_bytes(retained.data["arguments"]["suffix_rules"])
    monkeypatch.setattr(api, "recheck_correction", lambda value: None)
    calls = []

    def run(command, **kwargs):
        stage = command[command.index("--worker") + 1]
        calls.append(stage)
        try:
            api._run_worker(binding, paths, stage)
        except api.CorrectionError:
            return SimpleNamespace(returncode=2, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(api.subprocess, "run", run)
    return SimpleNamespace(binding=binding, paths=paths, calls=calls)


@pytest.mark.parametrize("finalization", ["failed", "completed"])
def test_finalized_root_cannot_start_a_worker(
    api, execution, monkeypatch, finalization
):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=api._identity(execution.binding)
    )
    if finalization == "failed":
        execution_receipt.record_failure(
            parent, stage="random_forest", error_type="ValueError"
        )
    else:
        execution_receipt.publish_completion(
            parent,
            private_outputs={"fixture.json": b"{}"},
            public_summary={"fixture": True},
            public_path=execution.paths.public_summary,
        )
    monkeypatch.setattr(
        api, "_inputs", lambda *args: pytest.fail("read after root finalized")
    )
    with pytest.raises(api.CorrectionError, match="root_finalized"):
        api._run_worker(execution.binding, execution.paths, "retained_audit")
    assert not (parent.directory / "retained_audit").exists()


def test_correction_runs_one_rf_fit_after_no_fit_audit(api, execution, monkeypatch):
    original = api.secondary_tabular.fit_random_forest_v2
    fits = []

    def fit(*args, **kwargs):
        assert execution.calls == ["retained_audit", "random_forest"]
        fits.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(api.secondary_tabular, "fit_random_forest_v2", fit)
    for name in (
        "fit_random_forest",
        "fit_formatting",
        "fit_label_permutation",
    ):
        monkeypatch.setattr(
            api.secondary_tabular,
            name,
            lambda *a, **k: pytest.fail("old fit repeated"),
        )
    summary = api._supervise(execution.binding, execution.paths)
    assert fits == [1]
    assert summary["new_fits"] == 1
    assert summary["worker_exit_codes"] == {"retained_audit": 0, "random_forest": 0}
    assert summary["original_aggregate_accepted"] is False
    assert summary["retained_audit"]["result"]["fits"] == 0
    assert (execution.paths.attempt / "random_forest/fit-state.json").is_file()
    assert json.loads(execution.paths.public_summary.read_bytes()) == summary


def test_failed_audit_never_fits_or_reserves_rf(api, execution, monkeypatch):
    execution.paths.validation.write_bytes(b"wrong bytes")
    monkeypatch.setattr(api, "_rf", lambda *a: pytest.fail("RF after failed audit"))
    with pytest.raises(api.CorrectionError, match="correction_stopped"):
        api._supervise(execution.binding, execution.paths)
    assert execution.calls == ["retained_audit"]
    assert not (execution.paths.attempt / "random_forest").exists()
    assert not execution.paths.public_summary.exists()
    process = json.loads(
        (execution.paths.attempt / "retained_audit-process.json").read_bytes()
    )
    assert process["exit_code"] == 2


def test_earlier_stage_reverified_before_root_completion(api, execution, monkeypatch):
    verify = api._verify_stage
    counts = []

    def changed(binding, paths, parent, stage, **kwargs):
        counts.append(stage)
        if stage == "retained_audit" and counts.count(stage) == 3:
            raise api.CorrectionError("fixture_evidence_changed")
        return verify(binding, paths, parent, stage, **kwargs)

    monkeypatch.setattr(api, "_verify_stage", changed)
    with pytest.raises(api.CorrectionError, match="correction_stopped"):
        api._supervise(execution.binding, execution.paths)
    assert counts.count("retained_audit") == 3
    assert not execution.paths.public_summary.exists()


def test_failed_check_retains_fitted_state_and_safe_identifier(
    api, execution, monkeypatch
):
    def reject(*args, **kwargs):
        kwargs["checkpoint"](b'{"invented_fitted_state":true}\n')
        raise api.secondary_tabular.SecondaryTabularError(
            "private row must not escape", check_id="portable_exact_parity"
        )

    monkeypatch.setattr(api.secondary_tabular, "fit_random_forest_v2", reject)
    with pytest.raises(api.CorrectionError, match="correction_stopped"):
        api._supervise(execution.binding, execution.paths)
    child = execution.paths.attempt / "random_forest"
    details = (child / "failure-details.json").read_text()
    assert json.loads(details)["check_id"] == "portable_exact_parity"
    assert "private row" not in details
    assert (
        child / "fit-state.json"
    ).read_bytes() == b'{"invented_fitted_state":true}\n'
    assert (child / "fit-input.json").exists()
    assert (
        json.loads(
            (execution.paths.attempt / "random_forest-process.json").read_bytes()
        )["exit_code"]
        == 2
    )
    assert not execution.paths.public_summary.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("seed", 43),
        ("model_kind", "permutation"),
        ("training_record_ids_sha256", "bad"),
        ("training_labels_sha256", "F" * 64),
        ("label_digest_encoding", "unknown"),
        ("extra", 1),
    ],
)
def test_fit_input_exact_schema(api, execution, field, value):
    train = [{"record_id": "train", "is_phishing": 0}]
    validation = [{"record_id": "validation", "is_phishing": 1}]
    prepared = {"splits": {"train": {"row_count": 1}}}
    record = api._fit_input(execution.binding, train, validation)
    api._verify_fit_input(record, execution.binding, prepared, ["validation"], [1])
    record[field] = value
    with pytest.raises(api.CorrectionError, match="fit_input_mismatch"):
        api._verify_fit_input(record, execution.binding, prepared, ["validation"], [1])


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("exit_code", False),
        ("stdout_sha256", None),
        ("stderr_sha256", "G" * 64),
        ("extra", 1),
    ],
)
def test_process_exact_schema(api, tmp_path, field, value):
    parent = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"fixture": True}
    )
    record = {
        "schema_version": 1,
        "stage": "random_forest",
        "status": "worker_exited",
        "exit_code": 0,
        "root_reservation_sha256": parent.reservation_sha256,
        "stdout_sha256": "a" * 64,
        "stderr_sha256": "b" * 64,
    }
    api._verify_process(record, parent, "random_forest", 0)
    record[field] = value
    with pytest.raises(api.CorrectionError, match="process_observation_mismatch"):
        api._verify_process(record, parent, "random_forest", 0)


@pytest.mark.parametrize("code", [0, 7])
def test_real_subprocess_observed_exit(api, tmp_path, code):
    parent = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"fixture": True}
    )
    actual = api._launch(
        parent, "random_forest", [sys.executable, "-c", f"raise SystemExit({code})"]
    )
    record = json.loads((parent.directory / "random_forest-process.json").read_bytes())
    assert actual == record["exit_code"] == code
    api._verify_process(record, parent, "random_forest", code)


def test_no_retry_after_failed_attempt(api, execution, monkeypatch):
    execution.paths.train.write_bytes(b"wrong source")
    with pytest.raises(api.CorrectionError):
        api._supervise(execution.binding, execution.paths)
    before = {
        path: path.read_bytes()
        for path in execution.paths.attempt.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(api, "_launch", lambda *a: pytest.fail("retried failed root"))
    with pytest.raises(execution_receipt.ExecutionReceiptError):
        api._supervise(execution.binding, execution.paths)
    assert all(path.read_bytes() == value for path, value in before.items())


def test_final_verification_pins_both_stages_together(api, execution, monkeypatch):
    verify = api._verify_stage
    rf_checks = []

    def change_earlier(binding, paths, parent, stage, **kwargs):
        result = verify(binding, paths, parent, stage, **kwargs)
        if stage == "random_forest":
            rf_checks.append(1)
            if len(rf_checks) == 2:
                (parent.directory / "retained_audit/evidence/audit.json").write_bytes(
                    b"changed"
                )
        return result

    monkeypatch.setattr(api, "_verify_stage", change_earlier)
    with pytest.raises(api.CorrectionError, match="correction_stopped"):
        api._supervise(execution.binding, execution.paths)
    assert len(rf_checks) == 2
    assert not execution.paths.public_summary.exists()
