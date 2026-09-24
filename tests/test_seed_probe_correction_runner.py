"""The probe correction runner uses invented retained evidence and inputs only."""

from __future__ import annotations

import json
import multiprocessing
import sys
import threading
from contextlib import contextmanager
from dataclasses import fields, replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_seed_probe_audit import (  # noqa: F401
    ORIGINAL_REVISION,
    SEED_STAGES,
    _accounting,
    _public_bytes,
    stopped_v2,
)
from test_seed_probe_probes import fixture_data, synthetic_scorer
from test_seed_probe_seeds import sample  # noqa: F401

from automated_phishing_detection import (
    character_transformer,
    execution_receipt,
    probe_replay,
    secondary_development,
    secondary_transformer,
    seed_probe_audit,
    seed_probe_correction,
    seed_probe_probes,
    seed_probe_runner,
    seed_probe_seeds,
)
from automated_phishing_detection import seed_probe_correction_runner as runner
from automated_phishing_detection.seed_probe_execution import SeedProbeExecutionBinding


def _load(path):
    return json.loads(path.read_bytes())


def _inventory(root):
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _instrument_lock_attempts(monkeypatch, *thread_names):
    attempts = {name: threading.Event() for name in thread_names}
    blocked = {name: threading.Event() for name in thread_names}
    real_flock = runner.fcntl.flock

    def observed_flock(descriptor, operation):
        if operation == runner.fcntl.LOCK_EX:
            name = threading.current_thread().name
            attempted = attempts.get(name)
            if attempted is not None:
                attempted.set()
                try:
                    return real_flock(descriptor, operation | runner.fcntl.LOCK_NB)
                except BlockingIOError:
                    blocked[name].set()
        return real_flock(descriptor, operation)

    monkeypatch.setattr(runner.fcntl, "flock", observed_flock)
    return attempts, blocked


def test_no_training_path_and_exact_two_stage_authority():
    assert [field.name for field in fields(runner.ProbeCorrectionPaths)] == [
        "validation",
        "suffix_rules",
        "length_only",
        "logistic_l1",
        "transformer_bundle",
        "gmm",
        "drift_reference",
        "drift_audit",
        "original_attempt",
        "attempt",
        "public_summary",
    ]
    assert "train" not in {field.name for field in fields(runner.ProbeCorrectionPaths)}
    assert runner.STAGES == ("retained_seed_audit", "probes")


@pytest.fixture
def execution(stopped_v2, sample, tmp_path, monkeypatch):  # noqa: F811
    inputs = tmp_path / "correction-inputs"
    inputs.mkdir()
    bundle = inputs / "bundle"
    bundle.mkdir()
    original_probe = stopped_v2.original / "probes"
    copied = {
        "length-only.json": (original_probe / "input-length-only.json").read_bytes(),
        "logistic-l1.json": (original_probe / "input-logistic-l1.json").read_bytes(),
        "gmm.json": (original_probe / "input-gmm.json").read_bytes(),
        "transformer.json": (original_probe / "input-transformer.json").read_bytes(),
        "cascade.json": (original_probe / "input-cascade.json").read_bytes(),
        "vocabulary.json": (original_probe / "input-vocabulary.json").read_bytes(),
        "transformer-weights.npz": sample.artifacts["transformer-weights.npz"],
    }
    for name, content in copied.items():
        (bundle / name).write_bytes(content)
    paths = runner.ProbeCorrectionPaths(
        validation=inputs / "validation.jsonl",
        suffix_rules=inputs / "suffix.dat",
        length_only=inputs / "length-only.json",
        logistic_l1=inputs / "logistic-l1.json",
        transformer_bundle=bundle,
        gmm=inputs / "gmm.json",
        drift_reference=inputs / "training-reference.json",
        drift_audit=inputs / "validation-audit.json",
        original_attempt=stopped_v2.original,
        attempt=tmp_path / "correction-attempt",
        public_summary=tmp_path / "correction-summary.json",
    )
    paths.validation.write_bytes(sample.arguments["validation_bytes"])
    paths.suffix_rules.write_bytes(sample.arguments["suffix_rules_bytes"])
    paths.length_only.write_bytes(copied["length-only.json"])
    paths.logistic_l1.write_bytes(copied["logistic-l1.json"])
    paths.gmm.write_bytes(copied["gmm.json"])
    paths.drift_reference.write_bytes(
        (original_probe / "input-training-reference.json").read_bytes()
    )
    paths.drift_audit.write_bytes(
        (original_probe / "input-validation-audit.json").read_bytes()
    )
    correction_rechecks = []
    monkeypatch.setattr(
        runner,
        "recheck_seed_probe_correction",
        lambda value: correction_rechecks.append(value),
    )
    probe_calls = []

    def probe(binding, **arguments):
        retain = arguments.pop("retain")
        assert binding is stopped_v2.binding.seed_probe
        assert arguments == {
            "validation_bytes": paths.validation.read_bytes(),
            "suffix_rules_bytes": paths.suffix_rules.read_bytes(),
            "artifacts": copied,
            "drift_reference_bytes": paths.drift_reference.read_bytes(),
            "drift_audit_bytes": paths.drift_audit.read_bytes(),
        }
        probe_calls.append(arguments)
        retain(
            "score-row-00-000001.json",
            runner._json_bytes({"invented": "retained row"}),
        )
        retain("stream-00.json", runner._json_bytes({"invented": "stream"}))
        result = {
            "schema_version": 1,
            "status": "completed_invented_probe",
            "protected_evaluation_authorized": False,
        }
        return {"comparison.json": runner._json_bytes(result)}, result

    def verify_probe(binding, *, outputs, auxiliary):
        assert binding is stopped_v2.binding.seed_probe
        assert set(outputs) == {"comparison.json"}
        assert set(auxiliary) == {
            "score-row-00-000001.json",
            "stream-00.json",
        }
        return secondary_development._json(outputs["comparison.json"])

    monkeypatch.setattr(seed_probe_probes, "run_probe_stage", probe)
    monkeypatch.setattr(seed_probe_probes, "verify_probe_stage", verify_probe)
    launches = []

    def dispatch(command, **kwargs):
        stage = command[command.index("--worker") + 1]
        launches.append(stage)
        try:
            runner._run_worker(stopped_v2.binding, paths, stage)
        except runner.ProbeCorrectionRunError:
            return SimpleNamespace(returncode=2, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(runner.subprocess, "run", dispatch)
    return SimpleNamespace(
        binding=stopped_v2.binding,
        paths=paths,
        launches=launches,
        probe_calls=probe_calls,
        correction_rechecks=correction_rechecks,
    )


@pytest.fixture
def verified_execution(tmp_path, monkeypatch):
    seed = sample.__wrapped__(monkeypatch, SimpleNamespace(param=640))
    probe = fixture_data(train_count=8)
    assert seed.arguments["validation_bytes"] == probe.arguments["validation_bytes"]
    assert seed.arguments["suffix_rules_bytes"] == probe.arguments["suffix_rules_bytes"]
    assert seed.fixture["arguments"]["train_content"]
    assert (
        sha256(seed.fixture["arguments"]["train_content"]).hexdigest()
        == probe.binding.pins.train_sha256
    )
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    inputs = tmp_path / "verified-inputs"
    inputs.mkdir()
    bundle = inputs / "bundle"
    bundle.mkdir()
    for name, content in probe.artifacts.items():
        (bundle / name).write_bytes(content)
    base_values = {
        "root": checkout,
        "contract_sha256": "c" * 64,
        "runtime_json": '{"invented":true}',
        "source_hashes": probe.binding.base.source_hashes,
    }

    def seed_binding(revision):
        return SeedProbeExecutionBinding(
            base=SimpleNamespace(revision=revision, **base_values),
            profile_sha256=seed_probe_correction.BASE_PROFILE_SHA256,
            base_profile_sha256="d" * 64,
            stopped_attempt_sha256="e" * 64,
            methods_sha256="f" * 64,
            accepted_development_sha256="1" * 64,
            pins=probe.binding.pins,
            preparation_bytes=probe.binding.preparation_bytes,
            transformer_summary_bytes=probe.binding.transformer_summary_bytes,
            primary_artifact_hashes=probe.binding.primary_artifact_hashes,
            public_operating_points_json=probe.binding.public_operating_points_json,
            retained_drift_summary_json=probe.binding.retained_drift_summary_json,
            training_reference_sha256=probe.binding.training_reference_sha256,
            validation_audit_sha256=probe.binding.validation_audit_sha256,
        )

    original_binding = seed_binding(ORIGINAL_REVISION)
    original_paths = seed_probe_runner.SeedProbePaths(
        train=inputs / "train.jsonl",
        validation=inputs / "validation.jsonl",
        suffix_rules=inputs / "suffix.dat",
        length_only=inputs / "length-only.json",
        logistic_l1=inputs / "logistic-l1.json",
        transformer_bundle=bundle,
        gmm=inputs / "gmm.json",
        drift_reference=inputs / "training-reference.json",
        drift_audit=inputs / "validation-audit.json",
        attempt=tmp_path / "original-attempt",
        public_summary=tmp_path / "unused-original-summary.json",
    )
    original_paths.validation.write_bytes(probe.arguments["validation_bytes"])
    original_paths.suffix_rules.write_bytes(probe.arguments["suffix_rules_bytes"])
    original_paths.length_only.write_bytes(probe.artifacts["length-only.json"])
    original_paths.logistic_l1.write_bytes(probe.artifacts["logistic-l1.json"])
    original_paths.gmm.write_bytes(probe.artifacts["gmm.json"])
    original_paths.drift_reference.write_bytes(probe.arguments["drift_reference_bytes"])
    original_paths.drift_audit.write_bytes(probe.arguments["drift_audit_bytes"])
    monkeypatch.setattr(seed_probe_runner, "recheck_seed_probe_binding", lambda _: None)
    parent = execution_receipt.reserve_attempt(
        original_paths.attempt, identity=seed_probe_runner._identity(original_binding)
    )
    completed = [
        seed_probe_runner._run_worker(
            original_binding, original_paths, "seed_42_calibration"
        )
    ]
    seed_probe_runner._launch(
        parent, "seed_42_calibration", [sys.executable, "-c", "pass"]
    )
    original_paths.train.write_bytes(seed.fixture["arguments"]["train_content"])
    monkeypatch.setattr(
        character_transformer, "_run_training_epoch", lambda *args: 0.25
    )
    monkeypatch.setattr(
        character_transformer,
        "_evaluate_validation",
        lambda *args: (1.0, seed.probabilities),
    )
    for stage in SEED_STAGES[1:]:
        completed.append(
            seed_probe_runner._run_worker(original_binding, original_paths, stage)
        )
        seed_probe_runner._launch(parent, stage, [sys.executable, "-c", "pass"])
    retained_probe_inputs = {
        **{
            f"input-{name}": probe.artifacts[name]
            for name in (
                "length-only.json",
                "logistic-l1.json",
                "gmm.json",
                "transformer.json",
                "cascade.json",
                "vocabulary.json",
            )
        },
        "input-training-reference.json": probe.arguments["drift_reference_bytes"],
        "input-validation-audit.json": probe.arguments["drift_audit_bytes"],
    }
    failed_probe = execution_receipt.reserve_attempt(
        original_paths.attempt / "probes",
        identity={
            "root_reservation_sha256": parent.reservation_sha256,
            "stage": "probes",
        },
    )
    for name, content in retained_probe_inputs.items():
        seed_probe_runner._record(failed_probe, name, content)
    seed_probe_runner._failure(
        failed_probe,
        "probes",
        seed_probe_probes.ProbeStageError("probe_metadata"),
        accepted=SEED_STAGES,
        check="probe_metadata",
        progress={
            "completed_row_prefix": dict.fromkeys(("0", "1", "2", "3"), 0),
            "completed_streams": [],
        },
    )
    seed_probe_runner._launch(
        parent, "probes", [sys.executable, "-c", "raise SystemExit(2)"]
    )
    seed_probe_runner._failure(
        parent,
        "probes",
        Exception(),
        accepted=SEED_STAGES,
        check="worker_exit_not_successful",
        progress=seed_probe_runner._observed_probe_progress(parent),
    )
    accounting = _accounting(original_binding, original_paths.attempt, completed)
    binding = seed_probe_correction.SeedProbeCorrectionBinding(
        seed_probe=seed_binding("a" * 40),
        profile_sha256=seed_probe_correction.PROFILE_SHA256,
        accounting_bytes=_public_bytes(accounting),
    )
    paths = runner.ProbeCorrectionPaths(
        validation=original_paths.validation,
        suffix_rules=original_paths.suffix_rules,
        length_only=original_paths.length_only,
        logistic_l1=original_paths.logistic_l1,
        transformer_bundle=original_paths.transformer_bundle,
        gmm=original_paths.gmm,
        drift_reference=original_paths.drift_reference,
        drift_audit=original_paths.drift_audit,
        original_attempt=original_paths.attempt,
        attempt=tmp_path / "verified-correction-attempt",
        public_summary=tmp_path / "verified-correction-summary.json",
    )
    correction_rechecks = []
    monkeypatch.setattr(
        runner,
        "recheck_seed_probe_correction",
        lambda value: correction_rechecks.append(value),
    )
    monkeypatch.setattr(
        seed_probe_audit, "recheck_seed_probe_correction", lambda value: None
    )
    saved_auxiliary = {}
    scoring_calls = []
    monkeypatch.setattr(
        probe_replay, "make_primary_scorer", synthetic_scorer(scoring_calls)
    )
    saved_outputs, saved_result = seed_probe_probes._run_probe_stage(
        binding.seed_probe,
        **probe.arguments,
        retain=saved_auxiliary.__setitem__,
        _fixture_cpu=True,
    )

    def replay_saved_probe(probe_binding, **arguments):
        retain = arguments.pop("retain")
        assert probe_binding is binding.seed_probe
        assert arguments == probe.arguments
        for name, content in saved_auxiliary.items():
            retain(name, content)
        return saved_outputs, saved_result

    monkeypatch.setattr(seed_probe_probes, "run_probe_stage", replay_saved_probe)
    real_saved_verifier = seed_probe_probes._verify_probe_stage

    def fixture_saved_verifier(
        probe_binding, *, outputs, auxiliary, _fixture_cpu=False
    ):
        return real_saved_verifier(
            probe_binding, outputs=outputs, auxiliary=auxiliary, _fixture_cpu=True
        )

    monkeypatch.setattr(
        seed_probe_probes,
        "_verify_probe_stage",
        fixture_saved_verifier,
    )
    launches = []

    def dispatch(command, **kwargs):
        stage = command[command.index("--worker") + 1]
        launches.append(stage)
        try:
            runner._run_worker(binding, paths, stage)
        except runner.ProbeCorrectionRunError:
            return SimpleNamespace(returncode=2, stdout=b"", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(runner.subprocess, "run", dispatch)
    return SimpleNamespace(
        binding=binding,
        paths=paths,
        launches=launches,
        correction_rechecks=correction_rechecks,
        scoring_calls=scoring_calls,
        saved_auxiliary=saved_auxiliary,
    )


@pytest.mark.parametrize("code", [0, 2, -9])
def test_launch_records_externally_observed_exit(tmp_path, monkeypatch, code):
    parent = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"invented": True}
    )
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=code, stdout=b"out", stderr=b"err"
        ),
    )
    assert runner._launch(parent, "probes", ["invented-command"]) == code
    record = _load(parent.directory / "probes-process.json")
    assert record["status"] == "worker_exited"
    assert record["exit_code"] == code
    assert record["stdout_sha256"] == sha256(b"out").hexdigest()
    assert record["stderr_sha256"] == sha256(b"err").hexdigest()


def test_launch_failure_has_no_invented_exit_or_private_text(tmp_path, monkeypatch):
    parent = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"invented": True}
    )

    def fail(*args, **kwargs):
        raise OSError("/private/input/path")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(runner.ProbeCorrectionRunError, match="worker_launch_failed"):
        runner._launch(parent, "retained_seed_audit", ["invented-command"])
    content = (parent.directory / "retained_seed_audit-process.json").read_text()
    assert (
        _load(parent.directory / "retained_seed_audit-process.json")["exit_code"]
        is None
    )
    assert "private" not in content


@pytest.mark.parametrize("finalization", ["failed", "completed"])
@pytest.mark.parametrize("stage", runner.STAGES)
def test_finalized_root_cannot_start_either_worker(
    execution, monkeypatch, finalization, stage
):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    if finalization == "failed":
        execution_receipt.record_failure(
            parent, stage="reservation", error_type="ValueError"
        )
    else:
        execution_receipt.publish_completion(
            parent,
            private_outputs={"invented.json": b"{}"},
            public_summary={"invented": True},
            public_path=execution.paths.public_summary,
        )
    monkeypatch.setattr(
        runner,
        "_read_probe_inputs",
        lambda *args: pytest.fail("input read after root finalization"),
    )
    with pytest.raises(runner.ProbeCorrectionRunError, match="root_finalized"):
        runner._run_worker(execution.binding, execution.paths, stage)
    assert not (parent.directory / stage).exists()


@pytest.mark.parametrize("observed", [None, False, 2, -9])
# SP-CORR-06: no probe child or input read exists before verified audit success.
def test_probe_worker_requires_exact_zero_audit_exit_before_reservation_or_inputs(
    execution, monkeypatch, observed
):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    if observed is not None:
        runner._record(
            parent,
            "retained_seed_audit-process.json",
            runner._json_bytes(
                {
                    "schema_version": 1,
                    "stage": "retained_seed_audit",
                    "root_reservation_sha256": parent.reservation_sha256,
                    "status": "worker_exited",
                    "exit_code": observed,
                    "stdout_sha256": "a" * 64,
                    "stderr_sha256": "b" * 64,
                }
            ),
        )
    monkeypatch.setattr(
        runner,
        "_read_probe_inputs",
        lambda *args: pytest.fail("probe inputs read before audit gate"),
    )
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="audit_worker_not_successful"
    ):
        runner._run_worker(execution.binding, execution.paths, "probes")
    assert not (parent.directory / "probes").exists()


def test_probe_worker_independently_validates_saved_audit_before_reservation_or_inputs(
    execution, monkeypatch
):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    runner._run_worker(execution.binding, execution.paths, "retained_seed_audit")
    runner._record(
        parent,
        "retained_seed_audit-process.json",
        runner._json_bytes(
            {
                "schema_version": 1,
                "stage": "retained_seed_audit",
                "root_reservation_sha256": parent.reservation_sha256,
                "status": "worker_exited",
                "exit_code": 0,
                "stdout_sha256": sha256(b"").hexdigest(),
                "stderr_sha256": sha256(b"").hexdigest(),
            }
        ),
    )
    monkeypatch.setattr(
        seed_probe_audit,
        "validate_audit_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            seed_probe_audit.SeedProbeAuditError("audit_summary_mismatch")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_read_probe_inputs",
        lambda *args: pytest.fail("probe inputs read before saved audit validation"),
    )
    with pytest.raises(seed_probe_audit.SeedProbeAuditError):
        runner._run_worker(execution.binding, execution.paths, "probes")
    assert not (parent.directory / "probes").exists()


def test_audit_worker_publishes_canonical_exact_private_audit_only(
    execution, monkeypatch
):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    monkeypatch.setattr(
        runner,
        "_read_probe_inputs",
        lambda *args: pytest.fail("audit worker read probe inputs"),
    )
    summary = runner._run_worker(
        execution.binding, execution.paths, "retained_seed_audit"
    )
    audit = parent.directory / "retained_seed_audit/evidence/audit.json"
    assert audit.read_bytes() == runner._json_bytes(summary["result"])
    assert summary["private_sha256"] == {
        "audit.json": sha256(audit.read_bytes()).hexdigest()
    }


def test_worker_binding_recheck_after_computation_precedes_completion(
    execution, monkeypatch
):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    computed = False
    audit = seed_probe_audit.audit_retained_seed_stages

    def compute(*args, **kwargs):
        nonlocal computed
        result = audit(*args, **kwargs)
        computed = True
        return result

    def recheck(_binding):
        if computed:
            raise seed_probe_correction.SeedProbeCorrectionError(
                "invented_binding_change"
            )

    monkeypatch.setattr(seed_probe_audit, "audit_retained_seed_stages", compute)
    monkeypatch.setattr(runner, "recheck_seed_probe_correction", recheck)
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="probe_correction_worker_stopped"
    ):
        runner._run_worker(execution.binding, execution.paths, "retained_seed_audit")
    child = parent.directory / "retained_seed_audit"
    assert computed is True
    assert not (parent.directory / "retained_seed_audit.json").exists()
    assert not (child / "evidence").exists()
    assert _load(child / "outcome.json")["status"] == "failed"


# SP-CORR-07: success is one probe, zero seed executions, zero fits, and failed v2.
def test_supervisor_runs_audit_then_one_probe_with_no_seed_or_fit(
    execution, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("correction attempted a seed, fit, source partition, or scorer")

    for module, name in (
        (seed_probe_runner, "run_seed_probes"),
        (seed_probe_seeds, "run_seed_stage"),
        (seed_probe_seeds, "_fit"),
        (secondary_development, "_partition"),
        (secondary_transformer, "score_secondary_stage1_urls"),
        (character_transformer, "fit_character_transformer"),
        (probe_replay, "make_primary_scorer"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    original = _inventory(execution.paths.original_attempt)
    summary = runner._supervise(execution.binding, execution.paths)
    assert execution.launches == list(runner.STAGES)
    assert len(execution.probe_calls) == 1
    assert len(execution.correction_rechecks) >= 3
    assert execution.correction_rechecks[-1] is execution.binding
    assert summary["new_fits"] == 0
    assert summary["seed_stage_executions"] == 0
    assert summary["probe_executions"] == 1
    assert summary["retries"] == 0
    assert summary["original_v2_aggregate_accepted"] is False
    assert summary["original_v2_profile_status"] == "exhausted"
    assert summary["protected_evaluation_authorized"] is False
    accepted = _load(execution.paths.attempt / "evidence/stage-summaries.json")
    assert summary["retained_seed_audit"] == accepted[0]
    assert summary["probes"] == accepted[1]
    assert "selected_seed" not in summary
    assert "pure_seed_effect_claim" not in summary
    assert accepted[0]["result"]["pure_seed_effect_claim"] is False
    encoded = json.dumps(summary, sort_keys=True)
    assert '"selected_seed"' not in encoded
    assert '"pure_seed_effect_claim": true' not in encoded
    assert _load(execution.paths.public_summary) == summary
    assert _inventory(execution.paths.original_attempt) == original


def test_final_binding_recheck_after_joint_verification_precedes_root_completion(
    execution, monkeypatch
):
    verify = runner._verify_stage
    final_probe_verified = False
    probe_checks = 0

    def observe(binding, paths, parent, stage, **kwargs):
        nonlocal final_probe_verified, probe_checks
        result = verify(binding, paths, parent, stage, **kwargs)
        if stage == "probes":
            probe_checks += 1
            if probe_checks == 2:
                final_probe_verified = True
        return result

    def recheck(_binding):
        if final_probe_verified:
            raise seed_probe_correction.SeedProbeCorrectionError(
                "invented_binding_change"
            )

    monkeypatch.setattr(runner, "_verify_stage", observe)
    monkeypatch.setattr(runner, "recheck_seed_probe_correction", recheck)
    with pytest.raises(runner.ProbeCorrectionRunError):
        runner._supervise(execution.binding, execution.paths)
    assert final_probe_verified is True
    assert not (execution.paths.attempt / "evidence").exists()
    assert _load(execution.paths.attempt / "outcome.json")["status"] == "failed"
    assert not execution.paths.public_summary.exists()


def test_happy_path_manifest_composes_with_real_saved_probe_verifier(
    verified_execution, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("correction attempted a seed, fit, partition, or primary scoring")

    for module, name in (
        (seed_probe_runner, "run_seed_probes"),
        (seed_probe_seeds, "run_seed_stage"),
        (seed_probe_seeds, "_fit"),
        (secondary_development, "_partition"),
        (secondary_transformer, "score_secondary_stage1_urls"),
        (character_transformer, "fit_character_transformer"),
        (probe_replay, "make_primary_scorer"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    scored_before = len(verified_execution.scoring_calls)
    summary = runner._supervise(verified_execution.binding, verified_execution.paths)
    assert verified_execution.launches == list(runner.STAGES)
    assert len(verified_execution.scoring_calls) == scored_before
    manifest = _load(
        verified_execution.paths.attempt / "probes/evidence/artifact-manifest.json"
    )
    assert set(manifest["auxiliary_sha256"]) == set(verified_execution.saved_auxiliary)
    rechecks_before = len(verified_execution.correction_rechecks)
    assert (
        runner._verify_completion(
            verified_execution.binding,
            verified_execution.paths,
            producer_exit_code=0,
        )
        == summary
    )
    assert len(verified_execution.correction_rechecks) > rechecks_before
    assert verified_execution.correction_rechecks[-1] is verified_execution.binding


def test_audit_failure_never_launches_or_reserves_probe_or_reads_inputs(
    execution, monkeypatch
):
    monkeypatch.setattr(
        seed_probe_audit,
        "audit_retained_seed_stages",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            seed_probe_audit.SeedProbeAuditError("retained_seed_audit_failed")
        ),
    )
    monkeypatch.setattr(
        runner,
        "_read_probe_inputs",
        lambda *args: pytest.fail("probe inputs read after failed audit"),
    )
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="probe_correction_stopped"
    ):
        runner._supervise(execution.binding, execution.paths)
    assert execution.launches == ["retained_seed_audit"]
    assert not (execution.paths.attempt / "probes").exists()
    assert not (execution.paths.attempt / "probes-process.json").exists()
    assert not execution.paths.public_summary.exists()


# SP-CORR-08: nonzero execution retains its prefix and never succeeds or retries.
def test_probe_failure_retains_installed_prefix_and_never_retries(
    execution, monkeypatch
):
    def fail(binding, **arguments):
        retain = arguments["retain"]
        retain("score-row-00-000003.json", runner._json_bytes({"row": 3}))
        retain("score-row-00-000001.json", runner._json_bytes({"row": 1}))
        retain("score-row-01-000002.json", runner._json_bytes({"row": 2}))
        retain("score-row-01-000001.json", runner._json_bytes({"row": 1}))
        retain("score-row-02-000004.json", runner._json_bytes({"row": 4}))
        retain("stream-03.json", runner._json_bytes({"stream": 3}))
        retain("stream-01.json", runner._json_bytes({"stream": 1}))
        raise seed_probe_probes.ProbeStageError("probe_replay")

    monkeypatch.setattr(seed_probe_probes, "run_probe_stage", fail)
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="probe_correction_stopped"
    ):
        runner._supervise(execution.binding, execution.paths)
    child = execution.paths.attempt / "probes"
    assert (child / "score-row-00-000001.json").is_file()
    assert (child / "score-row-00-000003.json").is_file()
    assert _load(child / "failure-details.json")["probe_progress"] == {
        "completed_row_prefix": {"0": 1, "1": 2, "2": 0, "3": 0},
        "completed_streams": [1, 3],
    }
    assert _load(execution.paths.attempt / "failure-details.json")[
        "probe_progress"
    ] == {
        "scope": "installed_record_inventory_not_accepted_scientific_evidence",
        "completed_row_prefix": {"0": 1, "1": 2, "2": 0, "3": 0},
        "observed_completed_stream_records": [1, 3],
        "unaccepted_row_records_outside_prefix": 2,
        "inventory_incomplete": False,
        "failed_row_position": None,
        "durability_confirmed": False,
    }
    before = _inventory(execution.paths.attempt)
    monkeypatch.setattr(
        runner, "_launch", lambda *args: pytest.fail("failed root retried")
    )
    with pytest.raises(execution_receipt.ExecutionReceiptError):
        runner._supervise(execution.binding, execution.paths)
    assert _inventory(execution.paths.attempt) == before


def test_unknown_top_level_exception_is_redacted_from_all_failure_records(tmp_path):
    parent = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"invented": True}
    )
    private = type("PrivateResearchException", (Exception,), {})("/secret/PrivatePath")
    private.__cause__ = private
    runner._failure(parent, "probes", private, accepted=("retained_seed_audit",))
    details = _load(parent.directory / "failure-details.json")
    outcome = _load(parent.directory / "outcome.json")
    assert details["error_type"] == "Exception"
    assert details["check"] == "unclassified_check"
    assert outcome["error_type"] == "Exception"
    persisted = "\n".join(
        (parent.directory / name).read_text()
        for name in ("failure-details.json", "outcome.json")
    )
    for sentinel in ("PrivateResearchException", "secret", "PrivatePath"):
        assert sentinel not in persisted


def test_output_paths_must_be_outside_checkout_and_original_attempt(execution):
    for changed in (
        {"attempt": execution.binding.base.root / "attempt"},
        {"public_summary": execution.binding.base.root / "summary.json"},
        {"attempt": execution.paths.original_attempt / "correction"},
        {"public_summary": execution.paths.original_attempt / "summary.json"},
    ):
        with pytest.raises(
            runner.ProbeCorrectionRunError, match="outputs_inside_preserved_input"
        ):
            runner._outside_outputs(
                execution.binding, replace(execution.paths, **changed)
            )


@pytest.mark.parametrize(
    "protected",
    ["transformer_bundle", "drift_reference", "drift_audit"],
)
def test_output_paths_reject_retained_artifact_directory_containment(
    execution, tmp_path, protected
):
    if protected == "transformer_bundle":
        preserved = execution.paths.transformer_bundle
        paths = execution.paths
    else:
        preserved = tmp_path / protected
        preserved.mkdir()
        paths = replace(execution.paths, **{protected: preserved})
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="outputs_inside_preserved_input"
    ):
        runner._outside_outputs(
            execution.binding, replace(paths, attempt=preserved / "correction")
        )


def test_output_path_may_be_distinct_sibling_of_flat_inputs(execution, tmp_path):
    flat_inputs = tmp_path / "generic-flat-inputs"
    flat_inputs.mkdir()
    sibling = flat_inputs / "distinct-output"
    runner._outside_outputs(
        execution.binding,
        replace(
            execution.paths,
            validation=flat_inputs / "validation.jsonl",
            attempt=sibling,
        ),
    )


@pytest.mark.parametrize("drift", ["drift_reference", "drift_audit"])
@pytest.mark.parametrize("output", ["attempt", "public_summary"])
def test_outputs_beneath_retained_drift_parent_are_rejected(
    execution, tmp_path, drift, output
):
    reference_parent = tmp_path / "retained-reference"
    audit_parent = tmp_path / "retained-audit"
    reference_parent.mkdir()
    audit_parent.mkdir()
    paths = replace(
        execution.paths,
        drift_reference=reference_parent / "reference.json",
        drift_audit=audit_parent / "audit.json",
    )
    changed = {output: getattr(paths, drift).parent / f"{output}-output"}
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="outputs_inside_preserved_input"
    ):
        runner._outside_outputs(
            execution.binding,
            replace(paths, **changed),
        )


@pytest.mark.parametrize(
    "input_name",
    [
        "validation",
        "suffix_rules",
        "length_only",
        "logistic_l1",
        "transformer_bundle",
        "gmm",
        "drift_reference",
        "drift_audit",
        "original_attempt",
    ],
)
def test_parent_traversal_in_input_path_stops_before_root_reservation(
    execution, monkeypatch, input_name
):
    actual = getattr(execution.paths, input_name)
    detour = actual.parent / f"{input_name}-detour"
    detour.mkdir()
    lexical = detour / ".." / actual.name
    paths = replace(execution.paths, **{input_name: lexical})
    reservations = []

    def reserve(*args, **kwargs):
        reservations.append((args, kwargs))
        pytest.fail("unsafe preserved path reached root reservation")

    monkeypatch.setattr(runner, "recheck_seed_probe_correction", lambda binding: None)
    monkeypatch.setattr(runner.receipt, "reserve_attempt", reserve)

    with pytest.raises(runner.ProbeCorrectionRunError, match="invalid_paths"):
        runner._supervise(execution.binding, paths)
    assert reservations == []


@pytest.mark.parametrize("input_name", ["drift_reference", "drift_audit"])
def test_terminal_parent_traversal_in_drift_path_stops_before_root_reservation(
    execution, monkeypatch, input_name
):
    actual = getattr(execution.paths, input_name)
    detour = actual.parent / f"terminal-{input_name}-detour"
    detour.mkdir()
    paths = replace(execution.paths, **{input_name: detour / ".."})
    reservations = []

    def reserve(*args, **kwargs):
        reservations.append((args, kwargs))
        pytest.fail("unsafe input path reached root reservation")

    monkeypatch.setattr(runner, "recheck_seed_probe_correction", lambda binding: None)
    monkeypatch.setattr(runner.receipt, "reserve_attempt", reserve)

    with pytest.raises(runner.ProbeCorrectionRunError, match="invalid_paths"):
        runner._supervise(execution.binding, paths)
    assert reservations == []


@pytest.mark.parametrize("preserved", ["checkout", "original_attempt"])
def test_direct_worker_rejects_unsafe_root_before_child(execution, preserved):
    parent = (
        execution.binding.base.root
        if preserved == "checkout"
        else execution.paths.original_attempt
    )
    unsafe = parent / "correction-attempt"
    paths = replace(execution.paths, attempt=unsafe)
    execution_receipt.reserve_attempt(
        unsafe, identity=runner._identity(execution.binding)
    )
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="invalid_completion_paths"
    ):
        runner._run_worker(execution.binding, paths, "retained_seed_audit")
    assert not (unsafe / "retained_seed_audit").exists()


@pytest.mark.parametrize(
    "preserved",
    ["transformer_bundle", "drift_reference", "drift_audit"],
)
def test_direct_worker_rejects_retained_artifact_containment_before_child(
    execution, tmp_path, preserved
):
    if preserved == "transformer_bundle":
        directory = execution.paths.transformer_bundle
        paths = execution.paths
    else:
        directory = tmp_path / f"{preserved}-directory"
        directory.mkdir()
        paths = replace(
            execution.paths,
            **{preserved: directory / f"{preserved}.json"},
        )
    unsafe = directory / "correction-attempt"
    paths = replace(paths, attempt=unsafe)
    execution_receipt.reserve_attempt(
        unsafe, identity=runner._identity(execution.binding)
    )
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="invalid_completion_paths"
    ):
        runner._run_worker(execution.binding, paths, "retained_seed_audit")
    assert not (unsafe / "retained_seed_audit").exists()


def test_changed_audit_after_probe_prevents_root_success(execution, monkeypatch):
    verify = runner._verify_stage
    probe_checks = []

    def mutate(binding, paths, parent, stage, **kwargs):
        result = verify(binding, paths, parent, stage, **kwargs)
        if stage == "probes":
            probe_checks.append(1)
            if len(probe_checks) == 1:
                (
                    parent.directory / "retained_seed_audit/evidence/audit.json"
                ).write_bytes(b"changed")
        return result

    monkeypatch.setattr(runner, "_verify_stage", mutate)
    with pytest.raises(runner.ProbeCorrectionRunError):
        runner._supervise(execution.binding, execution.paths)
    assert probe_checks == [1]
    assert not execution.paths.public_summary.exists()


def test_changed_probe_during_final_joint_snapshot_prevents_success(
    execution, monkeypatch
):
    verify = runner._verify_stage
    probe_checks = []

    def mutate(binding, paths, parent, stage, **kwargs):
        result = verify(binding, paths, parent, stage, **kwargs)
        if stage == "probes":
            probe_checks.append(1)
            if len(probe_checks) == 2:
                (parent.directory / "probes/score-row-00-000001.json").write_bytes(
                    b"changed"
                )
        return result

    monkeypatch.setattr(runner, "_verify_stage", mutate)
    with pytest.raises(runner.ProbeCorrectionRunError):
        runner._supervise(execution.binding, execution.paths)
    assert probe_checks == [1, 1]
    assert not execution.paths.public_summary.exists()


def test_final_probe_verification_keeps_audit_snapshot_open(execution, monkeypatch):
    verify = runner._verify_stage
    probe_checks = []

    def mutate(binding, paths, parent, stage, **kwargs):
        result = verify(binding, paths, parent, stage, **kwargs)
        if stage == "probes":
            probe_checks.append(1)
            if len(probe_checks) == 2:
                (
                    parent.directory / "retained_seed_audit/evidence/audit.json"
                ).write_bytes(b"changed-during-final-probe-verification")
        return result

    monkeypatch.setattr(runner, "_verify_stage", mutate)
    with pytest.raises(runner.ProbeCorrectionRunError):
        runner._supervise(execution.binding, execution.paths)
    assert probe_checks == [1, 1]
    assert not execution.paths.public_summary.exists()


def test_stage_mutation_during_root_summary_prevents_publication(
    execution, monkeypatch
):
    root_summary = runner._root_summary

    def mutate(binding, parent, accepted):
        summary = root_summary(binding, parent, accepted)
        (parent.directory / "probes/score-row-00-000001.json").write_bytes(
            b"changed-during-publication"
        )
        return summary

    monkeypatch.setattr(runner, "_root_summary", mutate)
    with pytest.raises(runner.ProbeCorrectionRunError):
        runner._supervise(execution.binding, execution.paths)
    assert not execution.paths.public_summary.exists()
    assert _load(execution.paths.attempt / "outcome.json")["status"] == "failed"


def test_root_publication_occurs_with_both_stage_snapshots_active(
    execution, monkeypatch
):
    snapshot = runner._stage_snapshot
    publish = execution_receipt.publish_completion
    active = set()
    observed = []

    @contextmanager
    def tracked_snapshot(parent, stage):
        with snapshot(parent, stage) as opened:
            active.add(stage)
            try:
                yield opened
            finally:
                active.remove(stage)

    def checked_publish(attempt, **kwargs):
        if attempt.directory == execution.paths.attempt:
            observed.append(frozenset(active))
            assert active == set(runner.STAGES)
        return publish(attempt, **kwargs)

    monkeypatch.setattr(runner, "_stage_snapshot", tracked_snapshot)
    monkeypatch.setattr(execution_receipt, "publish_completion", checked_publish)
    runner._supervise(execution.binding, execution.paths)
    assert observed == [frozenset(runner.STAGES)]


def test_unexpected_root_file_during_summary_prevents_publication(
    execution, monkeypatch
):
    root_summary = runner._root_summary

    def mutate(binding, parent, accepted):
        summary = root_summary(binding, parent, accepted)
        (parent.directory / "unexpected.json").write_bytes(b"{}")
        return summary

    monkeypatch.setattr(runner, "_root_summary", mutate)
    with pytest.raises(runner.ProbeCorrectionRunError):
        runner._supervise(execution.binding, execution.paths)
    assert not execution.paths.public_summary.exists()
    assert not (execution.paths.attempt / "evidence").exists()
    assert _load(execution.paths.attempt / "outcome.json")["status"] == "failed"


@pytest.mark.parametrize("code", [False, 2, -9])
def test_saved_verifier_rejects_nonzero_or_boolean_producer_before_io(
    tmp_path, monkeypatch, code
):
    monkeypatch.setattr(
        runner,
        "bind_seed_probe_correction",
        lambda *args, **kwargs: pytest.fail("binding read before producer exit gate"),
    )
    paths = runner.ProbeCorrectionPaths(
        **{
            field.name: tmp_path / field.name
            for field in fields(runner.ProbeCorrectionPaths)
        }
    )
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="producer_exit_not_successful"
    ):
        runner.verify_probe_correction(
            tmp_path,
            expected_revision="a" * 40,
            expected_profile_sha256="b" * 64,
            paths=paths,
            producer_exit_code=code,
        )


def test_verify_stage_rejects_boolean_exit_before_snapshot(execution, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_stage_snapshot",
        lambda *args: pytest.fail("stage snapshot opened before exact-int gate"),
    )
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="worker_exit_not_successful"
    ):
        runner._verify_stage(
            execution.binding,
            execution.paths,
            None,
            "probes",
            observed_exit=False,
        )


def test_verify_completion_rejects_boolean_exit_before_binding_io(
    execution, monkeypatch
):
    monkeypatch.setattr(
        runner,
        "recheck_seed_probe_correction",
        lambda *args: pytest.fail("binding read before exact-int gate"),
    )
    with pytest.raises(
        runner.ProbeCorrectionRunError, match="producer_exit_not_successful"
    ):
        runner._verify_completion(
            execution.binding, execution.paths, producer_exit_code=False
        )


def test_supervisor_rejects_boolean_worker_exit_without_next_launch(
    execution, monkeypatch
):
    launches = []

    def launch(*args):
        launches.append(args[1])
        return False

    monkeypatch.setattr(runner, "_launch", launch)
    with pytest.raises(runner.ProbeCorrectionRunError):
        runner._supervise(execution.binding, execution.paths)
    assert launches == ["retained_seed_audit"]
    assert not (execution.paths.attempt / "retained_seed_audit").exists()
    assert not execution.paths.public_summary.exists()


# SP-CORR-09: completion verification is saved-only and mutation-sensitive.
def test_saved_verification_uses_saved_evidence_without_source_reads_or_execution(
    execution, monkeypatch
):
    summary = runner._supervise(execution.binding, execution.paths)
    source_inputs = {
        execution.paths.validation.absolute(),
        execution.paths.suffix_rules.absolute(),
        execution.paths.length_only.absolute(),
        execution.paths.logistic_l1.absolute(),
        execution.paths.gmm.absolute(),
        execution.paths.drift_reference.absolute(),
        execution.paths.drift_audit.absolute(),
        *(
            (execution.paths.transformer_bundle / name).absolute()
            for name in runner._PROBE_ARTIFACTS
        ),
    }
    for path in source_inputs:
        path.unlink()
    read_once = runner.source_runner._read_file_once

    def saved_read(path, **kwargs):
        assert path.absolute() not in source_inputs, "saved verifier reread probe input"
        return read_once(path, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("saved verification reread a source, fit, or executed a probe")

    monkeypatch.setattr(runner.source_runner, "_read_file_once", saved_read)
    for module, name in (
        (runner, "_read_probe_inputs"),
        (seed_probe_probes, "run_probe_stage"),
        (seed_probe_seeds, "run_seed_stage"),
        (seed_probe_seeds, "_fit"),
        (secondary_development, "_partition"),
        (character_transformer, "fit_character_transformer"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    assert (
        runner._verify_completion(
            execution.binding, execution.paths, producer_exit_code=0
        )
        == summary
    )


def test_saved_verification_authenticates_root_receipt_summary_and_stage_bytes(
    execution,
):
    runner._supervise(execution.binding, execution.paths)
    targets = (
        execution.paths.attempt / "outcome.json",
        execution.paths.attempt / "evidence/stage-summaries.json",
        execution.paths.attempt / "probes/evidence/artifact-manifest.json",
        execution.paths.attempt / "probes/score-row-00-000001.json",
        execution.paths.public_summary,
    )
    for target in targets:
        original = target.read_bytes()
        target.write_bytes(b"{}")
        with pytest.raises(Exception):
            runner._verify_completion(
                execution.binding, execution.paths, producer_exit_code=0
            )
        target.write_bytes(original)
    runner._verify_completion(execution.binding, execution.paths, producer_exit_code=0)


def test_saved_verification_rejects_unexpected_files_at_every_inventory_boundary(
    execution,
):
    runner._supervise(execution.binding, execution.paths)
    directories = (
        execution.paths.attempt / "probes",
        execution.paths.attempt / "probes/evidence",
        execution.paths.attempt,
        execution.paths.attempt / "evidence",
    )
    for directory in directories:
        unexpected = directory / "unexpected.json"
        unexpected.write_bytes(b"{}")
        with pytest.raises(Exception):
            runner._verify_completion(
                execution.binding, execution.paths, producer_exit_code=0
            )
        unexpected.unlink()
    runner._verify_completion(execution.binding, execution.paths, producer_exit_code=0)


def test_saved_verification_rejects_authenticated_public_stage_summary_tamper(
    execution,
):
    runner._supervise(execution.binding, execution.paths)
    public = _load(execution.paths.public_summary)
    public["probes"] = public["retained_seed_audit"]
    public_bytes = execution_receipt._json_bytes(public, "public_summary")
    execution.paths.public_summary.write_bytes(public_bytes)
    outcome_path = execution.paths.attempt / "outcome.json"
    outcome = _load(outcome_path)
    outcome["public_summary_sha256"] = sha256(public_bytes).hexdigest()
    outcome_path.write_bytes(execution_receipt._json_bytes(outcome, "outcome"))
    with pytest.raises(runner.ProbeCorrectionRunError, match="root_summary_mismatch"):
        runner._verify_completion(
            execution.binding, execution.paths, producer_exit_code=0
        )


def test_completed_root_and_child_records_are_create_only(execution):
    runner._supervise(execution.binding, execution.paths)
    root = execution_receipt.Attempt(
        execution.paths.attempt,
        sha256((execution.paths.attempt / "reservation.json").read_bytes()).hexdigest(),
    )
    child = execution_receipt.Attempt(
        execution.paths.attempt / "probes",
        sha256(
            (execution.paths.attempt / "probes/reservation.json").read_bytes()
        ).hexdigest(),
    )
    for attempt in (root, child):
        with pytest.raises(execution_receipt.ExecutionReceiptError):
            runner._record(attempt, "late.json", b"{}")


def test_failure_finalization_serializes_against_late_record(tmp_path, monkeypatch):
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"invented": True}
    )
    claim_entered = threading.Event()
    release_claim = threading.Event()
    late_finished = threading.Event()
    real_claim = execution_receipt._claim
    finalizer_errors = []
    late_errors = []
    lock_attempts, lock_blocked = _instrument_lock_attempts(
        monkeypatch, "late-record-contender"
    )

    def paused_claim(*args, **kwargs):
        claim_entered.set()
        assert release_claim.wait(2)
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(execution_receipt, "_claim", paused_claim)

    def finalize():
        try:
            runner._failure(attempt, "probes", RuntimeError("invented"))
        except Exception as error:  # pragma: no cover - asserted below
            finalizer_errors.append(error)

    def install_late():
        try:
            runner._record(attempt, "late.json", b"{}")
        except Exception as error:
            late_errors.append(error)
        finally:
            late_finished.set()

    finalizer = threading.Thread(target=finalize)
    finalizer.start()
    assert claim_entered.wait(2)
    late = threading.Thread(target=install_late, name="late-record-contender")
    late.start()
    try:
        assert lock_attempts["late-record-contender"].wait(2)
        assert lock_blocked["late-record-contender"].wait(2)
        assert not late_finished.is_set()
    finally:
        release_claim.set()
        finalizer.join(2)
        late.join(2)
    assert not finalizer.is_alive()
    assert not late.is_alive()
    assert finalizer_errors == []
    assert len(late_errors) == 1
    assert isinstance(late_errors[0], execution_receipt.ExecutionReceiptError)
    assert not (attempt.directory / "late.json").exists()


def test_failure_detects_attempt_inode_swap_before_reporting_success(
    tmp_path, monkeypatch
):
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"invented": True}
    )
    displaced = tmp_path / "displaced-attempt"
    real_install = execution_receipt._install_record

    def install_then_swap(directory, name, content):
        real_install(directory, name, content)
        if name == "outcome.json":
            attempt.directory.rename(displaced)
            attempt.directory.mkdir()

    monkeypatch.setattr(execution_receipt, "_install_record", install_then_swap)
    with pytest.raises(
        execution_receipt.ExecutionReceiptError,
        match="directory identity changed during publication",
    ):
        runner._failure(attempt, "probes", RuntimeError("invented"))
    assert not (attempt.directory / "outcome.json").exists()
    assert (displaced / "outcome.json").is_file()


def test_mutation_guard_detects_inode_swap_on_exceptional_body_exit(tmp_path):
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"invented": True}
    )
    displaced = tmp_path / "displaced-attempt"

    with pytest.raises(
        execution_receipt.ExecutionReceiptError,
        match="directory identity changed during publication",
    ):
        with runner._mutation_guard(attempt):
            attempt.directory.rename(displaced)
            attempt.directory.mkdir()
            raise RuntimeError("private body failure")
    assert not (attempt.directory / "outcome.json").exists()


def test_flock_serializes_same_inode_and_not_distinct_inode(tmp_path, monkeypatch):
    first = execution_receipt.reserve_attempt(
        tmp_path / "first", identity={"invented": "first"}
    )
    second = execution_receipt.reserve_attempt(
        tmp_path / "second", identity={"invented": "second"}
    )
    attempts, blocked = _instrument_lock_attempts(
        monkeypatch, "same-inode-contender", "distinct-inode-contender"
    )
    same_finished = threading.Event()
    distinct_finished = threading.Event()
    errors = []

    def acquire(attempt, finished):
        try:
            with runner._mutation_guard(attempt):
                pass
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)
        finally:
            finished.set()

    same = threading.Thread(
        target=acquire,
        args=(first, same_finished),
        name="same-inode-contender",
    )
    distinct = threading.Thread(
        target=acquire,
        args=(second, distinct_finished),
        name="distinct-inode-contender",
    )
    with runner._mutation_guard(first):
        same.start()
        assert attempts["same-inode-contender"].wait(2)
        assert blocked["same-inode-contender"].wait(2)
        assert not same_finished.is_set()
        distinct.start()
        assert attempts["distinct-inode-contender"].wait(2)
        assert distinct_finished.wait(2)
        assert not blocked["distinct-inode-contender"].is_set()
    assert same_finished.wait(2)
    same.join(2)
    distinct.join(2)
    assert not same.is_alive()
    assert not distinct.is_alive()
    assert errors == []


def test_failure_finalization_serializes_across_processes(tmp_path, monkeypatch):
    attempt = execution_receipt.reserve_attempt(
        tmp_path / "attempt", identity={"invented": True}
    )
    context = multiprocessing.get_context("fork")
    claim_entered = context.Event()
    release_claim = context.Event()
    late_finished = threading.Event()
    real_claim = execution_receipt._claim
    late_errors = []
    lock_attempts, lock_blocked = _instrument_lock_attempts(
        monkeypatch, "process-late-record-contender"
    )

    def paused_claim(*args, **kwargs):
        claim_entered.set()
        assert release_claim.wait(2)
        return real_claim(*args, **kwargs)

    monkeypatch.setattr(execution_receipt, "_claim", paused_claim)

    def finalize():
        runner._failure(attempt, "probes", RuntimeError("invented"))

    def install_late():
        try:
            runner._record(attempt, "late.json", b"{}")
        except Exception as error:
            late_errors.append(error)
        finally:
            late_finished.set()

    finalizer = context.Process(target=finalize)
    finalizer.start()
    assert claim_entered.wait(2)
    late = threading.Thread(target=install_late, name="process-late-record-contender")
    late.start()
    try:
        assert lock_attempts["process-late-record-contender"].wait(2)
        assert lock_blocked["process-late-record-contender"].wait(2)
        assert not late_finished.is_set()
    finally:
        release_claim.set()
        finalizer.join(2)
        late.join(2)
    assert finalizer.exitcode == 0
    assert not late.is_alive()
    assert len(late_errors) == 1
    assert isinstance(late_errors[0], execution_receipt.ExecutionReceiptError)
    assert not (attempt.directory / "late.json").exists()


def test_root_finalization_serializes_against_child_reservation(execution, monkeypatch):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    claim_entered = threading.Event()
    release_claim = threading.Event()
    real_claim = execution_receipt._claim
    finalizer_errors = []
    worker_errors = []
    lock_attempts, lock_blocked = _instrument_lock_attempts(
        monkeypatch, "child-reservation-contender"
    )

    def paused_claim(attempt, directory, operation):
        if attempt.directory == parent.directory:
            claim_entered.set()
            assert release_claim.wait(2)
        return real_claim(attempt, directory, operation)

    monkeypatch.setattr(execution_receipt, "_claim", paused_claim)

    def finalize():
        try:
            runner._failure(parent, "retained_seed_audit", RuntimeError("invented"))
        except Exception as error:  # pragma: no cover - asserted below
            finalizer_errors.append(error)

    def run_worker():
        try:
            runner._run_worker(
                execution.binding, execution.paths, "retained_seed_audit"
            )
        except Exception as error:
            worker_errors.append(error)

    finalizer = threading.Thread(target=finalize)
    finalizer.start()
    assert claim_entered.wait(2)
    worker = threading.Thread(target=run_worker, name="child-reservation-contender")
    worker.start()
    try:
        assert lock_attempts["child-reservation-contender"].wait(2)
        assert lock_blocked["child-reservation-contender"].wait(2)
        assert not (parent.directory / "retained_seed_audit").exists()
        assert worker.is_alive()
    finally:
        release_claim.set()
        finalizer.join(2)
        worker.join(2)
    assert not finalizer.is_alive()
    assert not worker.is_alive()
    assert finalizer_errors == []
    assert len(worker_errors) == 1
    assert isinstance(worker_errors[0], runner.ProbeCorrectionRunError)
    assert not (parent.directory / "retained_seed_audit").exists()


def test_child_completion_serializes_against_late_record(execution, monkeypatch):
    execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    claim_entered = threading.Event()
    release_claim = threading.Event()
    late_finished = threading.Event()
    real_claim = execution_receipt._claim
    worker_errors = []
    late_errors = []
    lock_attempts, lock_blocked = _instrument_lock_attempts(
        monkeypatch, "child-late-contender"
    )

    def paused_claim(attempt, directory, operation):
        if (
            attempt.directory.name == "retained_seed_audit"
            and operation == "completion"
        ):
            claim_entered.set()
            assert release_claim.wait(2)
        return real_claim(attempt, directory, operation)

    monkeypatch.setattr(execution_receipt, "_claim", paused_claim)

    def run_worker():
        try:
            runner._run_worker(
                execution.binding, execution.paths, "retained_seed_audit"
            )
        except Exception as error:  # pragma: no cover - asserted below
            worker_errors.append(error)

    worker = threading.Thread(target=run_worker)
    worker.start()
    assert claim_entered.wait(2)
    child_path = execution.paths.attempt / "retained_seed_audit"
    child = execution_receipt.Attempt(
        child_path,
        sha256((child_path / "reservation.json").read_bytes()).hexdigest(),
    )

    def install_late():
        try:
            runner._record(child, "late.json", b"{}")
        except Exception as error:
            late_errors.append(error)
        finally:
            late_finished.set()

    late = threading.Thread(target=install_late, name="child-late-contender")
    late.start()
    try:
        assert lock_attempts["child-late-contender"].wait(2)
        assert lock_blocked["child-late-contender"].wait(2)
        assert not late_finished.is_set()
    finally:
        release_claim.set()
        worker.join(2)
        late.join(2)
    assert not worker.is_alive()
    assert not late.is_alive()
    assert worker_errors == []
    assert len(late_errors) == 1
    assert isinstance(late_errors[0], execution_receipt.ExecutionReceiptError)
    assert not (child_path / "late.json").exists()


def test_root_finalization_after_computation_prevents_child_completion(
    execution, monkeypatch
):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    computed = threading.Event()
    release_worker = threading.Event()
    worker_errors = []
    audit = seed_probe_audit.audit_retained_seed_stages

    def compute(*args, **kwargs):
        result = audit(*args, **kwargs)
        computed.set()
        return result

    def recheck(_binding):
        if computed.is_set():
            assert release_worker.wait(2)

    monkeypatch.setattr(seed_probe_audit, "audit_retained_seed_stages", compute)
    monkeypatch.setattr(runner, "recheck_seed_probe_correction", recheck)

    def run_worker():
        try:
            runner._run_worker(
                execution.binding, execution.paths, "retained_seed_audit"
            )
        except Exception as error:
            worker_errors.append(error)

    worker = threading.Thread(target=run_worker)
    worker.start()
    assert computed.wait(2)
    runner._failure(parent, "retained_seed_audit", RuntimeError("invented"))
    release_worker.set()
    worker.join(2)
    assert not worker.is_alive()
    assert len(worker_errors) == 1
    assert isinstance(worker_errors[0], runner.ProbeCorrectionRunError)
    assert _load(parent.directory / "outcome.json")["status"] == "failed"
    assert not (parent.directory / "retained_seed_audit.json").exists()


def test_worker_publication_guard_order_is_parent_then_child(execution, monkeypatch):
    parent = execution_receipt.reserve_attempt(
        execution.paths.attempt, identity=runner._identity(execution.binding)
    )
    guard = runner._mutation_guard
    events = []

    @contextmanager
    def observed(attempt):
        label = (
            "parent"
            if attempt.directory == parent.directory
            else attempt.directory.name
        )
        events.append(("enter", label))
        try:
            with guard(attempt) as directory:
                yield directory
        finally:
            events.append(("exit", label))

    monkeypatch.setattr(runner, "_mutation_guard", observed)
    runner._run_worker(execution.binding, execution.paths, "retained_seed_audit")
    assert events == [
        ("enter", "parent"),
        ("exit", "parent"),
        ("enter", "parent"),
        ("enter", "retained_seed_audit"),
        ("exit", "retained_seed_audit"),
        ("exit", "parent"),
    ]


def test_root_publication_guard_order_is_parent_then_both_stages(
    execution, monkeypatch
):
    guard = runner._mutation_guard
    events = []

    @contextmanager
    def observed(attempt):
        label = (
            "parent"
            if attempt.directory == execution.paths.attempt
            else attempt.directory.name
        )
        events.append(("enter", label))
        try:
            with guard(attempt) as directory:
                yield directory
        finally:
            events.append(("exit", label))

    monkeypatch.setattr(runner, "_mutation_guard", observed)
    runner._supervise(execution.binding, execution.paths)
    assert events[-6:] == [
        ("enter", "parent"),
        ("enter", "retained_seed_audit"),
        ("enter", "probes"),
        ("exit", "probes"),
        ("exit", "retained_seed_audit"),
        ("exit", "parent"),
    ]


def test_root_completion_serializes_against_late_record(execution, monkeypatch):
    claim_entered = threading.Event()
    release_claim = threading.Event()
    late_finished = threading.Event()
    real_claim = execution_receipt._claim
    supervisor_errors = []
    late_errors = []
    lock_attempts, lock_blocked = _instrument_lock_attempts(
        monkeypatch, "root-late-contender"
    )

    def paused_claim(attempt, directory, operation):
        if attempt.directory == execution.paths.attempt and operation == "completion":
            claim_entered.set()
            assert release_claim.wait(2)
        return real_claim(attempt, directory, operation)

    monkeypatch.setattr(execution_receipt, "_claim", paused_claim)

    def supervise():
        try:
            runner._supervise(execution.binding, execution.paths)
        except Exception as error:  # pragma: no cover - asserted below
            supervisor_errors.append(error)

    supervisor = threading.Thread(target=supervise)
    supervisor.start()
    assert claim_entered.wait(8)
    parent = execution_receipt.Attempt(
        execution.paths.attempt,
        sha256((execution.paths.attempt / "reservation.json").read_bytes()).hexdigest(),
    )

    def install_late():
        try:
            runner._record(parent, "late.json", b"{}")
        except Exception as error:
            late_errors.append(error)
        finally:
            late_finished.set()

    late = threading.Thread(target=install_late, name="root-late-contender")
    late.start()
    try:
        assert lock_attempts["root-late-contender"].wait(2)
        assert lock_blocked["root-late-contender"].wait(2)
        assert not late_finished.is_set()
    finally:
        release_claim.set()
        supervisor.join(2)
        late.join(2)
    assert not supervisor.is_alive()
    assert not late.is_alive()
    assert supervisor_errors == []
    assert len(late_errors) == 1
    assert isinstance(late_errors[0], execution_receipt.ExecutionReceiptError)
    assert execution.paths.public_summary.is_file()
    assert not (execution.paths.attempt / "late.json").exists()
