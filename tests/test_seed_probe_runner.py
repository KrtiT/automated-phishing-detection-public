import json
import subprocess
import sys
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_seed_probe_seeds import sample  # noqa: F401

from automated_phishing_detection import execution_receipt as receipt
from automated_phishing_detection import seed_probe_runner as runner


def parent(tmp_path):
    return receipt.reserve_attempt(tmp_path / "attempt", identity={"fixture": True})


def test_record_is_create_only_and_rejects_finalized_attempt(tmp_path):
    attempt = parent(tmp_path)
    runner._record(attempt, "epoch-001.json", b"first")
    with pytest.raises(receipt.ExecutionReceiptError):
        runner._record(attempt, "epoch-001.json", b"replacement")
    assert (attempt.directory / "epoch-001.json").read_bytes() == b"first"
    receipt.record_failure(attempt, stage="fixture", error_type="ValueError")
    with pytest.raises(receipt.ExecutionReceiptError):
        runner._record(attempt, "epoch-002.json", b"late")


@pytest.mark.parametrize(
    "name",
    ["../escape", "reservation.json", "outcome.json", "evidence", "finalize.claim"],
)
def test_record_rejects_reserved_or_unsafe_names(tmp_path, name):
    with pytest.raises(runner.SeedProbeRunError):
        runner._record(parent(tmp_path), name, b"payload")


def test_launch_retains_actual_nonzero_exit_without_output_text(tmp_path):
    attempt = parent(tmp_path)
    code = runner._launch(
        attempt,
        "seed_43",
        [sys.executable, "-c", "import sys; print('private-url'); sys.exit(7)"],
    )
    assert code == 7
    content = (attempt.directory / "seed_43-process.json").read_bytes()
    record = json.loads(content)
    assert record["exit_code"] == 7
    assert record["stdout_sha256"] == sha256(b"private-url\n").hexdigest()
    assert b"private-url" not in content


def test_launch_retains_os_error_as_launch_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("private-path")

    monkeypatch.setattr(subprocess, "run", fail)
    attempt = parent(tmp_path)
    with pytest.raises(runner.SeedProbeRunError, match="worker_launch_failed"):
        runner._launch(attempt, "probes", ["unused"])
    record = json.loads((attempt.directory / "probes-process.json").read_bytes())
    assert record["status"] == "worker_launch_failed"
    assert record["exit_code"] is None
    assert "private-path" not in json.dumps(record)


def test_failure_records_accepted_prefix_and_unattempted_tail(tmp_path):
    attempt = parent(tmp_path)
    runner._failure(
        attempt,
        "seed_44",
        ValueError("private-input"),
        accepted=["seed_42_calibration", "seed_43"],
        check="fit",
    )
    details = json.loads((attempt.directory / "failure-details.json").read_bytes())
    assert details["completed_stages"] == ["seed_42_calibration", "seed_43"]
    assert details["failed_stage"] == "seed_44"
    assert details["unattempted_stages"] == ["seed_45", "seed_46", "probes"]
    assert details["check"] == "fit"
    assert "private-input" not in json.dumps(details)


def test_failure_does_not_copy_unknown_exception_name_or_check(tmp_path):
    PrivateName = type("PrivateCustomer", (Exception,), {})
    attempt = parent(tmp_path)
    runner._failure(
        attempt, "probes", PrivateName("secret"), accepted=[], check="private-url"
    )
    content = (attempt.directory / "failure-details.json").read_bytes()
    assert b"PrivateCustomer" not in content
    assert b"private-url" not in content
    assert json.loads(content)["check"] == "unclassified_check"


def test_supervisor_stops_without_launching_remaining_workers(tmp_path, monkeypatch):
    binding = SimpleNamespace(base=SimpleNamespace(root=tmp_path / "checkout"))
    paths = SimpleNamespace(
        attempt=tmp_path / "attempt", public_summary=tmp_path / "summary.json"
    )
    launches = []
    monkeypatch.setattr(runner, "recheck_seed_probe_binding", lambda value: None)
    monkeypatch.setattr(runner, "_outside_outputs", lambda *args: None)
    monkeypatch.setattr(runner, "_identity", lambda value: {"fixture": True})
    monkeypatch.setattr(runner, "_command", lambda *args: ["fixture"])
    monkeypatch.setattr(
        runner, "_launch", lambda attempt, stage, command: launches.append(stage) or 7
    )
    with pytest.raises(runner.SeedProbeRunError):
        runner._supervise(binding, paths)
    assert launches == ["seed_42_calibration"]
    assert not paths.public_summary.exists()
    assert json.loads((paths.attempt / "failure-details.json").read_bytes())[
        "unattempted_stages"
    ] == list(runner.STAGES[1:])


@pytest.fixture
def worker(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bundle = inputs / "bundle"
    bundle.mkdir()
    names = (
        "length-only.json",
        "logistic-l1.json",
        "transformer-weights.npz",
        "vocabulary.json",
        "transformer.json",
        "cascade.json",
        "gmm.json",
    )
    artifact_hashes = {}
    for name in names:
        content = ("invented " + name).encode()
        (bundle / name).write_bytes(content)
        artifact_hashes[name] = sha256(content).hexdigest()
    roles = {}
    for name in (
        "train",
        "validation",
        "suffix_rules",
        "drift_reference",
        "drift_audit",
    ):
        path = inputs / name
        path.write_bytes(name.encode())
        roles[name] = path
    binding = SimpleNamespace(
        base=SimpleNamespace(root=tmp_path / "checkout"),
        pins=SimpleNamespace(
            train_sha256=sha256(b"train").hexdigest(),
            validation_sha256=sha256(b"validation").hexdigest(),
            suffix_rules_sha256=sha256(b"suffix_rules").hexdigest(),
        ),
        primary_artifact_hashes=tuple(artifact_hashes.items()),
        training_reference_sha256=sha256(b"drift_reference").hexdigest(),
        validation_audit_sha256=sha256(b"drift_audit").hexdigest(),
    )
    paths = runner.SeedProbePaths(
        **roles,
        length_only=bundle / "length-only.json",
        logistic_l1=bundle / "logistic-l1.json",
        transformer_bundle=bundle,
        gmm=bundle / "gmm.json",
        attempt=tmp_path / "attempt",
        public_summary=tmp_path / "summary.json",
    )
    monkeypatch.setattr(runner, "recheck_seed_probe_binding", lambda value: None)
    monkeypatch.setattr(runner, "_identity", lambda value: {"fixture": True})
    parent = receipt.reserve_attempt(paths.attempt, identity={"fixture": True})
    return SimpleNamespace(binding=binding, paths=paths, parent=parent)


def test_worker_reserves_before_reading_inputs_and_never_reads_train_for_42(
    worker, monkeypatch
):
    reads = []
    original = runner.source_runner._read_file_once

    def read(path, **kwargs):
        if path.is_relative_to(worker.paths.validation.parent):
            assert (
                worker.paths.attempt / runner.STAGES[0] / "reservation.json"
            ).exists()
            reads.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(runner.source_runner, "_read_file_once", read)
    monkeypatch.setattr(
        runner,
        "_compute_stage",
        lambda binding, stage, inputs, retain: (
            {"fixture.json": b"{}"},
            {"fixture": True},
        ),
    )
    runner._run_worker(worker.binding, worker.paths, runner.STAGES[0])
    assert worker.paths.train not in reads
    assert len(reads) == len(set(reads)) == 5
    assert (worker.paths.attempt / "seed_42_calibration.json").exists()


@pytest.mark.parametrize("finalized", ["failed", "claimed"])
def test_worker_refuses_finalized_root_before_child_or_inputs(
    worker, monkeypatch, finalized
):
    if finalized == "failed":
        receipt.record_failure(worker.parent, stage="fixture", error_type="ValueError")
    else:
        (worker.paths.attempt / "finalize.claim").write_bytes(b"{}")
    monkeypatch.setattr(
        runner,
        "_read_inputs",
        lambda *args: pytest.fail("read after root finalization"),
    )
    with pytest.raises(runner.SeedProbeRunError, match="root_finalized"):
        runner._run_worker(worker.binding, worker.paths, runner.STAGES[0])
    assert not (worker.paths.attempt / runner.STAGES[0]).exists()


def test_worker_refuses_out_of_order_stage_before_child_or_inputs(worker, monkeypatch):
    monkeypatch.setattr(
        runner, "_read_inputs", lambda *args: pytest.fail("read out of order")
    )
    with pytest.raises(runner.SeedProbeRunError):
        runner._run_worker(worker.binding, worker.paths, "seed_43")
    assert not (worker.paths.attempt / "seed_43").exists()


def test_worker_failure_preserves_checkpoint_and_no_success_marker(worker, monkeypatch):
    def compute(binding, stage, inputs, retain):
        retain("best-001.npz", b"fitted-state")
        raise ValueError("private detail")

    monkeypatch.setattr(runner, "_compute_stage", compute)
    with pytest.raises(runner.SeedProbeRunError):
        runner._run_worker(worker.binding, worker.paths, runner.STAGES[0])
    child = worker.paths.attempt / runner.STAGES[0]
    assert (child / "best-001.npz").read_bytes() == b"fitted-state"
    assert json.loads((child / "outcome.json").read_bytes())["status"] == "failed"
    assert not (worker.paths.attempt / "seed_42_calibration.json").exists()


def test_bound_read_rejects_changed_input(tmp_path):
    path = tmp_path / "invented"
    path.write_bytes(b"changed")
    with pytest.raises(runner.SeedProbeRunError, match="input_hash_mismatch"):
        runner._bound_input(path, sha256(b"expected").hexdigest())


def test_snapshot_detects_auxiliary_mutation(worker, monkeypatch):
    def compute(binding, stage, inputs, retain):
        retain("epoch-001.json", b"first")
        return {"fixture.json": b"{}"}, {"fixture": True}

    monkeypatch.setattr(runner, "_compute_stage", compute)
    runner._run_worker(worker.binding, worker.paths, runner.STAGES[0])
    runner._launch(worker.parent, runner.STAGES[0], [sys.executable, "-c", "pass"])
    with pytest.raises(runner.SeedProbeRunError, match="stage_output_changed"):
        with runner._stage_snapshot(worker.parent, runner.STAGES[0]):
            (worker.paths.attempt / runner.STAGES[0] / "epoch-001.json").write_bytes(
                b"later"
            )


def test_snapshot_rejects_extra_auxiliary_file(worker, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_compute_stage",
        lambda *args: ({"fixture.json": b"{}"}, {"fixture": True}),
    )
    runner._run_worker(worker.binding, worker.paths, runner.STAGES[0])
    (worker.paths.attempt / runner.STAGES[0] / "extra.json").write_bytes(b"{}")
    with pytest.raises(completion_error()):
        with runner._stage_snapshot(worker.parent, runner.STAGES[0]):
            pytest.fail("accepted extra file")


def test_snapshot_rejects_changed_final_evidence_before_yield(worker, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_compute_stage",
        lambda *args: ({"fixture.json": b"{}"}, {"fixture": True}),
    )
    runner._run_worker(worker.binding, worker.paths, runner.STAGES[0])
    runner._launch(worker.parent, runner.STAGES[0], [sys.executable, "-c", "pass"])
    (worker.paths.attempt / runner.STAGES[0] / "evidence/fixture.json").write_bytes(
        b'{"changed":true}'
    )
    with pytest.raises(runner.SeedProbeRunError, match="private_hash_mismatch"):
        with runner._stage_snapshot(worker.parent, runner.STAGES[0]):
            pytest.fail("unverified final evidence reached caller")


def test_dependency_evidence_is_not_read_before_child_reservation(worker, monkeypatch):
    for previous in (runner.STAGES[0],):
        (worker.paths.attempt / previous).mkdir()
        (worker.paths.attempt / f"{previous}.json").write_bytes(b"{}")
        (worker.paths.attempt / f"{previous}-process.json").write_bytes(b"{}")

    def prior(binding, parent, stage):
        assert (worker.paths.attempt / stage / "reservation.json").is_file()
        raise runner.SeedProbeRunError("process_observation_mismatch")

    monkeypatch.setattr(runner, "_prior_observations", prior)
    monkeypatch.setattr(
        runner,
        "_read_inputs",
        lambda *args: pytest.fail("read after invalid prior observation"),
    )
    with pytest.raises(runner.SeedProbeRunError):
        runner._run_worker(worker.binding, worker.paths, "seed_43")


def test_stopped_probe_progress_counts_only_contiguous_observed_records(tmp_path):
    attempt = parent(tmp_path)
    child = receipt.reserve_attempt(
        attempt.directory / "probes", identity={"fixture": True}
    )
    for position in (1, 2, 4):
        runner._record(child, f"score-row-00-{position:06d}.json", b"{}")
    runner._record(child, "stream-00.json", b"{}")
    progress = runner._observed_probe_progress(attempt)
    assert progress["completed_row_prefix"]["0"] == 2
    assert progress["observed_completed_stream_records"] == [0]
    assert progress["unaccepted_row_records_outside_prefix"] == 1
    assert (
        progress["scope"]
        == "installed_record_inventory_not_accepted_scientific_evidence"
    )
    assert progress["failed_row_position"] is None


def completion_error():
    from automated_phishing_detection.development_completion import (
        DevelopmentCompletionError,
    )

    return DevelopmentCompletionError


def test_terminal_failure_does_not_call_completed_stages_unattempted(tmp_path):
    attempt = parent(tmp_path)
    runner._failure(
        attempt,
        "finalization",
        ValueError(),
        accepted=list(runner.STAGES),
        check="final_binding",
    )
    details = json.loads((attempt.directory / "failure-details.json").read_bytes())
    assert details["failed_stage"] is None
    assert details["unattempted_stages"] == []
    assert details["completed_stages"] == list(runner.STAGES)


def test_saved_completion_refuses_nonzero_parent_exit_before_io(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runner.source_runner,
        "_read_file_once",
        lambda *args, **kwargs: pytest.fail("read before exit check"),
    )
    with pytest.raises(runner.SeedProbeRunError, match="producer_exit_not_successful"):
        runner._verify_completion(
            SimpleNamespace(), SimpleNamespace(), producer_exit_code=2
        )


@pytest.mark.parametrize("code", [0, 2, -9])
def test_launch_records_signal_exit_exactly(tmp_path, monkeypatch, code):
    attempt = parent(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=code, stdout=b"", stderr=b""
        ),
    )
    assert runner._launch(attempt, "seed_45", ["invented"]) == code
    assert (
        json.loads((attempt.directory / "seed_45-process.json").read_bytes())[
            "exit_code"
        ]
        == code
    )


def test_closed_metadata_binding_cannot_touch_paths(tmp_path, monkeypatch):
    from automated_phishing_detection import seed_probe_execution

    def closed(*args, **kwargs):
        raise seed_probe_execution.SeedProbeExecutionError("invalid_fixture")

    monkeypatch.setattr(seed_probe_execution, "bind_seed_probe_execution", closed)
    monkeypatch.setattr(
        runner,
        "_supervise",
        lambda *args: pytest.fail("supervised despite invalid binding"),
    )
    with pytest.raises(seed_probe_execution.SeedProbeExecutionError):
        runner.run_seed_probes(
            tmp_path,
            expected_revision="a" * 40,
            expected_profile_sha256="b" * 64,
            paths=object(),
        )


def test_specific_known_seed_failure_check_is_retained(tmp_path):
    from automated_phishing_detection.seed_probe_seeds import SeedStageError

    attempt = parent(tmp_path)
    runner._failure(
        attempt, "seed_43", SeedStageError("checkpoint_coverage_mismatch"), check="fit"
    )
    details = json.loads((attempt.directory / "failure-details.json").read_bytes())
    assert details["check"] == "checkpoint_coverage_mismatch"


def test_specific_runner_check_is_retained(tmp_path):
    attempt = parent(tmp_path)
    runner._failure(
        attempt,
        "seed_43",
        runner.SeedProbeRunError("input_hash_mismatch"),
        check="input_validation",
    )
    assert (
        json.loads((attempt.directory / "failure-details.json").read_bytes())["check"]
        == "input_hash_mismatch"
    )


@pytest.mark.parametrize("killed_stage", [None, "seed_43", "probes"])
def test_real_subprocess_sequence_and_killed_worker(
    tmp_path, monkeypatch, killed_stage
):
    from pathlib import Path

    binding = SimpleNamespace(base=SimpleNamespace(root=tmp_path / "checkout"))
    paths = runner.SeedProbePaths(
        **{name: tmp_path / name for name in runner.SeedProbePaths.__dataclass_fields__}
    )
    monkeypatch.setattr(runner, "recheck_seed_probe_binding", lambda value: None)
    monkeypatch.setattr(runner, "_outside_outputs", lambda *args: None)
    monkeypatch.setattr(runner, "_identity", lambda value: {"fixture": True})
    monkeypatch.setattr(
        runner,
        "_verify_science",
        lambda binding, paths, parent, stage, outputs, auxiliary: {"stage": stage},
    )

    def command(binding, paths, stage):
        code = f"""
import os, signal
from pathlib import Path
from types import SimpleNamespace
from automated_phishing_detection import seed_probe_runner as r
r.recheck_seed_probe_binding = lambda value: None
r._identity = lambda value: {{'fixture': True}}
r._read_inputs = lambda *args: {{}}
r._verify_science = lambda binding, paths, parent, stage, outputs, auxiliary: {{'stage': stage}}
def compute(binding, stage, inputs, retain):
    retain('checkpoint.json', b'{{}}')
    if stage == 'probes':
        retain('score-row-00-000001.json', b'{{}}')
    if stage == {killed_stage!r}:
        os.kill(os.getpid(), signal.SIGKILL)
    return {{'fixture.json': b'{{}}'}}, {{'stage': stage}}
r._compute_stage = compute
paths = r.SeedProbePaths(**{{name: Path(path) for name, path in { {name: str(value) for name, value in vars(paths).items()}!r}.items()}})
binding = SimpleNamespace(base=SimpleNamespace(root=Path({str(binding.base.root)!r})))
r._run_worker(binding, paths, {stage!r})
"""
        return [sys.executable, "-c", code]

    monkeypatch.setattr(runner, "_command", command)
    if killed_stage is not None:
        with pytest.raises(runner.SeedProbeRunError):
            runner._supervise(binding, paths)
        process = json.loads(
            (paths.attempt / f"{killed_stage}-process.json").read_bytes()
        )
        assert process["exit_code"] == -9
        assert (paths.attempt / killed_stage / "checkpoint.json").read_bytes() == b"{}"
        assert not paths.public_summary.exists()
        if killed_stage == "seed_43":
            assert not (paths.attempt / "seed_44").exists()
        else:
            details = json.loads((paths.attempt / "failure-details.json").read_bytes())
            assert details["probe_progress"]["completed_row_prefix"]["0"] == 1
            assert details["probe_progress"]["failed_row_position"] is None
            assert details["unattempted_stages"] == []
        return
    summary = runner._supervise(binding, paths)
    assert summary["worker_exit_codes"] == dict.fromkeys(runner.STAGES, 0)
    assert runner._verify_completion(binding, paths, producer_exit_code=0) == summary
    with pytest.raises(receipt.ExecutionReceiptError):
        runner._supervise(binding, paths)
    assert Path(paths.public_summary).is_file()


def test_seed_evidence_composes_with_real_receipts_and_input_reads(
    sample,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    from automated_phishing_detection import character_transformer as trainer

    binding = sample.binding
    binding.base = SimpleNamespace(root=tmp_path / "checkout")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bundle = inputs / "bundle"
    bundle.mkdir()
    for name, content in sample.artifacts.items():
        (bundle / name).write_bytes(content)
    paths = runner.SeedProbePaths(
        train=inputs / "train.jsonl",
        validation=inputs / "validation.jsonl",
        suffix_rules=inputs / "suffix.dat",
        length_only=inputs / "unused-length",
        logistic_l1=bundle / "logistic-l1.json",
        transformer_bundle=bundle,
        gmm=inputs / "unused-gmm",
        drift_reference=inputs / "unused-reference",
        drift_audit=inputs / "unused-audit",
        attempt=tmp_path / "attempt",
        public_summary=tmp_path / "summary.json",
    )
    paths.validation.write_bytes(sample.arguments["validation_bytes"])
    paths.suffix_rules.write_bytes(sample.arguments["suffix_rules_bytes"])
    monkeypatch.setattr(runner, "recheck_seed_probe_binding", lambda value: None)
    monkeypatch.setattr(runner, "_identity", lambda value: {"fixture": True})
    parent = receipt.reserve_attempt(paths.attempt, identity={"fixture": True})
    primary = runner._run_worker(binding, paths, runner.STAGES[0])
    runner._launch(parent, runner.STAGES[0], [sys.executable, "-c", "pass"])
    assert (
        runner._verify_stage(binding, paths, parent, runner.STAGES[0], observed_exit=0)
        == primary
    )
    assert not paths.train.exists()

    paths.train.write_bytes(sample.fixture["arguments"]["train_content"])
    monkeypatch.setattr(trainer, "_run_training_epoch", lambda *args: 0.25)
    monkeypatch.setattr(
        trainer, "_evaluate_validation", lambda *args: (1.0, sample.probabilities)
    )
    secondary = runner._run_worker(binding, paths, "seed_43")
    runner._launch(parent, "seed_43", [sys.executable, "-c", "pass"])
    assert (
        runner._verify_stage(binding, paths, parent, "seed_43", observed_exit=0)
        == secondary
    )
    assert (paths.attempt / "seed_43/restored-best.npz").is_file()


def test_probe_evidence_composes_with_real_receipts_and_input_reads(
    worker, monkeypatch
):
    from test_seed_probe_probes import fixture_data, synthetic_scorer

    from automated_phishing_detection import (
        character_transformer,
        probe_replay,
        seed_probe_probes,
    )

    data = fixture_data()
    data.binding.base.root = worker.binding.base.root
    paths, attempt = worker.paths, worker.parent
    for name, content in data.artifacts.items():
        (paths.transformer_bundle / name).write_bytes(content)
    for path, key in (
        (paths.validation, "validation_bytes"),
        (paths.suffix_rules, "suffix_rules_bytes"),
        (paths.drift_reference, "drift_reference_bytes"),
        (paths.drift_audit, "drift_audit_bytes"),
    ):
        path.write_bytes(data.arguments[key])
    # Prior-stage observations are synthetic; only the probe stage is under test.
    for stage in runner.STAGES[:-1]:
        receipt.reserve_attempt(paths.attempt / stage, identity={"fixture": True})
        runner._record(attempt, f"{stage}.json", b"{}")
        runner._launch(attempt, stage, [sys.executable, "-c", "pass"])

    monkeypatch.setattr(
        seed_probe_probes,
        "run_probe_stage",
        lambda *args, **kwargs: seed_probe_probes._run_probe_stage(
            *args, **kwargs, _fixture_cpu=True
        ),
    )
    monkeypatch.setattr(
        seed_probe_probes,
        "verify_probe_stage",
        lambda *args, **kwargs: seed_probe_probes._verify_probe_stage(
            *args, **kwargs, _fixture_cpu=True
        ),
    )
    calls, reads = [], []
    monkeypatch.setattr(probe_replay, "make_primary_scorer", synthetic_scorer(calls))
    monkeypatch.setattr(
        character_transformer,
        "fit_character_transformer",
        lambda *args, **kwargs: pytest.fail("probe stage attempted training"),
    )
    read_file, verifying = runner.source_runner._read_file_once, False

    def read(path, **kwargs):
        assert path != paths.train
        if path.is_relative_to(paths.validation.parent):
            assert not verifying, "saved verification reread a source input"
            assert (paths.attempt / "probes/reservation.json").is_file()
            reads.append(path)
        return read_file(path, **kwargs)

    monkeypatch.setattr(runner.source_runner, "_read_file_once", read)
    summary = runner._run_worker(data.binding, paths, "probes")
    assert len(calls) == 4 * 320
    assert len(reads) == len(set(reads)) == 11
    runner._launch(attempt, "probes", [sys.executable, "-c", "pass"])
    verifying = True
    assert (
        runner._verify_stage(data.binding, paths, attempt, "probes", observed_exit=0)
        == summary
    )
    assert len(calls) == 4 * 320
    child = paths.attempt / "probes"
    manifest = json.loads((child / "evidence/artifact-manifest.json").read_bytes())
    expected = {
        f"input-{name}"
        for name in (
            "length-only.json",
            "logistic-l1.json",
            "gmm.json",
            "transformer.json",
            "cascade.json",
            "vocabulary.json",
            "training-reference.json",
            "validation-audit.json",
        )
    }
    expected.update(f"stream-{index:02d}.json" for index in range(4))
    expected.update(
        f"score-row-{index:02d}-{position:06d}.json"
        for index in range(4)
        for position in range(1, 321)
    )
    assert set(manifest["auxiliary_sha256"]) == expected
    assert json.loads((child / "score-row-00-000001.json").read_bytes())["phase"] == (
        "scored_pre_routing"
    )
    assert json.loads((child / "stream-03.json").read_bytes())["phase"] == (
        "completed_stream"
    )
