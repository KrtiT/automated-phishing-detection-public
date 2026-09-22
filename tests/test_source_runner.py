"""Run the file/publication boundary on temporary invented evidence only."""

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_evaluation_producer import SOURCE, encoded, source_rows, synthetic_session

from automated_phishing_detection import evaluation_producer, execution_preflight
from automated_phishing_detection.bound_models import ArtifactPaths

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner():
    assert (ROOT / "src/automated_phishing_detection/source_runner.py").is_file(), (
        "missing authenticated source runner"
    )
    from automated_phishing_detection import source_runner

    return source_runner


@pytest.fixture
def inputs(tmp_path, runner, monkeypatch):
    root = tmp_path / "checkout"
    (root / "data").mkdir(parents=True)
    (root / "reports").mkdir()
    psl = tmp_path / "rules.dat"
    psl.write_bytes(b"com\nco.uk\n")
    partition = tmp_path / "group_test.jsonl"
    partition.write_bytes(encoded(source_rows()))
    source = json.loads((ROOT / "data/sources.json").read_bytes())
    source["phiusiil"]["csv_sha256"] = SOURCE
    source["public_suffix_list"]["sha256"] = sha256(psl.read_bytes()).hexdigest()
    source_bytes = json.dumps(source).encode()
    summary = json.loads(
        (ROOT / "reports/phiusiil-preparation-summary.json").read_bytes()
    )
    summary["declared_sources"] = source
    summary["source_spec_sha256"] = sha256(source_bytes).hexdigest()
    summary["splits"]["group_test"] = {
        "row_count": 4,
        "domain_count": 4,
        "class_counts": {"0": 2, "1": 2},
    }
    summary["output_hashes"]["group_test.jsonl"] = sha256(
        partition.read_bytes()
    ).hexdigest()
    contents = {
        "data/sources.json": source_bytes,
        "reports/phiusiil-preparation-summary.json": json.dumps(summary).encode(),
    }
    for path, data in contents.items():
        (root / path).write_bytes(data)
    binding = execution_preflight.ExecutionBinding(
        root,
        "c" * 40,
        "d" * 64,
        tuple((path, sha256(data).hexdigest()) for path, data in contents.items()),
        '{"fixture":true}',
    )
    paths = runner.InternalRunPaths(
        partition,
        psl,
        ArtifactPaths(*(tmp_path / name for name in ("length", "lr", "tf", "gmm"))),
        tmp_path / "attempt",
        tmp_path / "summary.json",
    )
    session, *_ = synthetic_session(evaluation_producer, monkeypatch)
    events = []

    @contextmanager
    def open_session(*args):
        events.append("enter")
        yield session
        events.append("exit")

    monkeypatch.setattr(runner, "open_bound_session", open_session)
    monkeypatch.setattr(
        runner, "recheck_binding", lambda *args: events.append("recheck")
    )
    return binding, paths, session, events


def test_incomplete_freeze_rejects_before_any_supplied_path_access(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs
    monkeypatch.setattr(runner, "bind_execution", lambda *a, **k: binding)

    def forbidden(*args, **kwargs):
        pytest.fail("closed entry reached an input or reservation")

    monkeypatch.setattr(runner, "_read_file_once", forbidden)
    monkeypatch.setattr(runner, "reserve_attempt", forbidden)
    monkeypatch.setattr(runner, "open_bound_session", forbidden)
    monkeypatch.setattr(runner, "_output_paths", forbidden)
    with pytest.raises(runner.SourceExecutionError, match="pre_access_freeze"):
        runner.run_internal_evaluation(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            paths=paths,
        )
    assert not paths.attempt.exists()


def test_single_read_reserved_before_access_and_published_after_teardown(
    runner, inputs, monkeypatch
):
    binding, paths, session, events = inputs
    read = runner._read_file_once
    reads = []

    def observed(path):
        if path in (paths.partition, paths.suffix_rules):
            assert (paths.attempt / "reservation.json").is_file()
            reads.append(path)
        return read(path)

    publish = runner.publish_completion

    def checked_publish(*args, **kwargs):
        assert events[-2:] == ["exit", "recheck"]
        return publish(*args, **kwargs)

    monkeypatch.setattr(runner, "_read_file_once", observed)
    monkeypatch.setattr(runner, "publish_completion", checked_publish)
    result = runner._run_bound_internal(binding, paths)
    assert reads == [paths.suffix_rules, paths.partition]
    assert len(session.scorer.urls) == 4
    assert result == paths.public_summary
    public = json.loads(result.read_bytes())
    assert public["source_binding"] == "authenticated_public_preparation"
    assert public["protected_evaluation_authorized"] is False
    assert (
        public["secondary"]["metrics"]["logistic_l1"]["counts"]["recall"]["denominator"]
        == 2
    )
    assert "example0.com" not in result.read_text()
    for name, digest in public["private_sha256"].items():
        assert (
            sha256((paths.attempt / "evidence" / name).read_bytes()).hexdigest()
            == digest
        )
    assert set(public["private_sha256"]) == {
        "predictions.jsonl",
        "manifests.json",
        "bindings.json",
        "secondary.json",
    }


@pytest.mark.parametrize("bad", ["partition", "suffix_rules"])
def test_bad_input_hash_records_failure_without_scoring(runner, inputs, bad):
    binding, paths, session, _ = inputs
    getattr(paths, bad).write_bytes(b"bad bytes")
    with pytest.raises(runner.SourceExecutionError):
        runner._run_bound_internal(binding, paths)
    assert session.scorer.urls == []
    assert not paths.public_summary.exists()
    outcome = json.loads((paths.attempt / "outcome.json").read_bytes())
    assert outcome["status"] == "failed"
    assert "bad bytes" not in json.dumps(outcome)


def test_duplicate_attempt_does_not_reopen_source(runner, inputs, monkeypatch):
    binding, paths, *_ = inputs
    runner._run_bound_internal(binding, paths)
    original = runner._read_file_once

    def guarded(path):
        assert path not in (paths.partition, paths.suffix_rules)
        return original(path)

    monkeypatch.setattr(runner, "_read_file_once", guarded)
    with pytest.raises(runner.SourceExecutionError):
        runner._run_bound_internal(binding, paths)


def test_model_loading_follows_reservation_and_precedes_partition_read(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs
    original = runner._read_file_once

    def guarded(path):
        assert path != paths.partition
        return original(path)

    @contextmanager
    def broken(*args):
        assert (paths.attempt / "reservation.json").is_file()
        raise ValueError("invalid synthetic artifact")
        yield

    monkeypatch.setattr(runner, "_read_file_once", guarded)
    monkeypatch.setattr(runner, "open_bound_session", broken)
    with pytest.raises(runner.SourceExecutionError, match="model_loading"):
        runner._run_bound_internal(binding, paths)
    assert not paths.public_summary.exists()


@pytest.mark.parametrize(
    "which", ["source_hash", "source_chain", "declared_sources", "duplicate_key"]
)
def test_bad_public_record_never_reserves_attempt(runner, inputs, which):
    binding, paths, _, _ = inputs
    path = binding.root / "reports/phiusiil-preparation-summary.json"
    data = json.loads(path.read_bytes())
    if which == "source_chain":
        data["source_spec_sha256"] = "e" * 64
    elif which == "declared_sources":
        data["declared_sources"]["phiusiil"]["csv_sha256"] = "e" * 64
    else:
        data["schema_version"] = 2
    content = json.dumps(data).encode()
    if which == "duplicate_key":
        content = content.replace(
            b'"schema_version": 2', b'"schema_version":1,"schema_version":1', 1
        )
    path.write_bytes(content)
    if which != "source_hash":
        pins = dict(binding.source_hashes)
        pins["reports/phiusiil-preparation-summary.json"] = sha256(content).hexdigest()
        binding = replace(binding, source_hashes=tuple(pins.items()))
    with pytest.raises(runner.SourceExecutionError, match="public_preflight"):
        runner._run_bound_internal(binding, paths)
    assert not paths.attempt.exists()


@pytest.mark.parametrize("destination", ["attempt", "public_summary"])
def test_outputs_cannot_dirty_authenticated_checkout(runner, inputs, destination):
    binding, paths, *_ = inputs
    paths = replace(paths, **{destination: binding.root / "new-output"})
    with pytest.raises(runner.SourceExecutionError, match="public_preflight"):
        runner._run_bound_internal(binding, paths)
    assert not (binding.root / "new-output").exists()


def test_publication_failure_is_not_replaced_by_failure_finalization(
    runner, inputs, monkeypatch
):
    binding, paths, *_ = inputs
    original = runner.publish_completion

    def fail_after_install(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("post-publication sync failure")

    def forbidden(*args, **kwargs):
        pytest.fail("attempted second finalization")

    monkeypatch.setattr(runner, "publish_completion", fail_after_install)
    monkeypatch.setattr(runner, "record_failure", forbidden)
    with pytest.raises(runner.SourceExecutionError, match="publication"):
        runner._run_bound_internal(binding, paths)
    assert paths.public_summary.is_file()
    assert (
        json.loads((paths.attempt / "outcome.json").read_bytes())["status"]
        == "completion_prepared"
    )


def test_fresh_process_entry_has_no_readiness_override(runner, tmp_path):
    script = ROOT / "scripts/run_internal_evaluation.py"
    assert script.is_file(), "missing fresh-process entry"
    result = subprocess.run(
        [sys.executable, str(script), "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--expected-revision" in result.stdout
    assert "--authorize" not in result.stdout and "--force" not in result.stdout
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(ROOT),
            "--expected-revision",
            "not-a-commit",
            "--expected-contract-sha256",
            "d" * 64,
            "--partition",
            str(tmp_path / "missing"),
            "--suffix-rules",
            str(tmp_path / "missing"),
            "--length-only",
            "missing",
            "--logistic-l1",
            "missing",
            "--transformer-bundle",
            "missing",
            "--gmm",
            "missing",
            "--attempt",
            str(tmp_path / "attempt"),
            "--public-summary",
            str(tmp_path / "summary"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "ExecutionPreflightError" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize("kind", ["in_place", "path_swap", "parent_swap"])
def test_changes_during_single_read_are_rejected(runner, tmp_path, monkeypatch, kind):
    parent = tmp_path / "input-dir"
    parent.mkdir()
    path = parent / "input"
    path.write_bytes(b"invented")
    inode = path.stat().st_ino
    original = runner.os.fstat
    calls = 0

    def changed(descriptor):
        nonlocal calls
        value = original(descriptor)
        if value.st_ino == inode:
            calls += 1
            if calls == 2:
                if kind == "in_place":
                    path.write_bytes(b"changed!")
                elif kind == "path_swap":
                    replacement = parent / "replacement"
                    replacement.write_bytes(b"invented")
                    os.replace(replacement, path)
                else:
                    parent.rename(tmp_path / "old-dir")
                    parent.mkdir()
                    path.write_bytes(b"invented")
                value = original(descriptor)
        return value

    monkeypatch.setattr(runner.os, "fstat", changed)
    with pytest.raises(runner.SourceExecutionError):
        runner._read_file_once(path)
    assert calls == 2


def test_final_binding_failure_prevents_publication(runner, inputs, monkeypatch):
    binding, paths, *_ = inputs
    count = 0

    def recheck(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise ValueError("changed runtime")

    monkeypatch.setattr(runner, "recheck_binding", recheck)
    with pytest.raises(runner.SourceExecutionError, match="final_binding"):
        runner._run_bound_internal(binding, paths)
    assert not paths.public_summary.exists()
    assert (
        json.loads((paths.attempt / "outcome.json").read_bytes())["status"] == "failed"
    )


def test_arbitrary_nested_producer_summary_is_not_forwarded(
    runner, inputs, monkeypatch
):
    binding, paths, *_ = inputs
    original = evaluation_producer.produce_internal_evidence

    def canary(*args):
        produced = original(*args)
        return replace(
            produced,
            public_summary={
                **produced.public_summary,
                "primary": {"secret": "private-url-canary"},
            },
        )

    monkeypatch.setattr(evaluation_producer, "produce_internal_evidence", canary)
    runner._run_bound_internal(binding, paths)
    assert "private-url-canary" not in paths.public_summary.read_text()


def test_parent_process_entry_is_closed_before_launch(runner, inputs, monkeypatch):
    binding, paths, *_ = inputs
    monkeypatch.setattr(runner, "bind_execution", lambda *a, **k: binding)

    def forbidden(*args, **kwargs):
        pytest.fail("closed entry launched a subprocess")

    monkeypatch.setattr(subprocess, "run", forbidden)
    assert hasattr(runner, "run_internal_process"), "missing process orchestration"
    with pytest.raises(runner.SourceExecutionError, match="pre_access_freeze"):
        runner.run_internal_process(
            binding.root,
            expected_revision=binding.revision,
            expected_contract_sha256=binding.contract_sha256,
            paths=paths,
        )


@pytest.mark.parametrize("exit_code", [0, 2, -9])
def test_parent_passes_actual_exit_to_independent_verifier(
    runner, inputs, monkeypatch, exit_code
):
    from automated_phishing_detection import source_completion

    binding, paths, *_ = inputs
    monkeypatch.setattr(runner, "bind_execution", lambda *a, **k: binding)
    monkeypatch.setattr(
        execution_preflight.ExecutionBinding,
        "protected_evaluation_ready",
        property(lambda self: True),
    )
    observed = []

    def child(command, **kwargs):
        observed.append(command)
        assert kwargs == {"capture_output": True, "check": False}
        return SimpleNamespace(
            returncode=exit_code,
            stdout=b"not completion proof",
            stderr=b"private diagnostic",
        )

    def verify(actual_binding, actual_paths, *, producer_exit_code):
        assert actual_binding is binding and actual_paths is paths
        assert producer_exit_code == exit_code
        if exit_code:
            raise source_completion.CompletionVerificationError("producer_exit")
        return {"verified": True}

    monkeypatch.setattr(subprocess, "run", child)
    monkeypatch.setattr(source_completion, "verify_internal_completion", verify)
    kwargs = dict(
        expected_revision=binding.revision,
        expected_contract_sha256=binding.contract_sha256,
        paths=paths,
    )
    if exit_code:
        with pytest.raises(source_completion.CompletionVerificationError):
            runner.run_internal_process(binding.root, **kwargs)
    else:
        assert runner.run_internal_process(binding.root, **kwargs) == {"verified": True}
    (command,) = observed
    assert command[:3] == [
        sys.executable,
        str(binding.root / "scripts/run_internal_evaluation.py"),
        "--worker",
    ]
    assert command[command.index("--partition") + 1] == str(paths.partition)
    assert command[command.index("--expected-revision") + 1] == binding.revision


def test_session_teardown_failure_never_publishes(runner, inputs, monkeypatch):
    binding, paths, session, _ = inputs

    @contextmanager
    def broken(*args):
        yield session
        raise RuntimeError("private-url-that-must-not-be-reported")

    monkeypatch.setattr(runner, "open_bound_session", broken)
    with pytest.raises(runner.SourceExecutionError) as caught:
        runner._run_bound_internal(binding, paths)
    assert "private-url" not in str(caught.value)
    assert not paths.public_summary.exists()
    assert (
        json.loads((paths.attempt / "outcome.json").read_bytes())["stage"] == "scoring"
    )


@pytest.mark.parametrize(
    "kind", ["symlink", "hardlink", "parent_symlink", "directory", "fifo"]
)
def test_single_read_rejects_aliases_and_nonregular_files(tmp_path, runner, kind):
    parent = tmp_path / "actual"
    parent.mkdir()
    target = parent / "data"
    target.write_bytes(b"invented")
    path = tmp_path / "input"
    if kind == "symlink":
        path.symlink_to(target)
    elif kind == "hardlink":
        os.link(target, path)
    elif kind == "parent_symlink":
        path.symlink_to(parent, target_is_directory=True)
        path = path / "data"
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises(runner.SourceExecutionError):
        runner._read_file_once(path)
