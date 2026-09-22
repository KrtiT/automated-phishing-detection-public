"""The development runner sees only temporary, invented partitions and models."""

import importlib.util
import json
import subprocess
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_fixed_cascade import _artifact
from test_secondary_development import _bytes, _fixture

from automated_phishing_detection import (
    execution_preflight,
    execution_receipt,
    gmm_monitor,
    secondary_development,
    secondary_tabular,
)

ROOT = Path(__file__).resolve().parents[1]
STEPS = (
    "drift",
    "formatting",
    "permutation_42",
    "permutation_43",
    "permutation_44",
    "permutation_45",
    "permutation_46",
    "random_forest",
)


@pytest.fixture
def runner():
    assert (
        ROOT / "src/automated_phishing_detection/development_runner.py"
    ).is_file(), "missing authenticated development runner"
    from automated_phishing_detection import development_runner

    return development_runner


@pytest.fixture
def inputs(tmp_path, runner, monkeypatch):
    data = _fixture(secondary_development, train_count=64, validation_count=12)
    values = data["arguments"]
    pins = values["pins"]
    logistic = _artifact(pins.baseline_contract_sha256)
    logistic["scaler"]["n_samples_seen"] = 64
    logistic["input_hashes"].update(
        train=pins.train_sha256,
        validation=pins.validation_sha256,
        preparation_summary=pins.preparation_summary_sha256,
    )
    contents = {
        "train": values["train_content"],
        "validation": data["validation_content"],
        "suffix_rules": values["suffix_rules"],
        "logistic_l1": _bytes(logistic),
        "gmm": gmm_monitor._canonical_json_bytes(values["gmm_state"]),
    }
    assert (
        sha256(contents["logistic_l1"]).hexdigest() == pins.logistic_l1_artifact_sha256
    )
    root = tmp_path / "checkout"
    root.mkdir()
    base = execution_preflight.ExecutionBinding(
        root, "a" * 40, "b" * 64, (), '{"fixture":true}'
    )
    binding = SimpleNamespace(
        base=base,
        profile_sha256="c" * 64,
        methods_sha256="d" * 64,
        pins=pins,
        preparation_bytes=values["preparation_summary"],
    )
    paths = runner.DevelopmentRunPaths(
        *(tmp_path / name for name in contents),
        tmp_path / "attempt",
        tmp_path / "public.json",
    )
    for name, content in contents.items():
        getattr(paths, name).write_bytes(content)
    events = []
    monkeypatch.setattr(
        runner, "recheck_development_binding", lambda _: events.append("recheck")
    )
    return binding, paths, data, events


def test_reserve_before_single_reads_and_freeze_training_before_validation(
    runner, inputs, monkeypatch
):
    binding, paths, _, events = inputs
    read = runner._read_file_once
    build = secondary_development.build_training_reference
    reference = None
    reads = []

    def observed_read(path):
        assert (paths.attempt / "reservation.json").is_file()
        assert (paths.attempt / "drift/reservation.json").is_file()
        if path == paths.validation:
            assert reference is not None
            assert reference.private_payload
        reads.append(path)
        return read(path)

    def observed_build(*args, **kwargs):
        nonlocal reference
        assert paths.validation not in reads
        reference = build(*args, **kwargs)
        return reference

    monkeypatch.setattr(runner, "_read_file_once", observed_read)
    monkeypatch.setattr(
        secondary_development, "build_training_reference", observed_build
    )
    result = runner._run_bound_development(binding, paths)
    assert result == paths.public_summary
    assert reads == [
        paths.suffix_rules,
        paths.logistic_l1,
        paths.gmm,
        paths.train,
        paths.validation,
    ]
    assert events == ["recheck", "recheck"]
    summary = json.loads(result.read_bytes())
    assert summary["status"] == "development_evidence_published"
    assert summary["protected_evaluation_authorized"] is False
    assert summary["execution"]["pins"] == asdict(binding.pins)
    assert [entry["member"] for entry in summary["members"]] == list(STEPS)
    for entry in summary["members"]:
        member = entry["member"]
        child = json.loads((paths.attempt / f"{member}.json").read_bytes())
        assert child == entry["summary"]
        assert (
            entry["public_summary_sha256"]
            == sha256((paths.attempt / f"{member}.json").read_bytes()).hexdigest()
        )
        assert (
            child["root_reservation_sha256"]
            == summary["execution"]["reservation_sha256"]
        )
        for name, digest in child["private_sha256"].items():
            assert (
                sha256(
                    (paths.attempt / member / "evidence" / name).read_bytes()
                ).hexdigest()
                == digest
            )
    assert "raw_url" not in result.read_text()
    assert ".example" not in result.read_text()


def test_fits_fixed_family_in_order_and_keeps_original_training_labels(
    runner, inputs, monkeypatch
):
    binding, paths, data, _ = inputs
    calls = []
    expected_train = tuple(row["is_phishing"] for row in data["train"])
    expected_validation = tuple(row["is_phishing"] for row in data["validation"])
    for name, role in (
        ("fit_formatting", "formatting"),
        ("fit_label_permutation", "permutation"),
        ("fit_random_forest", "random_forest"),
    ):
        fit = getattr(secondary_tabular, name)

        def observed(*args, _fit=fit, _role=role, **kwargs):
            assert args[1] == expected_train
            assert args[3] == expected_validation
            member = (
                f"permutation_{kwargs['seed']}" if _role == "permutation" else _role
            )
            assert (paths.attempt / member / "reservation.json").is_file()
            calls.append(member)
            return _fit(*args, **kwargs)

        monkeypatch.setattr(secondary_tabular, name, observed)
    runner._run_bound_development(binding, paths)
    assert calls == list(STEPS[1:])
    for member in STEPS[1:]:
        evidence = paths.attempt / member / "evidence"
        assert {path.name for path in evidence.iterdir()} == {
            "model.json",
            "validation-predictions.jsonl",
            "threshold.json",
            "scoring-audit.json",
        }
        rows = [
            json.loads(line)
            for line in (evidence / "validation-predictions.jsonl")
            .read_bytes()
            .splitlines()
        ]
        assert [row["record_id"] for row in rows] == [
            row["record_id"] for row in data["validation"]
        ]
        assert all(set(row) == {"record_id", "label", "probability"} for row in rows)
        model = secondary_tabular.load_secondary_model_bytes(
            (evidence / "model.json").read_bytes()
        )
        assert model.score_urls(
            tuple(row["raw_url"] for row in data["validation"])
        ) == tuple(row["probability"] for row in rows)


def test_failed_member_retains_prior_successes_and_unattempted_tail(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs
    fit = secondary_tabular.fit_label_permutation

    def broken(*args, seed):
        if seed == 43:
            assert (paths.attempt / "permutation_42.json").is_file()
            raise ValueError("https://private.example/secret")
        return fit(*args, seed=seed)

    monkeypatch.setattr(secondary_tabular, "fit_label_permutation", broken)
    with pytest.raises(
        runner.DevelopmentExecutionError, match="permutation_43"
    ) as caught:
        runner._run_bound_development(binding, paths)
    assert "private.example" not in str(caught.value)
    assert not paths.public_summary.exists()
    for member in STEPS[:3]:
        assert (paths.attempt / f"{member}.json").is_file()
        assert (
            json.loads((paths.attempt / member / "outcome.json").read_bytes())["status"]
            == "completion_prepared"
        )
    assert (
        json.loads((paths.attempt / "permutation_43/outcome.json").read_bytes())[
            "status"
        ]
        == "failed"
    )
    assert (
        json.loads((paths.attempt / "outcome.json").read_bytes())["stage"]
        == "permutation_43"
    )
    assert not any((paths.attempt / member).exists() for member in STEPS[4:])


@pytest.mark.parametrize(
    "name", ["train", "validation", "suffix_rules", "logistic_l1", "gmm"]
)
def test_input_hash_failure_leaves_reserved_failed_attempt(runner, inputs, name):
    binding, paths, _, _ = inputs
    getattr(paths, name).write_bytes(b"mismatched bytes")
    with pytest.raises(runner.DevelopmentExecutionError):
        runner._run_bound_development(binding, paths)
    assert not paths.public_summary.exists()
    assert (
        json.loads((paths.attempt / "drift/outcome.json").read_bytes())["status"]
        == "failed"
    )
    assert (
        json.loads((paths.attempt / "outcome.json").read_bytes())["status"] == "failed"
    )
    assert not (paths.attempt / "formatting").exists()


def test_existing_reservation_prevents_reads_and_does_not_overwrite(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs
    execution_receipt.reserve_attempt(paths.attempt, identity={"existing": True})
    before = (paths.attempt / "reservation.json").read_bytes()
    monkeypatch.setattr(
        runner,
        "_read_file_once",
        lambda _: pytest.fail("read after occupied reservation"),
    )
    with pytest.raises(runner.DevelopmentExecutionError):
        runner._run_bound_development(binding, paths)
    assert (paths.attempt / "reservation.json").read_bytes() == before
    assert not (paths.attempt / "outcome.json").exists()


def test_binding_failure_precedes_all_supplied_path_inspection(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs

    def rejected(*args, **kwargs):
        raise ValueError("development profile rejected")

    monkeypatch.setattr(runner, "bind_development_execution", rejected)
    monkeypatch.setattr(
        runner, "_read_file_once", lambda _: pytest.fail("read before binding")
    )
    monkeypatch.setattr(
        runner, "reserve_attempt", lambda *a, **k: pytest.fail("reserve before binding")
    )
    with pytest.raises(ValueError, match="profile rejected"):
        runner.run_development(
            binding.base.root,
            expected_revision=binding.base.revision,
            expected_profile_sha256=binding.profile_sha256,
            paths=paths,
        )
    assert not paths.attempt.exists()


def test_final_binding_failure_keeps_all_member_outputs_without_root_marker(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs
    calls = 0

    def recheck(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("changed checkout")

    monkeypatch.setattr(runner, "recheck_development_binding", recheck)
    with pytest.raises(runner.DevelopmentExecutionError, match="final_binding"):
        runner._run_bound_development(binding, paths)
    assert all((paths.attempt / f"{member}.json").exists() for member in STEPS)
    assert not paths.public_summary.exists()
    assert (
        json.loads((paths.attempt / "outcome.json").read_bytes())["stage"]
        == "final_binding"
    )


def test_publication_failure_is_never_finalized_twice(runner, inputs, monkeypatch):
    binding, paths, _, _ = inputs
    publish = runner.publish_completion

    def broken(attempt, **kwargs):
        result = publish(attempt, **kwargs)
        if attempt.directory.name == "formatting":
            raise OSError("post-publication failure")
        return result

    monkeypatch.setattr(runner, "publish_completion", broken)
    with pytest.raises(runner.DevelopmentExecutionError):
        runner._run_bound_development(binding, paths)
    assert (
        json.loads((paths.attempt / "formatting/outcome.json").read_bytes())["status"]
        == "completion_prepared"
    )
    assert (
        json.loads((paths.attempt / "outcome.json").read_bytes())["status"] == "failed"
    )
    assert not (paths.attempt / "permutation_42").exists()


def test_child_failure_record_error_still_consumes_root_attempt(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs
    record = runner.record_failure

    def broken_fit(*args):
        raise ValueError("invented fit failure")

    def broken_record(attempt, **kwargs):
        if attempt.directory.name == "formatting":
            raise OSError("invented storage failure")
        return record(attempt, **kwargs)

    monkeypatch.setattr(secondary_tabular, "fit_formatting", broken_fit)
    monkeypatch.setattr(runner, "record_failure", broken_record)
    with pytest.raises(
        runner.DevelopmentExecutionError, match="failure_record_incomplete"
    ):
        runner._run_bound_development(binding, paths)
    assert (paths.attempt / "drift.json").is_file()
    assert (paths.attempt / "formatting/reservation.json").is_file()
    assert not (paths.attempt / "formatting/outcome.json").exists()
    assert (
        json.loads((paths.attempt / "outcome.json").read_bytes())["status"] == "failed"
    )
    assert not (paths.attempt / "permutation_42").exists()


def test_interrupt_leaves_unfinished_reservations_without_claiming_failure(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs

    def interrupted(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(secondary_tabular, "fit_formatting", interrupted)
    with pytest.raises(KeyboardInterrupt):
        runner._run_bound_development(binding, paths)
    assert (paths.attempt / "drift.json").is_file()
    assert (paths.attempt / "formatting/reservation.json").is_file()
    assert not (paths.attempt / "outcome.json").exists()
    assert not (paths.attempt / "formatting/outcome.json").exists()
    assert not paths.public_summary.exists()
    with pytest.raises(runner.DevelopmentExecutionError):
        runner._run_bound_development(binding, paths)


def test_input_parent_swap_is_rejected_even_when_replacement_bytes_match(
    runner, inputs, monkeypatch, tmp_path
):
    binding, paths, _, _ = inputs
    parent = tmp_path / "input-directory"
    parent.mkdir()
    original_content = paths.train.read_bytes()
    paths.train.rename(parent / "train")
    paths = replace(paths, train=parent / "train")
    check = execution_receipt._Directory.check
    swapped = False

    def replace_parent(directory):
        nonlocal swapped
        if directory.path == parent and not swapped:
            swapped = True
            parent.rename(tmp_path / "old-input-directory")
            parent.mkdir()
            (parent / "train").write_bytes(original_content)
        return check(directory)

    monkeypatch.setattr(execution_receipt._Directory, "check", replace_parent)
    with pytest.raises(runner.DevelopmentExecutionError):
        runner._run_bound_development(binding, paths)
    assert swapped
    assert (
        json.loads((paths.attempt / "drift/outcome.json").read_bytes())["status"]
        == "failed"
    )
    assert not paths.public_summary.exists()


def test_score_metrics_match_known_answer_and_never_supply_fallback_cutoff(
    runner, monkeypatch
):
    rows = tuple(
        {
            "raw_url": f"https://invented-{i}.example/",
            "is_phishing": label,
            "record_id": str(i),
        }
        for i, label in enumerate((0, 1, 0, 1))
    )
    scores = (0.1, 0.8, 0.7, 0.2)
    fitted = secondary_tabular.SecondaryFitResult(
        b"invented model bytes",
        scores,
        '{"status":"target_not_met","threshold":null}',
        '{"fixture":true}',
    )
    monkeypatch.setattr(
        secondary_tabular, "fit_label_permutation", lambda *a, **k: fitted
    )
    _, public = runner._tabular("permutation_42", rows, rows)
    assert public["score_metrics"] == {
        "average_precision": pytest.approx(5 / 6),
        "roc_auc": 0.75,
    }
    assert public["analysis_role"] == "descriptive_secondary_not_primary"
    assert public["validation_threshold"]["threshold"] is None


def test_outputs_inside_checkout_rejected_before_private_reads(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs
    paths = replace(paths, attempt=binding.base.root / "attempt")
    monkeypatch.setattr(
        runner,
        "_read_file_once",
        lambda _: pytest.fail("input read for invalid outputs"),
    )
    with pytest.raises(runner.DevelopmentExecutionError):
        runner._run_bound_development(binding, paths)
    assert not paths.attempt.exists()


@pytest.mark.parametrize("exit_code", [0, 2, -9])
def test_process_uses_fresh_worker_and_forwards_actual_exit_code(
    runner, inputs, monkeypatch, exit_code
):
    binding, paths, _, _ = inputs
    monkeypatch.setattr(runner, "bind_development_execution", lambda *a, **k: binding)
    launched = []
    verified = []

    def launch(command, **kwargs):
        assert kwargs == {"capture_output": True, "check": False}
        launched.append(command)
        return SimpleNamespace(
            returncode=exit_code, stdout=b"private", stderr=b"private"
        )

    def verify(actual_binding, actual_paths, *, producer_exit_code):
        verified.append((actual_binding, actual_paths, producer_exit_code))
        return {"status": "fixture_verified"}

    monkeypatch.setattr(runner.subprocess, "run", launch)
    monkeypatch.setitem(
        sys.modules,
        "automated_phishing_detection.development_completion",
        SimpleNamespace(verify_development_completion=verify),
    )
    result = runner.run_development_process(
        binding.base.root,
        expected_revision=binding.base.revision,
        expected_profile_sha256=binding.profile_sha256,
        paths=paths,
    )
    assert result == {"status": "fixture_verified"}
    assert verified == [(binding, paths, exit_code)]
    command = launched[0]
    assert command[:3] == [
        sys.executable,
        str(binding.base.root / "scripts/run_secondary_development.py"),
        "--worker",
    ]
    for option, value in (
        ("--expected-revision", binding.base.revision),
        ("--expected-profile-sha256", binding.profile_sha256),
        ("--train", str(paths.train)),
        ("--validation", str(paths.validation)),
    ):
        assert command[command.index(option) + 1] == value


def test_worker_launch_failure_is_symbolic_and_does_not_claim_completion(
    runner, inputs, monkeypatch
):
    binding, paths, _, _ = inputs
    monkeypatch.setattr(runner, "bind_development_execution", lambda *a, **k: binding)

    def rejected(*args, **kwargs):
        raise OSError("private path")

    monkeypatch.setitem(
        sys.modules,
        "automated_phishing_detection.development_completion",
        SimpleNamespace(
            verify_development_completion=lambda *a, **k: pytest.fail(
                "completion claimed after launch failure"
            )
        ),
    )
    monkeypatch.setattr(runner.subprocess, "run", rejected)
    with pytest.raises(
        runner.DevelopmentExecutionError, match="^worker_launch_failed$"
    ):
        runner.run_development_process(
            binding.base.root,
            expected_revision=binding.base.revision,
            expected_profile_sha256=binding.profile_sha256,
            paths=paths,
        )


@pytest.mark.parametrize("worker", [False, True])
def test_cli_defaults_to_parent_and_prints_only_symbolic_errors(
    runner, inputs, monkeypatch, capsys, worker
):
    script_path = ROOT / "scripts/run_secondary_development.py"
    assert script_path.is_file(), "missing development process command"
    specification = importlib.util.spec_from_file_location(
        "development_command_fixture", script_path
    )
    script = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(script)
    binding, paths, _, _ = inputs
    calls = []

    def rejected(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("https://private.example/never-print")

    monkeypatch.setattr(
        script, "run_development" if worker else "run_development_process", rejected
    )
    argv = [
        "--repo-root",
        str(binding.base.root),
        "--expected-revision",
        binding.base.revision,
        "--expected-profile-sha256",
        binding.profile_sha256,
    ]
    for name in (
        "train",
        "validation",
        "suffix_rules",
        "logistic_l1",
        "gmm",
        "attempt",
        "public_summary",
    ):
        argv.extend(("--" + name.replace("_", "-"), str(getattr(paths, name))))
    if worker:
        argv.append("--worker")
    assert script.main(argv) == 2
    assert calls[0][1]["paths"] == paths
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "Development execution stopped: ValueError\n"


def test_command_help_requires_no_binding_or_private_inputs():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_secondary_development.py"), "--help"],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert b"--expected-profile-sha256" in result.stdout
    assert b"--worker" not in result.stdout
    assert result.stderr == b""


def test_real_fresh_worker_produces_independently_verified_invented_evidence(
    runner, inputs, monkeypatch, tmp_path
):
    """Only public binding is substituted; worker, fits, receipts and verifier are real."""
    from automated_phishing_detection import development_execution
    from automated_phishing_detection.development_execution import (
        DevelopmentExecutionBinding,
    )

    original, paths, _, _ = inputs
    binding = DevelopmentExecutionBinding(**vars(original))
    scripts = binding.base.root / "scripts"
    scripts.mkdir()
    fixture_binding = tmp_path / "invented-binding.json"
    fixture_binding.write_text(
        json.dumps(
            {
                "root": str(binding.base.root),
                "revision": binding.base.revision,
                "contract_sha256": binding.base.contract_sha256,
                "runtime_json": binding.base.runtime_json,
                "profile_sha256": binding.profile_sha256,
                "methods_sha256": binding.methods_sha256,
                "pins": asdict(binding.pins),
                "preparation_hex": binding.preparation_bytes.hex(),
            }
        )
    )
    bootstrap = f"""import json
import runpy
from pathlib import Path
from automated_phishing_detection import development_runner
from automated_phishing_detection.development_execution import DevelopmentExecutionBinding
from automated_phishing_detection.execution_preflight import ExecutionBinding
from automated_phishing_detection.secondary_development import DevelopmentPins
data = json.loads(Path({str(fixture_binding)!r}).read_bytes())
base = ExecutionBinding(Path(data['root']), data['revision'], data['contract_sha256'], (), data['runtime_json'])
binding = DevelopmentExecutionBinding(base, data['profile_sha256'], data['methods_sha256'], DevelopmentPins(**data['pins']), bytes.fromhex(data['preparation_hex']))
development_runner.bind_development_execution = lambda *args, **kwargs: binding
development_runner.recheck_development_binding = lambda value: None
command = runpy.run_path({str(ROOT / "scripts/run_secondary_development.py")!r})
raise SystemExit(command['main']())
"""
    (scripts / "run_secondary_development.py").write_text(bootstrap)
    monkeypatch.setattr(runner, "bind_development_execution", lambda *a, **k: binding)
    monkeypatch.setattr(
        development_execution, "recheck_development_binding", lambda _: None
    )
    result = runner.run_development_process(
        binding.base.root,
        expected_revision=binding.base.revision,
        expected_profile_sha256=binding.profile_sha256,
        paths=paths,
    )
    assert result == json.loads(paths.public_summary.read_bytes())
    assert [entry["member"] for entry in result["members"]] == list(STEPS)
    assert result["protected_evaluation_authorized"] is False
