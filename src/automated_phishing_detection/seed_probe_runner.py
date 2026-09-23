"""Run the fixed seed/probe sequence once, retaining evidence before later work."""

from __future__ import annotations

import os
import re
import stat
import subprocess
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from . import development_completion as completion
from . import execution_receipt as receipt
from . import secondary_development, source_runner
from .development_runner import _failure_symbol

STAGES = ("seed_42_calibration", "seed_43", "seed_44", "seed_45", "seed_46", "probes")
_RECEIPTS = {"reservation.json", "finalize.claim", "outcome.json"}
_CHECKS = {
    "preflight",
    "reservation",
    "input_validation",
    "fit",
    "checkpoint_write",
    "seed_calibration",
    "probe_replay",
    "publication",
    "verification",
    "final_binding",
    "worker_exit_not_successful",
    "worker_launch_failed",
    "unclassified_check",
}
_RUN_FAILURES = (
    frozenset(
        {
            "invalid_record_name",
            "invalid_record_bytes",
            "invalid_paths",
            "outputs_inside_preserved_input",
            "marker_inside_attempt",
            "invalid_stage",
            "root_finalized",
            "root_identity_mismatch",
            "prior_stage_not_observed",
            "stage_already_attempted",
            "input_hash_mismatch",
            "invalid_stage_outputs",
            "retained_output_changed",
            "seed_probe_worker_stopped",
            "producer_exit_not_successful",
            "invalid_evidence_name",
            "missing_stage_output",
            "missing_artifact_manifest",
            "invalid_artifact_manifest",
            "invalid_auxiliary_names",
            "auxiliary_hash_mismatch",
            "private_hash_mismatch",
            "stage_output_changed",
            "invalid_process_record",
            "process_observation_mismatch",
            "invalid_process_digest",
            "stage_summary_mismatch",
            "accepted_stage_changed",
            "seed_probe_execution_stopped",
            "invalid_completion_paths",
            "missing_completion_record",
            "root_stage_evidence_mismatch",
            "root_summary_mismatch",
            "completion_record_changed",
        }
    )
    | _CHECKS
)


class SeedProbeRunError(ValueError):
    """A run stopped; the durable record carries its safe diagnostic fields."""

    def __init__(self, reason):
        self.check_id = (
            reason
            if type(reason) is str and reason in _RUN_FAILURES
            else "unclassified_check"
        )
        super().__init__(self.check_id)


@dataclass(frozen=True)
class SeedProbePaths:
    train: Path
    validation: Path
    suffix_rules: Path
    length_only: Path
    logistic_l1: Path
    transformer_bundle: Path
    gmm: Path
    drift_reference: Path
    drift_audit: Path
    attempt: Path
    public_summary: Path


def _require(condition, reason):
    if not condition:
        raise SeedProbeRunError(reason)


def _json_bytes(value):
    return secondary_development._json_bytes(value)


def recheck_seed_probe_binding(binding):
    from .seed_probe_execution import recheck_seed_probe_binding as recheck

    recheck(binding)


def _record(attempt, name, content):
    _require(
        type(name) is str
        and receipt._FILENAME.fullmatch(name) is not None
        and name not in _RECEIPTS | {"evidence"},
        "invalid_record_name",
    )
    _require(type(content) is bytes, "invalid_record_bytes")
    with receipt._attempt_directory(attempt) as directory:
        if any(
            receipt._entry(directory, name) is not None
            for name in ("finalize.claim", "outcome.json", "evidence")
        ):
            raise receipt.ExecutionReceiptError("attempt already finalized")
        receipt._install_record(directory, name, content)


def _identity(binding):
    return {
        "kind": "secondary_seed_probes",
        "revision": binding.base.revision,
        "profile_sha256": binding.profile_sha256,
        "methods_sha256": binding.methods_sha256,
        "execution_contract_sha256": binding.base.contract_sha256,
        "runtime_sha256": sha256(binding.base.runtime_json.encode()).hexdigest(),
        "pins": asdict(binding.pins),
        "stages": list(STAGES),
    }


def _outside_outputs(binding, paths):
    _require(type(paths) is SeedProbePaths, "invalid_paths")
    preserved = (
        binding.base.root,
        paths.transformer_bundle,
        paths.drift_reference.parent,
        paths.drift_audit.parent,
    )
    for path in (paths.attempt, paths.public_summary):
        absolute = receipt._absolute_path(path)
        _require(
            not any(absolute.is_relative_to(parent.absolute()) for parent in preserved),
            "outputs_inside_preserved_input",
        )
        with receipt._directory(absolute.parent) as directory:
            receipt._require_absent(directory, absolute.name)
    _require(
        not paths.public_summary.absolute().is_relative_to(paths.attempt.absolute()),
        "marker_inside_attempt",
    )


def _failure(
    attempt, stage, error, *, accepted=(), check="unclassified_check", progress=None
):
    safe_stage = stage if stage in STAGES else "preflight"
    failed = stage if stage in STAGES and stage not in accepted else None
    attempted_count = len(accepted) + int(failed is not None)
    details = {
        "schema_version": 1,
        "status": "stopped",
        "check": _failure_check(error, check),
        "error_type": _failure_symbol(error),
        "completed_stages": list(accepted),
        "failed_stage": failed,
        "unattempted_stages": list(STAGES[attempted_count:]),
    }
    if progress is not None:
        details["probe_progress"] = progress
    _record(attempt, "failure-details.json", _json_bytes(details))
    receipt.record_failure(attempt, stage=safe_stage, error_type=details["error_type"])


def _failure_check(error, fallback):
    known = [(SeedProbeRunError, _RUN_FAILURES)]
    # Only modules already used by this worker can contribute typed diagnostics.
    for module_name, error_name in (
        ("seed_probe_seeds", "SeedStageError"),
        ("seed_probe_probes", "ProbeStageError"),
    ):
        module = sys.modules.get(f"{__package__}.{module_name}")
        if module is not None:
            known.append((getattr(module, error_name, None), module.SAFE_CHECKS))
    seen = set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        for error_class, permitted in known:
            if (
                type(error) is error_class
                and error.check_id in permitted
                and error.check_id != "unclassified_check"
            ):
                return error.check_id
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return fallback if fallback in _CHECKS else "unclassified_check"


def _launch(parent, stage, command):
    _require(stage in STAGES, "invalid_stage")
    observation = {
        "schema_version": 1,
        "stage": stage,
        "root_reservation_sha256": parent.reservation_sha256,
    }
    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError:
        _record(
            parent,
            f"{stage}-process.json",
            _json_bytes(
                {**observation, "status": "worker_launch_failed", "exit_code": None}
            ),
        )
        raise SeedProbeRunError("worker_launch_failed") from None
    _record(
        parent,
        f"{stage}-process.json",
        _json_bytes(
            {
                **observation,
                "status": "worker_exited",
                "exit_code": result.returncode,
                "stdout_sha256": sha256(result.stdout).hexdigest(),
                "stderr_sha256": sha256(result.stderr).hexdigest(),
            }
        ),
    )
    return result.returncode


def _command(binding, paths, stage):
    command = [
        sys.executable,
        str(binding.base.root / "scripts/run_secondary_seed_probes.py"),
        "--repo-root",
        str(binding.base.root),
        "--expected-revision",
        binding.base.revision,
        "--expected-profile-sha256",
        binding.profile_sha256,
        "--worker",
        stage,
    ]
    for name, path in vars(paths).items():
        command.extend(("--" + name.replace("_", "-"), str(path)))
    return command


def _root_attempt(binding, paths, stage):
    _require(stage in STAGES, "invalid_stage")
    content = source_runner._read_file_once(paths.attempt / "reservation.json")
    parent = receipt.Attempt(paths.attempt.absolute(), sha256(content).hexdigest())
    with receipt._attempt_directory(parent) as directory:
        _require(
            all(
                receipt._entry(directory, name) is None
                for name in ("finalize.claim", "outcome.json", "evidence")
            ),
            "root_finalized",
        )
        _require(
            completion._same(
                source_runner._json(content)["identity"], _identity(binding)
            ),
            "root_identity_mismatch",
        )
        index = STAGES.index(stage)
        for previous in STAGES[:index]:
            _require(
                all(
                    receipt._entry(directory, name) is not None
                    for name in (
                        previous,
                        f"{previous}.json",
                        f"{previous}-process.json",
                    )
                ),
                "prior_stage_not_observed",
            )
        for later in STAGES[index:]:
            _require(
                all(
                    receipt._entry(directory, name) is None
                    for name in (later, f"{later}.json", f"{later}-process.json")
                ),
                "stage_already_attempted",
            )
    return parent


def _prior_observations(binding, parent, stage):
    for previous in STAGES[: STAGES.index(stage)]:
        record = source_runner._json(
            source_runner._read_file_once(parent.directory / f"{previous}-process.json")
        )
        _require(
            type(record) is dict
            and completion._same(
                {
                    key: record.get(key)
                    for key in (
                        "schema_version",
                        "stage",
                        "root_reservation_sha256",
                        "status",
                        "exit_code",
                    )
                },
                {
                    "schema_version": 1,
                    "stage": previous,
                    "root_reservation_sha256": parent.reservation_sha256,
                    "status": "worker_exited",
                    "exit_code": 0,
                },
            ),
            "process_observation_mismatch",
        )


def _common_stage1(binding, paths, parent):
    with _stage_snapshot(parent, STAGES[0]) as opened:
        _verify_stage(
            binding, paths, parent, STAGES[0], observed_exit=0, _opened=opened
        )
        return opened[0][parent.directory / STAGES[0] / "common-stage1.jsonl"]


def _bound_input(path, digest):
    content = source_runner._read_file_once(path)
    _require(sha256(content).hexdigest() == digest, "input_hash_mismatch")
    return content


def _read_inputs(binding, paths, parent, stage):
    arguments = {
        "validation_bytes": _bound_input(
            paths.validation, binding.pins.validation_sha256
        ),
        "suffix_rules_bytes": _bound_input(
            paths.suffix_rules, binding.pins.suffix_rules_sha256
        ),
    }
    names = {"logistic-l1.json", "vocabulary.json"}
    if stage == STAGES[0]:
        names.add("transformer-weights.npz")
    elif stage == "probes":
        names = set(dict(binding.primary_artifact_hashes))
    artifact_paths = {
        "length-only.json": paths.length_only,
        "logistic-l1.json": paths.logistic_l1,
        "gmm.json": paths.gmm,
    }
    digests = dict(binding.primary_artifact_hashes)
    arguments["artifacts"] = {
        name: _bound_input(
            artifact_paths.get(name, paths.transformer_bundle / name), digests[name]
        )
        for name in sorted(names)
    }
    if stage == "probes":
        arguments["drift_reference_bytes"] = _bound_input(
            paths.drift_reference, binding.training_reference_sha256
        )
        arguments["drift_audit_bytes"] = _bound_input(
            paths.drift_audit, binding.validation_audit_sha256
        )
    else:
        arguments["train_bytes"] = None
        arguments["common_stage1"] = None
        if stage != STAGES[0]:
            arguments["train_bytes"] = _bound_input(
                paths.train, binding.pins.train_sha256
            )
            arguments["common_stage1"] = _common_stage1(binding, paths, parent)
    return arguments


def _compute_stage(binding, stage, inputs, retain):
    if stage == "probes":
        from .seed_probe_probes import run_probe_stage

        return run_probe_stage(binding, **inputs, retain=retain)
    from .seed_probe_seeds import run_seed_stage

    return run_seed_stage(binding, stage, **inputs, retain=retain)


def _run_worker(binding, paths, stage):
    recheck_seed_probe_binding(binding)
    parent = _root_attempt(binding, paths, stage)
    child = receipt.reserve_attempt(
        parent.directory / stage,
        identity={"root_reservation_sha256": parent.reservation_sha256, "stage": stage},
    )
    check, publishing = "input_validation", False
    hashes = {}
    progress = {
        "completed_row_prefix": {str(index): 0 for index in range(4)},
        "completed_streams": [],
    }

    def retain(name, content):
        nonlocal check
        previous = check
        check = "checkpoint_write"
        _record(child, name, content)
        hashes[name] = sha256(content).hexdigest()
        row = re.fullmatch(r"score-row-(0[0-3])-(\d{6})\.json", name)
        stream = re.fullmatch(r"stream-(0[0-3])\.json", name)
        if row is not None:
            progress["completed_row_prefix"][str(int(row[1]))] = int(row[2])
        if stream is not None:
            progress["completed_streams"].append(int(stream[1]))
        check = previous

    try:
        _prior_observations(binding, parent, stage)
        inputs = _read_inputs(binding, paths, parent, stage)
        check = (
            "probe_replay"
            if stage == "probes"
            else "seed_calibration"
            if stage == STAGES[0]
            else "fit"
        )
        outputs, result = _compute_stage(binding, stage, inputs, retain)
        _require(
            type(outputs) is dict and "artifact-manifest.json" not in outputs,
            "invalid_stage_outputs",
        )
        # A retained final record must agree with the bytes being published.
        for name in set(outputs) & set(hashes):
            _require(
                sha256(outputs[name]).hexdigest() == hashes[name],
                "retained_output_changed",
            )
        outputs = {
            **outputs,
            "artifact-manifest.json": _json_bytes(
                {"schema_version": 1, "auxiliary_sha256": hashes}
            ),
        }
        check = "final_binding"
        recheck_seed_probe_binding(binding)
        summary = {
            "schema_version": 1,
            "status": "completed_secondary_seed_probe_stage",
            "stage": stage,
            "root_reservation_sha256": parent.reservation_sha256,
            "reservation_sha256": child.reservation_sha256,
            "result": result,
            "private_sha256": {
                name: sha256(content).hexdigest() for name, content in outputs.items()
            },
        }
        check, publishing = "publication", True
        receipt.publish_completion(
            child,
            private_outputs=outputs,
            public_summary=summary,
            public_path=parent.directory / f"{stage}.json",
        )
        return summary
    except Exception as error:
        if not publishing:
            _failure(
                child,
                stage,
                error,
                accepted=STAGES[: STAGES.index(stage)],
                check=check,
                progress=progress if stage == "probes" else None,
            )
        raise SeedProbeRunError("seed_probe_worker_stopped") from None


def run_seed_probes(root, *, expected_revision, expected_profile_sha256, paths):
    from .seed_probe_execution import bind_seed_probe_execution

    binding = bind_seed_probe_execution(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _supervise(binding, paths)


def run_seed_probe_worker(
    root, *, expected_revision, expected_profile_sha256, paths, stage
):
    from .seed_probe_execution import bind_seed_probe_execution

    binding = bind_seed_probe_execution(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _run_worker(binding, paths, stage)


def verify_seed_probe_run(
    root, *, expected_revision, expected_profile_sha256, paths, producer_exit_code
):
    from .seed_probe_execution import bind_seed_probe_execution

    _require(
        type(producer_exit_code) is int and producer_exit_code == 0,
        "producer_exit_not_successful",
    )
    binding = bind_seed_probe_execution(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _verify_completion(binding, paths, producer_exit_code=producer_exit_code)


@contextmanager
def _stage_snapshot(parent, stage):
    """Pin exact inventories and file states through arithmetic verification."""
    path = parent.directory / stage
    with ExitStack() as stack:
        child = stack.enter_context(receipt._directory(path))
        evidence = stack.enter_context(receipt._directory(path / "evidence"))
        root = stack.enter_context(receipt._directory(parent.directory))
        contents, states = {}, []

        def read(directory, name):
            _require(
                type(name) is str and receipt._FILENAME.fullmatch(name),
                "invalid_evidence_name",
            )
            key = directory.path / name
            if key not in contents:
                metadata = receipt._entry(directory, name)
                _require(metadata is not None, "missing_stage_output")
                state = source_runner._file_state(metadata)
                contents[key] = source_runner._read_file_once(key, expected_state=state)
                states.append((directory, name, state))
            return contents[key]

        outcome = source_runner._json(read(child, "outcome.json"))
        names = set(outcome["private_sha256"])
        _require("artifact-manifest.json" in names, "missing_artifact_manifest")
        completion._directory_contents(evidence, names)
        manifest = source_runner._json(read(evidence, "artifact-manifest.json"))
        _require(
            type(manifest) is dict
            and set(manifest) == {"schema_version", "auxiliary_sha256"}
            and type(manifest["schema_version"]) is int
            and manifest["schema_version"] == 1
            and type(manifest["auxiliary_sha256"]) is dict,
            "invalid_artifact_manifest",
        )
        auxiliary = manifest["auxiliary_sha256"]
        _require(
            not set(auxiliary) & (_RECEIPTS | {"evidence"}), "invalid_auxiliary_names"
        )
        completion._directory_contents(child, _RECEIPTS | {"evidence"} | set(auxiliary))
        for name in sorted(_RECEIPTS | set(auxiliary)):
            content = read(child, name)
            if name in auxiliary:
                _require(
                    sha256(content).hexdigest() == auxiliary[name],
                    "auxiliary_hash_mismatch",
                )
        for name in sorted(names):
            _require(
                sha256(read(evidence, name)).hexdigest()
                == outcome["private_sha256"][name],
                "private_hash_mismatch",
            )
        read(root, f"{stage}.json")
        read(root, f"{stage}-process.json")
        _, summary, hashes = completion._receipt(
            contents,
            path,
            parent.directory / f"{stage}.json",
            {"root_reservation_sha256": parent.reservation_sha256, "stage": stage},
            names,
        )
        _require(
            completion._same(summary.get("private_sha256"), hashes),
            "private_hash_mismatch",
        )
        yield contents, names, set(auxiliary)
        completion._directory_contents(evidence, names)
        completion._directory_contents(child, _RECEIPTS | {"evidence"} | set(auxiliary))
        root.check()
        for directory, name, state in states:
            metadata = receipt._entry(directory, name)
            _require(
                metadata is not None and source_runner._file_state(metadata) == state,
                "stage_output_changed",
            )


def _verify_stage(binding, paths, parent, stage, *, observed_exit, _opened=None):
    _require(
        type(observed_exit) is int and observed_exit == 0, "worker_exit_not_successful"
    )
    path = parent.directory / stage
    snapshot = (
        _stage_snapshot(parent, stage) if _opened is None else nullcontext(_opened)
    )
    with snapshot as (contents, names, auxiliary_names):
        process = source_runner._json(
            contents[parent.directory / f"{stage}-process.json"]
        )
        _require(
            type(process) is dict
            and set(process)
            == {
                "schema_version",
                "stage",
                "root_reservation_sha256",
                "status",
                "exit_code",
                "stdout_sha256",
                "stderr_sha256",
            },
            "invalid_process_record",
        )
        _require(
            completion._same(
                {
                    key: process[key]
                    for key in (
                        "schema_version",
                        "stage",
                        "root_reservation_sha256",
                        "status",
                        "exit_code",
                    )
                },
                {
                    "schema_version": 1,
                    "stage": stage,
                    "root_reservation_sha256": parent.reservation_sha256,
                    "status": "worker_exited",
                    "exit_code": observed_exit,
                },
            ),
            "process_observation_mismatch",
        )
        for key in ("stdout_sha256", "stderr_sha256"):
            _require(
                type(process[key]) is str and receipt._SHA256.fullmatch(process[key]),
                "invalid_process_digest",
            )
        reservation_hash, summary, hashes = completion._receipt(
            contents,
            path,
            parent.directory / f"{stage}.json",
            {"root_reservation_sha256": parent.reservation_sha256, "stage": stage},
            names,
        )
        outputs = {
            name: contents[path / "evidence" / name]
            for name in names - {"artifact-manifest.json"}
        }
        auxiliary = {name: contents[path / name] for name in auxiliary_names}
        result = _verify_science(binding, paths, parent, stage, outputs, auxiliary)
        expected = {
            "schema_version": 1,
            "status": "completed_secondary_seed_probe_stage",
            "stage": stage,
            "root_reservation_sha256": parent.reservation_sha256,
            "reservation_sha256": reservation_hash,
            "result": result,
            "private_sha256": hashes,
        }
        _require(completion._same(summary, expected), "stage_summary_mismatch")
        recheck_seed_probe_binding(binding)
        return summary


def _verify_science(binding, paths, parent, stage, outputs, auxiliary):
    if stage == "probes":
        from .seed_probe_probes import verify_probe_stage

        return verify_probe_stage(binding, outputs=outputs, auxiliary=auxiliary)
    from .seed_probe_seeds import verify_seed_stage

    common = None
    if stage != STAGES[0]:
        common = _common_stage1(binding, paths, parent)
    return verify_seed_stage(
        binding, stage, outputs=outputs, auxiliary=auxiliary, common_stage1=common
    )


def _supervise(binding, paths):
    recheck_seed_probe_binding(binding)
    _outside_outputs(binding, paths)
    parent = receipt.reserve_attempt(paths.attempt, identity=_identity(binding))
    stage, check, publishing = "reservation", "reservation", False
    accepted = []
    try:
        for stage in STAGES:
            check = "worker_launch_failed"
            code = _launch(parent, stage, _command(binding, paths, stage))
            check = "worker_exit_not_successful" if code != 0 else "verification"
            accepted.append(
                _verify_stage(binding, paths, parent, stage, observed_exit=code)
            )
        stage, check = "finalization", "final_binding"
        recheck_seed_probe_binding(binding)
        with ExitStack() as snapshots:
            for member in STAGES:
                snapshots.enter_context(_stage_snapshot(parent, member))
            for member, previous in zip(STAGES, accepted, strict=True):
                current = _verify_stage(binding, paths, parent, member, observed_exit=0)
                _require(completion._same(current, previous), "accepted_stage_changed")
        summary = _root_summary(binding, parent, accepted)
        check, publishing = "publication", True
        receipt.publish_completion(
            parent,
            private_outputs={"stage-summaries.json": _json_bytes(accepted)},
            public_summary=summary,
            public_path=paths.public_summary,
        )
        return summary
    except Exception as error:
        if not publishing:
            _failure(
                parent,
                stage,
                error,
                accepted=[value["stage"] for value in accepted],
                check=check,
                progress=_observed_probe_progress(parent)
                if stage == "probes"
                else None,
            )
        raise SeedProbeRunError("seed_probe_execution_stopped") from None


def _observed_probe_progress(parent):
    """Count installed prefixes after a stop, without accepting their contents."""
    positions = {index: set() for index in range(4)}
    streams, uncertain = [], False
    try:
        with receipt._directory(parent.directory / "probes") as directory:
            for name in os.listdir(directory.descriptor):
                row = re.fullmatch(r"score-row-(0[0-3])-(\d{6})\.json", name)
                stream = re.fullmatch(r"stream-(0[0-3])\.json", name)
                if row is None and stream is None:
                    continue
                metadata = receipt._entry(directory, name)
                if (
                    metadata is None
                    or not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_nlink != 1
                ):
                    uncertain = True
                    continue
                if row is not None:
                    positions[int(row[1])].add(int(row[2]))
                else:
                    streams.append(int(stream[1]))
    except (OSError, receipt.ExecutionReceiptError):
        uncertain = True
    prefixes = {}
    for index, available in positions.items():
        count = 0
        while count + 1 in available:
            count += 1
        prefixes[str(index)] = count
    return {
        "scope": "installed_record_inventory_not_accepted_scientific_evidence",
        "completed_row_prefix": prefixes,
        "observed_completed_stream_records": sorted(streams),
        "unaccepted_row_records_outside_prefix": sum(
            len(positions[index]) - prefixes[str(index)] for index in range(4)
        ),
        "inventory_incomplete": uncertain,
        "failed_row_position": None,
        "durability_confirmed": False,
    }


def _root_summary(binding, parent, accepted):
    return {
        "schema_version": 1,
        "status": "completed_secondary_seed_probes",
        "analysis_stage": "development_validation_only",
        "protected_evaluation_authorized": False,
        "new_fits": 4,
        "seed_order": [42, 43, 44, 45, 46],
        "retries": 0,
        "execution": {
            **_identity(binding),
            "reservation_sha256": parent.reservation_sha256,
        },
        "worker_exit_codes": {name: 0 for name in STAGES},
        "stages": accepted,
        "verification_scope": "saved_evidence_arithmetic_and_actual_worker_exits_no_independent_refit_or_source_rescoring",
    }


def _verify_completion(binding, paths, *, producer_exit_code):
    """Accept only the complete saved family and an externally observed zero exit."""
    _require(
        type(producer_exit_code) is int and producer_exit_code == 0,
        "producer_exit_not_successful",
    )
    recheck_seed_probe_binding(binding)
    path = receipt._absolute_path(paths.attempt)
    public = receipt._absolute_path(paths.public_summary)
    _require(
        not path.is_relative_to(binding.base.root)
        and not public.is_relative_to(binding.base.root)
        and not public.is_relative_to(path),
        "invalid_completion_paths",
    )
    expected = (
        _RECEIPTS
        | {"evidence"}
        | set(STAGES)
        | {
            f"{name}{suffix}"
            for name in STAGES
            for suffix in (".json", "-process.json")
        }
    )
    with ExitStack() as stack:
        directory = stack.enter_context(receipt._directory(path))
        evidence = stack.enter_context(receipt._directory(path / "evidence"))
        public_parent = stack.enter_context(receipt._directory(public.parent))
        completion._directory_contents(directory, expected)
        completion._directory_contents(evidence, {"stage-summaries.json"})
        states, contents = [], {}
        files = [(directory, name) for name in sorted(_RECEIPTS)] + [
            (evidence, "stage-summaries.json"),
            (public_parent, public.name),
        ]
        for owner, name in files:
            metadata = receipt._entry(owner, name)
            _require(metadata is not None, "missing_completion_record")
            state = source_runner._file_state(metadata)
            contents[owner.path / name] = source_runner._read_file_once(
                owner.path / name, expected_state=state
            )
            states.append((owner, name, state))
        reservation_hash, summary, _ = completion._receipt(
            contents, path, public, _identity(binding), {"stage-summaries.json"}
        )
        parent = receipt.Attempt(path, reservation_hash)
        for stage in STAGES:
            stack.enter_context(_stage_snapshot(parent, stage))
        accepted = [
            _verify_stage(binding, paths, parent, stage, observed_exit=0)
            for stage in STAGES
        ]
        _require(
            contents[path / "evidence/stage-summaries.json"] == _json_bytes(accepted),
            "root_stage_evidence_mismatch",
        )
        _require(
            completion._same(summary, _root_summary(binding, parent, accepted)),
            "root_summary_mismatch",
        )
        recheck_seed_probe_binding(binding)
        completion._directory_contents(directory, expected)
        completion._directory_contents(evidence, {"stage-summaries.json"})
        for owner, name, state in states:
            metadata = receipt._entry(owner, name)
            _require(
                metadata is not None and source_runner._file_state(metadata) == state,
                "completion_record_changed",
            )
        return summary
