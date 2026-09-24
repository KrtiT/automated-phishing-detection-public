"""Audit retained seeds, then run one fresh zero-fit probe exactly once."""

from __future__ import annotations

import fcntl
import re
import subprocess
import sys
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from . import development_completion as completion
from . import execution_receipt as receipt
from . import (
    secondary_development,
    seed_probe_audit,
    seed_probe_probes,
    seed_probe_runner,
    source_runner,
)
from .development_runner import _ERROR_SYMBOLS
from .seed_probe_correction import (
    SeedProbeCorrectionError,
    bind_seed_probe_correction,
    recheck_seed_probe_correction,
)

STAGES = ("retained_seed_audit", "probes")
_RECEIPTS = {"reservation.json", "finalize.claim", "outcome.json"}
_PROBE_ARTIFACTS = {
    "length-only.json",
    "logistic-l1.json",
    "gmm.json",
    "transformer.json",
    "cascade.json",
    "vocabulary.json",
    "transformer-weights.npz",
}
_INPUT_PATH_NAMES = (
    "validation",
    "suffix_rules",
    "length_only",
    "logistic_l1",
    "transformer_bundle",
    "gmm",
    "drift_reference",
    "drift_audit",
    "original_attempt",
)
_CHECKS = {
    "preflight",
    "reservation",
    "input_validation",
    "retained_seed_audit",
    "probe_replay",
    "checkpoint_write",
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
            "stage_already_attempted",
            "audit_worker_not_successful",
            "input_hash_mismatch",
            "invalid_stage_outputs",
            "retained_output_changed",
            "probe_correction_worker_stopped",
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
            "audit_payload_mismatch",
            "accepted_stage_changed",
            "probe_correction_stopped",
            "invalid_completion_paths",
            "missing_completion_record",
            "root_stage_evidence_mismatch",
            "root_summary_mismatch",
            "completion_record_changed",
        }
    )
    | _CHECKS
)


class ProbeCorrectionRunError(ValueError):
    """A correction run stopped; durable records contain only safe symbols."""

    def __init__(self, reason):
        self.check_id = (
            reason
            if type(reason) is str and reason in _RUN_FAILURES
            else "unclassified_check"
        )
        super().__init__(self.check_id)


@dataclass(frozen=True)
class ProbeCorrectionPaths:
    validation: Path
    suffix_rules: Path
    length_only: Path
    logistic_l1: Path
    transformer_bundle: Path
    gmm: Path
    drift_reference: Path
    drift_audit: Path
    original_attempt: Path
    attempt: Path
    public_summary: Path


def _require(condition, reason):
    if not condition:
        raise ProbeCorrectionRunError(reason)


def _json_bytes(value):
    return secondary_development._json_bytes(value)


@contextmanager
def _mutation_guard(attempt):
    with receipt._attempt_directory(attempt) as directory:
        fcntl.flock(directory.descriptor, fcntl.LOCK_EX)
        try:
            directory.check()
            try:
                yield directory
            finally:
                directory.check()
        finally:
            fcntl.flock(directory.descriptor, fcntl.LOCK_UN)


def _install_unfinalized_record(directory, name, content):
    _require(
        type(name) is str
        and receipt._FILENAME.fullmatch(name) is not None
        and name not in _RECEIPTS | {"evidence"},
        "invalid_record_name",
    )
    _require(type(content) is bytes, "invalid_record_bytes")
    if any(
        receipt._entry(directory, name) is not None
        for name in ("finalize.claim", "outcome.json", "evidence")
    ):
        raise receipt.ExecutionReceiptError("attempt already finalized")
    receipt._install_record(directory, name, content)


def _record(attempt, name, content):
    with _mutation_guard(attempt) as directory:
        _install_unfinalized_record(directory, name, content)


def _identity(binding):
    seed_probe = binding.seed_probe
    return {
        "kind": "secondary_seed_probe_correction",
        "revision": binding.base.revision,
        "profile_sha256": binding.profile_sha256,
        "base_seed_probe_profile_sha256": seed_probe.profile_sha256,
        "methods_sha256": seed_probe.methods_sha256,
        "original_accounting_sha256": sha256(binding.accounting_bytes).hexdigest(),
        "execution_contract_sha256": seed_probe.base.contract_sha256,
        "runtime_sha256": sha256(seed_probe.base.runtime_json.encode()).hexdigest(),
        "pins": asdict(seed_probe.pins),
        "stages": list(STAGES),
    }


def _validated_input_paths(paths):
    try:
        return {
            name: receipt._absolute_path(getattr(paths, name))
            for name in _INPUT_PATH_NAMES
        }
    except receipt.ExecutionReceiptError:
        raise ProbeCorrectionRunError("invalid_paths") from None


def _preserved_paths(binding, paths):
    inputs = _validated_input_paths(paths)
    try:
        root = receipt._absolute_path(binding.base.root)
    except receipt.ExecutionReceiptError:
        raise ProbeCorrectionRunError("invalid_paths") from None
    return (
        root,
        inputs["original_attempt"],
        inputs["transformer_bundle"],
        inputs["drift_reference"].parent,
        inputs["drift_audit"].parent,
    )


def _outside_outputs(binding, paths):
    _require(type(paths) is ProbeCorrectionPaths, "invalid_paths")
    preserved = _preserved_paths(binding, paths)
    try:
        attempt, public = (
            receipt._absolute_path(path)
            for path in (paths.attempt, paths.public_summary)
        )
    except receipt.ExecutionReceiptError:
        raise ProbeCorrectionRunError("invalid_paths") from None
    for absolute in (attempt, public):
        _require(
            not any(absolute.is_relative_to(parent) for parent in preserved),
            "outputs_inside_preserved_input",
        )
        with receipt._directory(absolute.parent) as directory:
            receipt._require_absent(directory, absolute.name)
    _require(not public.is_relative_to(attempt), "marker_inside_attempt")


def _completion_paths(binding, paths):
    _require(type(paths) is ProbeCorrectionPaths, "invalid_paths")
    preserved = _preserved_paths(binding, paths)
    try:
        attempt = receipt._absolute_path(paths.attempt)
        public = receipt._absolute_path(paths.public_summary)
    except receipt.ExecutionReceiptError:
        raise ProbeCorrectionRunError("invalid_paths") from None
    _require(
        not any(attempt.is_relative_to(parent) for parent in preserved)
        and not any(public.is_relative_to(parent) for parent in preserved)
        and not public.is_relative_to(attempt),
        "invalid_completion_paths",
    )
    return attempt, public


def _require_unfinalized_root(binding, directory):
    _require(
        all(
            receipt._entry(directory, name) is None
            for name in ("finalize.claim", "outcome.json", "evidence")
        ),
        "root_finalized",
    )
    _require(
        completion._same(
            source_runner._json(receipt._read_reservation(directory))["identity"],
            _identity(binding),
        ),
        "root_identity_mismatch",
    )


def _root_attempt(binding, paths, stage):
    _require(stage in STAGES, "invalid_stage")
    content = source_runner._read_file_once(paths.attempt / "reservation.json")
    parent = receipt.Attempt(paths.attempt.absolute(), sha256(content).hexdigest())
    with receipt._attempt_directory(parent) as directory:
        _require_unfinalized_root(binding, directory)
        for later in STAGES[STAGES.index(stage) :]:
            _require(
                all(
                    receipt._entry(directory, name) is None
                    for name in (later, f"{later}.json", f"{later}-process.json")
                ),
                "stage_already_attempted",
            )
    return parent


def _bound_input(path, digest):
    content = source_runner._read_file_once(path)
    _require(sha256(content).hexdigest() == digest, "input_hash_mismatch")
    return content


def _read_probe_inputs(binding, paths):
    seed_probe = binding.seed_probe
    hashes = dict(seed_probe.primary_artifact_hashes)
    _require(set(hashes) == _PROBE_ARTIFACTS, "input_hash_mismatch")
    artifact_paths = {
        "length-only.json": paths.length_only,
        "logistic-l1.json": paths.logistic_l1,
        "gmm.json": paths.gmm,
    }
    artifacts = {
        name: _bound_input(
            artifact_paths.get(name, paths.transformer_bundle / name), hashes[name]
        )
        for name in sorted(_PROBE_ARTIFACTS)
    }
    return {
        "validation_bytes": _bound_input(
            paths.validation, seed_probe.pins.validation_sha256
        ),
        "suffix_rules_bytes": _bound_input(
            paths.suffix_rules, seed_probe.pins.suffix_rules_sha256
        ),
        "artifacts": artifacts,
        "drift_reference_bytes": _bound_input(
            paths.drift_reference, seed_probe.training_reference_sha256
        ),
        "drift_audit_bytes": _bound_input(
            paths.drift_audit, seed_probe.validation_audit_sha256
        ),
    }


def _failure_symbol(error):
    classes = {
        **_ERROR_SYMBOLS,
        ProbeCorrectionRunError: "ProbeCorrectionRunError",
        SeedProbeCorrectionError: "SeedProbeCorrectionError",
        seed_probe_audit.SeedProbeAuditError: "SeedProbeAuditError",
        seed_probe_probes.ProbeStageError: "ProbeStageError",
        completion.DevelopmentCompletionError: "DevelopmentCompletionError",
    }
    symbol, seen = "Exception", set()
    while isinstance(error, BaseException) and id(error) not in seen:
        seen.add(id(error))
        symbol = classes.get(type(error), symbol)
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return symbol


def _failure_check(error, fallback):
    known = (
        (ProbeCorrectionRunError, _RUN_FAILURES),
        (seed_probe_audit.SeedProbeAuditError, seed_probe_audit.CHECK_IDS),
        (seed_probe_probes.ProbeStageError, seed_probe_probes.SAFE_CHECKS),
    )
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
        if type(error) in (
            SeedProbeCorrectionError,
            completion.DevelopmentCompletionError,
        ):
            candidate = error.args[0] if len(error.args) == 1 else None
            if type(candidate) is str and candidate in _RUN_FAILURES:
                return candidate
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return fallback if fallback in _CHECKS else "unclassified_check"


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
    failure = receipt._json_bytes(
        {
            "schema_version": 1,
            "status": "failed",
            "reservation_sha256": attempt.reservation_sha256,
            "stage": safe_stage,
            "error_type": details["error_type"],
        },
        "failure",
    )
    with _mutation_guard(attempt) as directory:
        _install_unfinalized_record(
            directory, "failure-details.json", _json_bytes(details)
        )
        receipt._claim(attempt, directory, "failure")
        receipt._install_record(directory, "outcome.json", failure)


def _verify_process(value, parent, stage, observed_exit):
    fixed = {
        "schema_version": 1,
        "stage": stage,
        "root_reservation_sha256": parent.reservation_sha256,
        "status": "worker_exited",
        "exit_code": observed_exit,
    }
    _require(
        type(value) is dict
        and set(value) == set(fixed) | {"stdout_sha256", "stderr_sha256"},
        "invalid_process_record",
    )
    _require(
        completion._same({key: value[key] for key in fixed}, fixed),
        "process_observation_mismatch",
    )
    _require(
        all(
            type(value[name]) is str
            and receipt._SHA256.fullmatch(value[name]) is not None
            for name in ("stdout_sha256", "stderr_sha256")
        ),
        "invalid_process_digest",
    )


def _audit_gate(binding, paths, parent):
    try:
        process = source_runner._json(
            source_runner._read_file_once(
                parent.directory / "retained_seed_audit-process.json"
            )
        )
    except Exception:
        raise ProbeCorrectionRunError("audit_worker_not_successful") from None
    _require(
        type(process) is dict
        and type(process.get("exit_code")) is int
        and process["exit_code"] == 0,
        "audit_worker_not_successful",
    )
    _verify_process(process, parent, "retained_seed_audit", 0)
    _verify_stage(binding, paths, parent, "retained_seed_audit", observed_exit=0)


def _probe_progress(positions, streams):
    prefixes = {}
    for index, available in positions.items():
        count = 0
        while count + 1 in available:
            count += 1
        prefixes[str(index)] = count
    return {
        "completed_row_prefix": prefixes,
        "completed_streams": sorted(streams),
    }


def _run_worker(binding, paths, stage):
    _require(stage in STAGES, "invalid_stage")
    recheck_seed_probe_correction(binding)
    _completion_paths(binding, paths)
    parent = _root_attempt(binding, paths, stage)
    if stage == "probes":
        _audit_gate(binding, paths, parent)
    with _mutation_guard(parent):
        parent = _root_attempt(binding, paths, stage)
        child = receipt.reserve_attempt(
            parent.directory / stage,
            identity={
                "root_reservation_sha256": parent.reservation_sha256,
                "stage": stage,
            },
        )
    check, publishing = (
        (
            "retained_seed_audit"
            if stage == "retained_seed_audit"
            else "input_validation"
        ),
        False,
    )
    hashes = {}
    positions = {index: set() for index in range(4)}
    streams = set()

    def retain(name, content):
        nonlocal check
        previous = check
        check = "checkpoint_write"
        _record(child, name, content)
        hashes[name] = sha256(content).hexdigest()
        row = re.fullmatch(r"score-row-(0[0-3])-(\d{6})\.json", name)
        stream = re.fullmatch(r"stream-(0[0-3])\.json", name)
        if row is not None:
            positions[int(row[1])].add(int(row[2]))
        if stream is not None:
            streams.add(int(stream[1]))
        check = previous

    try:
        if stage == "retained_seed_audit":
            result = seed_probe_audit.audit_retained_seed_stages(
                binding, original_attempt=paths.original_attempt
            )
            outputs = {"audit.json": _json_bytes(result)}
        else:
            inputs = _read_probe_inputs(binding, paths)
            check = "probe_replay"
            outputs, result = seed_probe_probes.run_probe_stage(
                binding.seed_probe, **inputs, retain=retain
            )
            _require(
                type(outputs) is dict and "artifact-manifest.json" not in outputs,
                "invalid_stage_outputs",
            )
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
        recheck_seed_probe_correction(binding)
        summary = {
            "schema_version": 1,
            "status": "completed_secondary_seed_probe_correction_stage",
            "stage": stage,
            "root_reservation_sha256": parent.reservation_sha256,
            "reservation_sha256": child.reservation_sha256,
            "result": result,
            "private_sha256": {
                name: sha256(content).hexdigest() for name, content in outputs.items()
            },
        }
        check = "publication"
        with _mutation_guard(parent) as root_directory:
            _require_unfinalized_root(binding, root_directory)
            with _mutation_guard(child):
                publishing = True
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
                progress=(
                    _probe_progress(positions, streams) if stage == "probes" else None
                ),
            )
        raise ProbeCorrectionRunError("probe_correction_worker_stopped") from None


def run_probe_correction_worker(
    root, *, expected_revision, expected_profile_sha256, paths, stage
):
    binding = bind_seed_probe_correction(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _run_worker(binding, paths, stage)


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
        raise ProbeCorrectionRunError("worker_launch_failed") from None
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


@contextmanager
def _stage_snapshot(parent, stage):
    """Pin an exact stage inventory and every file state through verification."""
    _require(stage in STAGES, "invalid_stage")
    path = parent.directory / stage
    with ExitStack() as stack:
        child = stack.enter_context(receipt._directory(path))
        evidence = stack.enter_context(receipt._directory(path / "evidence"))
        root = stack.enter_context(receipt._directory(parent.directory))
        contents, states = {}, []

        def read(directory, name):
            _require(
                type(name) is str and receipt._FILENAME.fullmatch(name) is not None,
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
        private = outcome.get("private_sha256") if type(outcome) is dict else None
        _require(
            type(private) is dict
            and all(
                type(name) is str
                and receipt._FILENAME.fullmatch(name) is not None
                and type(digest) is str
                and receipt._SHA256.fullmatch(digest) is not None
                for name, digest in private.items()
            ),
            "private_hash_mismatch",
        )
        names = set(private)
        if stage == "retained_seed_audit":
            _require(names == {"audit.json"}, "private_hash_mismatch")
            auxiliary = {}
        else:
            _require("artifact-manifest.json" in names, "missing_artifact_manifest")
            manifest_bytes = read(evidence, "artifact-manifest.json")
            manifest = source_runner._json(manifest_bytes)
            _require(
                type(manifest) is dict
                and set(manifest) == {"schema_version", "auxiliary_sha256"}
                and type(manifest["schema_version"]) is int
                and manifest["schema_version"] == 1
                and type(manifest["auxiliary_sha256"]) is dict
                and _json_bytes(manifest) == manifest_bytes,
                "invalid_artifact_manifest",
            )
            auxiliary = manifest["auxiliary_sha256"]
            _require(
                all(
                    type(name) is str
                    and receipt._FILENAME.fullmatch(name) is not None
                    and name not in _RECEIPTS | {"evidence"}
                    and type(digest) is str
                    and receipt._SHA256.fullmatch(digest) is not None
                    for name, digest in auxiliary.items()
                ),
                "invalid_auxiliary_names",
            )
        completion._directory_contents(evidence, names)
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
                sha256(read(evidence, name)).hexdigest() == private[name],
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

        def validate():
            completion._directory_contents(evidence, names)
            completion._directory_contents(
                child, _RECEIPTS | {"evidence"} | set(auxiliary)
            )
            root.check()
            for directory, name, state in states:
                metadata = receipt._entry(directory, name)
                _require(
                    metadata is not None
                    and source_runner._file_state(metadata) == state,
                    "stage_output_changed",
                )

        yield contents, names, set(auxiliary), validate
        validate()


def _verify_stage(binding, paths, parent, stage, *, observed_exit, _opened=None):
    _require(
        type(observed_exit) is int and observed_exit == 0,
        "worker_exit_not_successful",
    )
    path = parent.directory / stage
    snapshot = (
        _stage_snapshot(parent, stage) if _opened is None else nullcontext(_opened)
    )
    with snapshot as (contents, names, auxiliary_names, _):
        process = source_runner._json(
            contents[parent.directory / f"{stage}-process.json"]
        )
        _verify_process(process, parent, stage, observed_exit)
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
        if stage == "retained_seed_audit":
            audit_bytes = outputs["audit.json"]
            result = source_runner._json(audit_bytes)
            _require(_json_bytes(result) == audit_bytes, "audit_payload_mismatch")
            seed_probe_audit.validate_audit_summary(
                result, binding=binding, original_attempt=paths.original_attempt
            )
        else:
            result = seed_probe_probes.verify_probe_stage(
                binding.seed_probe, outputs=outputs, auxiliary=auxiliary
            )
        expected = {
            "schema_version": 1,
            "status": "completed_secondary_seed_probe_correction_stage",
            "stage": stage,
            "root_reservation_sha256": parent.reservation_sha256,
            "reservation_sha256": reservation_hash,
            "result": result,
            "private_sha256": hashes,
        }
        _require(completion._same(summary, expected), "stage_summary_mismatch")
        recheck_seed_probe_correction(binding)
        return summary


def _command(binding, paths, stage):
    command = [
        sys.executable,
        str(binding.base.root / "scripts/run_secondary_probe_correction.py"),
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


def _root_summary(binding, parent, accepted):
    return {
        "schema_version": 1,
        "status": "completed_secondary_seed_probe_correction",
        "analysis_stage": "development_validation_only",
        "protected_evaluation_authorized": False,
        "original_v2_aggregate_accepted": False,
        "original_v2_profile_status": "exhausted",
        "new_fits": 0,
        "seed_stage_executions": 0,
        "probe_executions": 1,
        "retries": 0,
        "execution": {
            **_identity(binding),
            "reservation_sha256": parent.reservation_sha256,
        },
        "worker_exit_codes": dict.fromkeys(STAGES, 0),
        "retained_seed_audit": accepted[0],
        "probes": accepted[1],
        "verification_scope": (
            "saved_audit_and_probe_evidence_actual_worker_exits_no_refit_"
            "no_seed_execution_no_source_or_primary_rescoring"
        ),
    }


def _prepublication_root_inventory(directory):
    completion._directory_contents(
        directory,
        {"reservation.json"}
        | set(STAGES)
        | {
            f"{stage}{suffix}"
            for stage in STAGES
            for suffix in (".json", "-process.json")
        },
    )


def _supervise(binding, paths):
    recheck_seed_probe_correction(binding)
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
        recheck_seed_probe_correction(binding)
        with ExitStack() as snapshots:
            opened = {
                member: snapshots.enter_context(_stage_snapshot(parent, member))
                for member in STAGES
            }
            root_directory = snapshots.enter_context(_mutation_guard(parent))
            for member in STAGES:
                child_path = parent.directory / member
                snapshots.enter_context(
                    _mutation_guard(
                        receipt.Attempt(
                            child_path,
                            sha256(
                                opened[member][0][child_path / "reservation.json"]
                            ).hexdigest(),
                        )
                    )
                )
            for member, previous in zip(STAGES, accepted, strict=True):
                current = _verify_stage(
                    binding,
                    paths,
                    parent,
                    member,
                    observed_exit=0,
                    _opened=opened[member],
                )
                _require(completion._same(current, previous), "accepted_stage_changed")
            recheck_seed_probe_correction(binding)
            _prepublication_root_inventory(root_directory)
            summary = _root_summary(binding, parent, accepted)
            _prepublication_root_inventory(root_directory)
            for member in STAGES:
                opened[member][3]()
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
                progress=(
                    seed_probe_runner._observed_probe_progress(parent)
                    if stage == "probes"
                    else None
                ),
            )
        raise ProbeCorrectionRunError("probe_correction_stopped") from None


def run_probe_correction(root, *, expected_revision, expected_profile_sha256, paths):
    binding = bind_seed_probe_correction(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _supervise(binding, paths)


def _verify_completion(binding, paths, *, producer_exit_code):
    """Reverify both saved stages without source reads, fitting, or execution."""
    _require(
        type(producer_exit_code) is int and producer_exit_code == 0,
        "producer_exit_not_successful",
    )
    recheck_seed_probe_correction(binding)
    path, public = _completion_paths(binding, paths)
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
        opened = {
            stage: stack.enter_context(_stage_snapshot(parent, stage))
            for stage in STAGES
        }
        accepted = [
            _verify_stage(
                binding,
                paths,
                parent,
                stage,
                observed_exit=0,
                _opened=opened[stage],
            )
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
        recheck_seed_probe_correction(binding)
        completion._directory_contents(directory, expected)
        completion._directory_contents(evidence, {"stage-summaries.json"})
        public_parent.check()
        for owner, name, state in states:
            metadata = receipt._entry(owner, name)
            _require(
                metadata is not None and source_runner._file_state(metadata) == state,
                "completion_record_changed",
            )
        return summary


def verify_probe_correction(
    root,
    *,
    expected_revision,
    expected_profile_sha256,
    paths,
    producer_exit_code,
):
    _require(
        type(producer_exit_code) is int and producer_exit_code == 0,
        "producer_exit_not_successful",
    )
    binding = bind_seed_probe_correction(
        root,
        expected_revision=expected_revision,
        expected_profile_sha256=expected_profile_sha256,
    )
    return _verify_completion(binding, paths, producer_exit_code=producer_exit_code)
