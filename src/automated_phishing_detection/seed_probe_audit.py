"""Audit the five stopped-v2 seed stages from immutable saved evidence only."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from . import development_completion as completion
from . import execution_receipt, seed_probe_runner, source_runner
from .seed_probe_correction import (
    SeedProbeCorrectionBinding,
    recheck_seed_probe_correction,
)

SEED_STAGES = seed_probe_runner.STAGES[:-1]
ORIGINAL_REVISION = "1135b8e0f0750be7bd83ab0e314c6733ca609eb6"
_RECEIPTS = {"reservation.json", "finalize.claim", "outcome.json"}
_PROBE_INPUT_RECORDS = {
    "input-length-only.json",
    "input-logistic-l1.json",
    "input-gmm.json",
    "input-transformer.json",
    "input-cascade.json",
    "input-vocabulary.json",
    "input-training-reference.json",
    "input-validation-audit.json",
}
_ROOT_RECORDS = {
    *_RECEIPTS,
    "failure-details.json",
    *SEED_STAGES,
    "probes",
    *(f"{stage}.json" for stage in SEED_STAGES),
    *(f"{stage}-process.json" for stage in seed_probe_runner.STAGES),
}
_PROBE_RECEIPTS = _RECEIPTS | {"failure-details.json"}
_PROBE_RECORDS = _PROBE_RECEIPTS | _PROBE_INPUT_RECORDS
_ACCOUNTING_KEYS = {
    "schema_version",
    "record_kind",
    "date",
    "status",
    "analysis_stage",
    "aggregate_accepted",
    "execution",
    "execution_observation",
    "failure",
    "unattempted_stages",
    "completed_seed_stages",
    "process_observation",
    "receipt_sha256",
    "access",
    "authorization",
    "diagnosis",
    "verification",
    "interpretation",
}
CHECK_IDS = frozenset(
    {
        "accounting_requires_bytes",
        "audit_summary_mismatch",
        "invalid_accounting_record",
        "invalid_audit_binding",
        "invalid_audit_summary",
        "invalid_completed_seed_stages",
        "invalid_original_access",
        "invalid_original_authorization",
        "invalid_original_execution",
        "invalid_original_failure",
        "invalid_original_observation",
        "invalid_original_process_observation",
        "invalid_original_receipt_hashes",
        "invalid_original_revision",
        "missing_retained_file",
        "original_attempt_inside_checkout",
        "original_attempt_status_mismatch",
        "original_execution_binding_mismatch",
        "original_failed_reservation_mismatch",
        "original_failure_claim_mismatch",
        "original_failure_details_mismatch",
        "original_failure_outcome_mismatch",
        "original_process_observation_mismatch",
        "original_probe_input_mismatch",
        "original_receipt_hash_mismatch",
        "original_seed_summary_mismatch",
        "retained_seed_audit_failed",
        "retained_seed_file_changed_during_audit",
    }
)


class SeedProbeAuditError(ValueError):
    """A safe stopped-v2 audit check without private values or paths."""

    def __init__(self, check_id: str):
        if type(check_id) is not str or check_id not in CHECK_IDS:
            raise ValueError("unknown seed audit check identifier")
        super().__init__(check_id)
        self.check_id = check_id


def _require(condition, symbol):
    if not condition:
        raise SeedProbeAuditError(symbol)


def _keys(value, expected):
    return type(value) is dict and set(value) == set(expected)


def _hex(value):
    return type(value) is str and execution_receipt._SHA256.fullmatch(value) is not None


def _original_identity(binding, revision):
    seed_probe = binding.seed_probe
    return {
        "kind": "secondary_seed_probes",
        "revision": revision,
        "profile_sha256": seed_probe.profile_sha256,
        "methods_sha256": seed_probe.methods_sha256,
        "execution_contract_sha256": seed_probe.base.contract_sha256,
        "runtime_sha256": sha256(seed_probe.base.runtime_json.encode()).hexdigest(),
        "pins": asdict(seed_probe.pins),
        "stages": list(seed_probe_runner.STAGES),
    }


def _accounting(binding, content):
    _require(type(content) is bytes, "accounting_requires_bytes")
    report = source_runner._json(content)
    _require(_keys(report, _ACCOUNTING_KEYS), "invalid_accounting_record")
    expected_status = {
        "schema_version": 1,
        "record_kind": "secondary_seed_probe_attempt_accounting",
        "status": "failed_partial_evidence_retained",
        "analysis_stage": "development_validation_only",
        "aggregate_accepted": False,
        "unattempted_stages": [],
    }
    _require(
        completion._same(
            {key: report.get(key) for key in expected_status}, expected_status
        ),
        "original_attempt_status_mismatch",
    )
    seed_probe = binding.seed_probe
    execution = report["execution"]
    execution_keys = {
        "kind",
        "revision",
        "profile_sha256",
        "base_profile_sha256",
        "stopped_v1_attempt_sha256",
        "methods_sha256",
        "execution_contract_sha256",
        "runtime_sha256",
        "root_reservation_sha256",
        "stages",
        "seed_order",
    }
    _require(_keys(execution, execution_keys), "invalid_original_execution")
    revision = execution["revision"]
    _require(
        type(revision) is str
        and revision == ORIGINAL_REVISION
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "invalid_original_revision",
    )
    expected_execution = {
        "kind": "secondary_seed_probes",
        "revision": ORIGINAL_REVISION,
        "profile_sha256": seed_probe.profile_sha256,
        "base_profile_sha256": seed_probe.base_profile_sha256,
        "stopped_v1_attempt_sha256": seed_probe.stopped_attempt_sha256,
        "methods_sha256": seed_probe.methods_sha256,
        "execution_contract_sha256": seed_probe.base.contract_sha256,
        "runtime_sha256": sha256(seed_probe.base.runtime_json.encode()).hexdigest(),
        "root_reservation_sha256": execution["root_reservation_sha256"],
        "stages": list(seed_probe_runner.STAGES),
        "seed_order": [42, 43, 44, 45, 46],
    }
    _require(
        _hex(execution["root_reservation_sha256"])
        and completion._same(execution, expected_execution),
        "original_execution_binding_mismatch",
    )
    observation = report["execution_observation"]
    expected_observation = {
        "parent_exit_code": 2,
        "public_summary_present": False,
        "successful_public_marker_present": False,
        "new_fits": 4,
        "retries": 0,
        "resumes": 0,
        "completed_stages": list(SEED_STAGES),
        "worker_exit_codes": {**dict.fromkeys(SEED_STAGES, 0), "probes": 2},
    }
    _require(
        _keys(
            observation,
            set(expected_observation) | {"start_observed_utc", "exit_observed_utc"},
        )
        and all(
            type(observation[name]) is str and bool(observation[name])
            for name in ("start_observed_utc", "exit_observed_utc")
        )
        and completion._same(
            {key: observation[key] for key in expected_observation},
            expected_observation,
        ),
        "invalid_original_observation",
    )
    progress = {
        "completed_row_prefix": dict.fromkeys(("0", "1", "2", "3"), 0),
        "observed_completed_stream_records": [],
        "unaccepted_row_records_outside_prefix": 0,
        "inventory_incomplete": False,
        "failed_row_position": None,
        "durability_confirmed": False,
    }
    failure = report["failure"]
    expected_failure = {
        "stage": "probes",
        "root_safe_check": "worker_exit_not_successful",
        "worker_safe_check": "probe_metadata",
        "root_error_type": "Exception",
        "worker_error_type": "Exception",
        "probe_progress": progress,
    }
    _require(completion._same(failure, expected_failure), "invalid_original_failure")
    completed = report["completed_seed_stages"]
    _require(
        type(completed) is list
        and len(completed) == len(SEED_STAGES)
        and all(
            _keys(member, {"acceptance", "summary_sha256", "summary"})
            for member in completed
        )
        and [member["summary"].get("stage") for member in completed]
        == list(SEED_STAGES)
        and all(
            member["acceptance"]
            == "producer_completed_preliminary_not_independently_accepted"
            and _hex(member["summary_sha256"])
            for member in completed
        )
        and [member["summary"].get("result", {}).get("seed") for member in completed]
        == [42, 43, 44, 45, 46]
        and [member["summary"].get("result", {}).get("new_fit") for member in completed]
        == [False, True, True, True, True]
        and all(
            member["summary"].get("result", {}).get("pure_seed_effect_claim") is False
            and member["summary"].get("result", {}).get("primary_artifacts_changed")
            is False
            for member in completed
        ),
        "invalid_completed_seed_stages",
    )
    records = report["process_observation"]
    _require(
        _keys(records, {"records"})
        and type(records["records"]) is list
        and len(records["records"]) == len(seed_probe_runner.STAGES),
        "invalid_original_process_observation",
    )
    for stage, record in zip(seed_probe_runner.STAGES, records["records"], strict=True):
        _require(
            _keys(
                record,
                {
                    "stage",
                    "record",
                    "exit_code",
                    "stdout_sha256",
                    "stderr_sha256",
                },
            )
            and record["stage"] == stage
            and record["record"] == f"{stage}-process.json"
            and record["exit_code"] == (2 if stage == "probes" else 0)
            and _hex(record["stdout_sha256"])
            and _hex(record["stderr_sha256"]),
            "invalid_original_process_observation",
        )
    receipt_names = {
        "reservation.json",
        "finalize.claim",
        "outcome.json",
        "failure-details.json",
        *(f"{stage}-process.json" for stage in seed_probe_runner.STAGES),
        *(f"{stage}/{name}" for stage in SEED_STAGES for name in _RECEIPTS),
        *(f"probes/{name}" for name in _PROBE_RECEIPTS),
    }
    hashes = report["receipt_sha256"]
    _require(
        _keys(hashes, receipt_names) and all(_hex(value) for value in hashes.values()),
        "invalid_original_receipt_hashes",
    )
    expected_access = {
        "training_partition_accessed": True,
        "development_validation_accessed": True,
        "public_suffix_list_accessed": True,
        "group_test_accessed": False,
        "phishvn_accessed": False,
        "external_source_accessed": False,
        "protected_evaluation_accessed": False,
    }
    _require(
        completion._same(report["access"], expected_access),
        "invalid_original_access",
    )
    expected_authorization = {
        "profile_status": "exhausted",
        "retry_authorized": False,
        "resume_authorized": False,
        "refit_authorized": False,
        "required_before_any_new_probe_execution": [
            "separate_prospective_probe_correction_authority_frozen",
            "reviewed",
            "published",
            "ci_passed",
        ],
    }
    _require(
        completion._same(report["authorization"], expected_authorization),
        "invalid_original_authorization",
    )
    verification = report["verification"]
    _require(
        completion._same(
            verification,
            {
                "completed_seed_stage_verifier_exit_code": 0,
                "completed_seed_stage_scope": (
                    "saved_evidence_arithmetic_and_receipts_no_refit_or_source_rescoring"
                ),
                "complete_family_verification": "not_eligible_producer_exit_2",
                "complete_family_accepted": False,
            },
        ),
        "invalid_accounting_record",
    )
    _require(
        type(report["diagnosis"]) is dict
        and report["diagnosis"].get("failure_position")
        == "before_probe_row_preparation_or_scoring"
        and report["diagnosis"].get("retained_drift_artifacts_corrupt") is False
        and type(report["interpretation"]) is list,
        "invalid_accounting_record",
    )
    return report


def _failed_receipt(contents, path, identity, *, expected_check, progress):
    reservation = completion._canonical(
        {
            "schema_version": 1,
            "status": "reserved",
            "directory": str(path),
            "identity": identity,
        }
    )
    _require(
        contents[path / "reservation.json"] == reservation,
        "original_failed_reservation_mismatch",
    )
    digest = sha256(reservation).hexdigest()
    _require(
        contents[path / "finalize.claim"]
        == completion._canonical(
            {
                "schema_version": 1,
                "reservation_sha256": digest,
                "operation": "failure",
            }
        ),
        "original_failure_claim_mismatch",
    )
    _require(
        contents[path / "outcome.json"]
        == completion._canonical(
            {
                "schema_version": 1,
                "status": "failed",
                "reservation_sha256": digest,
                "stage": "probes",
                "error_type": "Exception",
            }
        ),
        "original_failure_outcome_mismatch",
    )
    expected_details = {
        "schema_version": 1,
        "status": "stopped",
        "check": expected_check,
        "error_type": "Exception",
        "completed_stages": list(SEED_STAGES),
        "failed_stage": "probes",
        "unattempted_stages": [],
        "probe_progress": progress,
    }
    _require(
        contents[path / "failure-details.json"]
        == seed_probe_runner._json_bytes(expected_details),
        "original_failure_details_mismatch",
    )
    return digest


def _read_records(directory, names):
    records = {}
    states = []
    for name in sorted(names):
        metadata = execution_receipt._entry(directory, name)
        _require(metadata is not None, "missing_retained_file")
        state = source_runner._file_state(metadata)
        records[directory.path / name] = source_runner._read_file_once(
            directory.path / name, expected_state=state
        )
        states.append((directory, name, state))
    return records, states


def _process_observations(report, path, contents, root_hash):
    reported = report["process_observation"]["records"]
    for stage, observation in zip(seed_probe_runner.STAGES, reported, strict=True):
        process = source_runner._json(contents[path / f"{stage}-process.json"])
        expected = {
            "schema_version": 1,
            "stage": stage,
            "root_reservation_sha256": root_hash,
            "status": "worker_exited",
            "exit_code": 2 if stage == "probes" else 0,
            "stdout_sha256": observation["stdout_sha256"],
            "stderr_sha256": observation["stderr_sha256"],
        }
        _require(
            completion._same(process, expected)
            and completion._same(
                observation,
                {
                    "stage": stage,
                    "record": f"{stage}-process.json",
                    "exit_code": expected["exit_code"],
                    "stdout_sha256": process["stdout_sha256"],
                    "stderr_sha256": process["stderr_sha256"],
                },
            ),
            "original_process_observation_mismatch",
        )


def _probe_inputs(binding, path, contents):
    primary = dict(binding.seed_probe.primary_artifact_hashes)
    expected = {
        f"input-{name}": primary.get(name)
        for name in (
            "length-only.json",
            "logistic-l1.json",
            "gmm.json",
            "transformer.json",
            "cascade.json",
            "vocabulary.json",
        )
    }
    expected.update(
        {
            "input-training-reference.json": (
                binding.seed_probe.training_reference_sha256
            ),
            "input-validation-audit.json": binding.seed_probe.validation_audit_sha256,
        }
    )
    _require(
        set(expected) == _PROBE_INPUT_RECORDS
        and all(_hex(digest) for digest in expected.values())
        and all(
            sha256(contents[path / "probes" / name]).hexdigest() == digest
            for name, digest in expected.items()
        ),
        "original_probe_input_mismatch",
    )


def _summary(binding, report, members):
    return {
        "schema_version": 1,
        "status": "retained_seed_stages_audited",
        "analysis_stage": "development_validation_only",
        "analysis_role": "secondary_descriptive_only",
        "protected_evaluation_authorized": False,
        "fits": 0,
        "historical_new_fits": 4,
        "original_aggregate_accepted": False,
        "pure_seed_effect_claim": False,
        "seed_selection_performed": False,
        "primary_artifacts_changed": False,
        "original_accounting_sha256": sha256(binding.accounting_bytes).hexdigest(),
        "original_execution": report["execution"],
        "original_reservation_sha256": report["execution"]["root_reservation_sha256"],
        "original_parent_exit_code": 2,
        "original_probe_exit_code": 2,
        "members": members,
        "verification_scope": (
            "saved_evidence_arithmetic_and_receipts_no_refit_or_source_rescoring"
        ),
    }


def _recompute(binding, original_attempt):
    _require(type(binding) is SeedProbeCorrectionBinding, "invalid_audit_binding")
    recheck_seed_probe_correction(binding)
    report = _accounting(binding, binding.accounting_bytes)
    path = execution_receipt._absolute_path(original_attempt)
    _require(
        not path.is_relative_to(binding.base.root), "original_attempt_inside_checkout"
    )
    with ExitStack() as stack:
        root = stack.enter_context(execution_receipt._directory(path))
        probe = stack.enter_context(execution_receipt._directory(path / "probes"))
        completion._directory_contents(root, _ROOT_RECORDS)
        completion._directory_contents(probe, _PROBE_RECORDS)
        root_files = _RECEIPTS | {
            "failure-details.json",
            *(f"{stage}.json" for stage in SEED_STAGES),
            *(f"{stage}-process.json" for stage in seed_probe_runner.STAGES),
        }
        root_contents, states = _read_records(root, root_files)
        probe_contents, probe_states = _read_records(probe, _PROBE_RECORDS)
        states.extend(probe_states)
        root_hash = _failed_receipt(
            root_contents,
            path,
            _original_identity(binding, ORIGINAL_REVISION),
            expected_check="worker_exit_not_successful",
            progress={
                "scope": "installed_record_inventory_not_accepted_scientific_evidence",
                **report["failure"]["probe_progress"],
            },
        )
        _require(
            root_hash == report["execution"]["root_reservation_sha256"],
            "original_failed_reservation_mismatch",
        )
        probe_hash = _failed_receipt(
            probe_contents,
            path / "probes",
            {"root_reservation_sha256": root_hash, "stage": "probes"},
            expected_check="probe_metadata",
            progress={
                "completed_row_prefix": dict.fromkeys(("0", "1", "2", "3"), 0),
                "completed_streams": [],
            },
        )
        _require(
            probe_hash == report["receipt_sha256"]["probes/reservation.json"],
            "original_failed_reservation_mismatch",
        )
        _probe_inputs(binding, path, probe_contents)
        snapshots = {
            stage: stack.enter_context(
                seed_probe_runner._stage_snapshot(
                    execution_receipt.Attempt(path, root_hash), stage
                )
            )
            for stage in SEED_STAGES
        }
        contents = dict(root_contents) | dict(probe_contents)
        for opened in snapshots.values():
            contents.update(opened[0])
        for name, expected in report["receipt_sha256"].items():
            _require(
                sha256(contents[path / name]).hexdigest() == expected,
                "original_receipt_hash_mismatch",
            )
        _process_observations(report, path, contents, root_hash)
        members = []
        for stage, reported in zip(
            SEED_STAGES, report["completed_seed_stages"], strict=True
        ):
            observed = seed_probe_runner._verify_stage(
                binding.seed_probe,
                None,
                execution_receipt.Attempt(path, root_hash),
                stage,
                observed_exit=0,
                _opened=snapshots[stage],
            )
            marker = contents[path / f"{stage}.json"]
            _require(
                sha256(marker).hexdigest() == reported["summary_sha256"]
                and completion._same(observed, reported["summary"]),
                "original_seed_summary_mismatch",
            )
            members.append(
                {
                    "stage": stage,
                    "summary_sha256": reported["summary_sha256"],
                    "summary": observed,
                    "checks": {
                        "saved_stage_receipts": True,
                        "saved_stage_arithmetic": True,
                        "source_reread": False,
                        "url_rescoring": False,
                        "refit": False,
                    },
                }
            )
        result = _summary(binding, report, members)
        recheck_seed_probe_correction(binding)
        completion._directory_contents(root, _ROOT_RECORDS)
        completion._directory_contents(probe, _PROBE_RECORDS)
        for directory, name, state in states:
            current = execution_receipt._entry(directory, name)
            _require(
                current is not None and source_runner._file_state(current) == state,
                "retained_seed_file_changed_during_audit",
            )
    return result


def audit_retained_seed_stages(
    binding: SeedProbeCorrectionBinding, *, original_attempt: Path
) -> dict:
    """Verify all five saved seed stages together without source work or fitting."""
    try:
        return _recompute(binding, original_attempt)
    except SeedProbeAuditError:
        raise
    except Exception:
        raise SeedProbeAuditError("retained_seed_audit_failed") from None


def validate_audit_summary(
    summary: dict,
    *,
    binding: SeedProbeCorrectionBinding,
    original_attempt: Path,
) -> dict:
    """Recompute the whole saved-only audit and compare its exact public result."""
    _require(type(summary) is dict, "invalid_audit_summary")
    expected = audit_retained_seed_stages(binding, original_attempt=original_attempt)
    _require(completion._same(summary, expected), "audit_summary_mismatch")
    return summary
