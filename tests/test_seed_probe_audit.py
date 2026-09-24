"""Saved-only seed audits use invented evidence and failed synthetic receipts."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace

import pytest
from test_seed_probe_seeds import sample  # noqa: F401

from automated_phishing_detection import (
    character_transformer,
    execution_receipt,
    secondary_development,
    secondary_transformer,
    seed_probe_audit,
    seed_probe_correction,
    seed_probe_probes,
    seed_probe_runner,
    seed_probe_seeds,
)
from automated_phishing_detection.seed_probe_execution import SeedProbeExecutionBinding

SEED_STAGES = seed_probe_runner.STAGES[:-1]
ORIGINAL_REVISION = "1135b8e0f0750be7bd83ab0e314c6733ca609eb6"


def _public_bytes(value):
    return (
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    ).encode("ascii")


def _load(path):
    return json.loads(path.read_bytes())


def _hash(path):
    return sha256(path.read_bytes()).hexdigest()


def _seed_binding(seed_sample, root, revision, probe_inputs):
    base = SimpleNamespace(
        root=root,
        revision=revision,
        contract_sha256="c" * 64,
        runtime_json='{"fixture":true}',
    )
    primary_hashes = dict(seed_sample.binding.primary_artifact_hashes)
    primary_hashes.update(
        {
            name.removeprefix("input-"): sha256(content).hexdigest()
            for name, content in probe_inputs.items()
            if name
            not in {"input-training-reference.json", "input-validation-audit.json"}
        }
    )
    return SeedProbeExecutionBinding(
        base=base,
        profile_sha256="4da034b1a46baa599ae04226ee2f4d9a26c2b2d639cac73fa576ff9cb7aa8839",
        base_profile_sha256="d" * 64,
        stopped_attempt_sha256="e" * 64,
        methods_sha256="f" * 64,
        accepted_development_sha256="1" * 64,
        pins=seed_sample.binding.pins,
        preparation_bytes=seed_sample.binding.preparation_bytes,
        transformer_summary_bytes=b"{}",
        primary_artifact_hashes=tuple(sorted(primary_hashes.items())),
        public_operating_points_json="{}",
        retained_drift_summary_json="{}",
        training_reference_sha256=sha256(
            probe_inputs["input-training-reference.json"]
        ).hexdigest(),
        validation_audit_sha256=sha256(
            probe_inputs["input-validation-audit.json"]
        ).hexdigest(),
    )


def _process_record(root, stage):
    record = _load(root / f"{stage}-process.json")
    return {
        "stage": stage,
        "record": f"{stage}-process.json",
        "exit_code": record["exit_code"],
        "stdout_sha256": record["stdout_sha256"],
        "stderr_sha256": record["stderr_sha256"],
    }


def _accounting(binding, root, completed):
    root_hash = _hash(root / "reservation.json")
    receipt_names = [
        "reservation.json",
        "finalize.claim",
        "outcome.json",
        "failure-details.json",
        *(f"{stage}-process.json" for stage in seed_probe_runner.STAGES),
        *(
            f"{stage}/{name}"
            for stage in SEED_STAGES
            for name in ("reservation.json", "finalize.claim", "outcome.json")
        ),
        *(
            f"probes/{name}"
            for name in (
                "reservation.json",
                "finalize.claim",
                "outcome.json",
                "failure-details.json",
            )
        ),
    ]
    progress = seed_probe_runner._observed_probe_progress(
        execution_receipt.Attempt(root, root_hash)
    )
    progress.pop("scope")
    return {
        "schema_version": 1,
        "record_kind": "secondary_seed_probe_attempt_accounting",
        "date": "2026-09-23",
        "status": "failed_partial_evidence_retained",
        "analysis_stage": "development_validation_only",
        "aggregate_accepted": False,
        "execution": {
            "kind": "secondary_seed_probes",
            "revision": ORIGINAL_REVISION,
            "profile_sha256": binding.profile_sha256,
            "base_profile_sha256": binding.base_profile_sha256,
            "stopped_v1_attempt_sha256": binding.stopped_attempt_sha256,
            "methods_sha256": binding.methods_sha256,
            "execution_contract_sha256": binding.base.contract_sha256,
            "runtime_sha256": sha256(binding.base.runtime_json.encode()).hexdigest(),
            "root_reservation_sha256": root_hash,
            "stages": list(seed_probe_runner.STAGES),
            "seed_order": [42, 43, 44, 45, 46],
        },
        "execution_observation": {
            "start_observed_utc": "2026-09-24T00:00:00Z",
            "exit_observed_utc": "2026-09-24T00:01:00Z",
            "parent_exit_code": 2,
            "public_summary_present": False,
            "successful_public_marker_present": False,
            "new_fits": 4,
            "retries": 0,
            "resumes": 0,
            "completed_stages": list(SEED_STAGES),
            "worker_exit_codes": {
                **dict.fromkeys(SEED_STAGES, 0),
                "probes": 2,
            },
        },
        "failure": {
            "stage": "probes",
            "root_safe_check": "worker_exit_not_successful",
            "worker_safe_check": "probe_metadata",
            "root_error_type": "Exception",
            "worker_error_type": "Exception",
            "probe_progress": progress,
        },
        "unattempted_stages": [],
        "completed_seed_stages": [
            {
                "acceptance": (
                    "producer_completed_preliminary_not_independently_accepted"
                ),
                "summary_sha256": _hash(root / f"{summary['stage']}.json"),
                "summary": summary,
            }
            for summary in completed
        ],
        "process_observation": {
            "records": [
                _process_record(root, stage) for stage in seed_probe_runner.STAGES
            ]
        },
        "receipt_sha256": {name: _hash(root / name) for name in receipt_names},
        "access": {
            "training_partition_accessed": True,
            "development_validation_accessed": True,
            "public_suffix_list_accessed": True,
            "group_test_accessed": False,
            "phishvn_accessed": False,
            "external_source_accessed": False,
            "protected_evaluation_accessed": False,
        },
        "authorization": {
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
        },
        "diagnosis": {
            "timing": "post_stop",
            "cause": (
                "public_preparation_report_was_validated_with_private_compact_json_rule"
            ),
            "preparation_summary_sha256": "4" * 64,
            "bound_report_bytes": 4208,
            "compact_reserialization_bytes": 3294,
            "first_divergent_byte": 1,
            "failure_position": "before_probe_row_preparation_or_scoring",
            "retained_drift_artifacts_corrupt": False,
            "test_gap": (
                "invented_retained_drift_fixtures_used_compact_preparation_json"
            ),
            "scope": "fixture diagnosis does not accept probe evidence",
        },
        "verification": {
            "completed_seed_stage_verifier_exit_code": 0,
            "completed_seed_stage_scope": (
                "saved_evidence_arithmetic_and_receipts_no_refit_or_source_rescoring"
            ),
            "complete_family_verification": "not_eligible_producer_exit_2",
            "complete_family_accepted": False,
        },
        "interpretation": [
            "The complete v2 seed/probe family is not accepted.",
            "The seed summaries remain preliminary pending this audit.",
        ],
    }


@pytest.fixture
def stopped_v2(sample, tmp_path, monkeypatch):  # noqa: F811
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bundle = inputs / "bundle"
    bundle.mkdir()
    for name, content in sample.artifacts.items():
        (bundle / name).write_bytes(content)
    metadata = {
        "length-only.json": b'{"invented":"length"}',
        "logistic-l1.json": sample.artifacts["logistic-l1.json"],
        "gmm.json": b'{"invented":"gmm"}',
        "transformer.json": b'{"invented":"transformer"}',
        "cascade.json": b'{"invented":"cascade"}',
        "vocabulary.json": sample.artifacts["vocabulary.json"],
    }
    probe_inputs = {
        **{f"input-{name}": content for name, content in metadata.items()},
        "input-training-reference.json": b'{"invented":"reference"}',
        "input-validation-audit.json": b'{"invented":"audit"}',
    }
    original_binding = _seed_binding(sample, checkout, ORIGINAL_REVISION, probe_inputs)
    paths = seed_probe_runner.SeedProbePaths(
        train=inputs / "train.jsonl",
        validation=inputs / "validation.jsonl",
        suffix_rules=inputs / "suffix.dat",
        length_only=inputs / "unused-length.json",
        logistic_l1=bundle / "logistic-l1.json",
        transformer_bundle=bundle,
        gmm=inputs / "unused-gmm.json",
        drift_reference=inputs / "unused-reference.json",
        drift_audit=inputs / "unused-audit.json",
        attempt=tmp_path / "original-attempt",
        public_summary=tmp_path / "unused-public-summary.json",
    )
    paths.validation.write_bytes(sample.arguments["validation_bytes"])
    paths.suffix_rules.write_bytes(sample.arguments["suffix_rules_bytes"])
    monkeypatch.setattr(seed_probe_runner, "recheck_seed_probe_binding", lambda _: None)
    parent = execution_receipt.reserve_attempt(
        paths.attempt, identity=seed_probe_runner._identity(original_binding)
    )
    completed = []
    completed.append(
        seed_probe_runner._run_worker(original_binding, paths, "seed_42_calibration")
    )
    seed_probe_runner._launch(
        parent, "seed_42_calibration", [sys.executable, "-c", "pass"]
    )
    paths.train.write_bytes(sample.fixture["arguments"]["train_content"])
    monkeypatch.setattr(
        character_transformer, "_run_training_epoch", lambda *args: 0.25
    )
    monkeypatch.setattr(
        character_transformer,
        "_evaluate_validation",
        lambda *args: (1.0, sample.probabilities),
    )
    for stage in SEED_STAGES[1:]:
        completed.append(seed_probe_runner._run_worker(original_binding, paths, stage))
        seed_probe_runner._launch(parent, stage, [sys.executable, "-c", "pass"])
    probe = execution_receipt.reserve_attempt(
        paths.attempt / "probes",
        identity={
            "root_reservation_sha256": parent.reservation_sha256,
            "stage": "probes",
        },
    )
    for name, content in probe_inputs.items():
        seed_probe_runner._record(probe, name, content)
    seed_probe_runner._failure(
        probe,
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
    accounting = _accounting(original_binding, paths.attempt, completed)
    current = _seed_binding(sample, checkout, "a" * 40, probe_inputs)
    binding = seed_probe_correction.SeedProbeCorrectionBinding(
        seed_probe=current,
        profile_sha256=seed_probe_correction.PROFILE_SHA256,
        accounting_bytes=_public_bytes(accounting),
    )
    rechecks = []
    monkeypatch.setattr(
        seed_probe_audit,
        "recheck_seed_probe_correction",
        lambda value: rechecks.append(value),
    )
    return SimpleNamespace(
        binding=binding,
        accounting=accounting,
        original=paths.attempt,
        rechecks=rechecks,
    )


def _inventory(root):
    return {
        str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def _with_accounting(fixture, accounting):
    return replace(fixture.binding, accounting_bytes=_public_bytes(accounting))


# SP-CORR-03: all five retained stages verify together without source work or fitting.
def test_audits_exactly_five_saved_seed_stages_without_source_work_or_fit(
    stopped_v2, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("retained audit attempted source work, fitting, or probe execution")

    for module, name in (
        (seed_probe_seeds, "run_seed_stage"),
        (seed_probe_seeds, "_fit"),
        (secondary_development, "_partition"),
        (secondary_transformer, "score_secondary_stage1_urls"),
        (character_transformer, "fit_character_transformer"),
        (seed_probe_probes, "run_probe_stage"),
    ):
        monkeypatch.setattr(module, name, forbidden)
    before = _inventory(stopped_v2.original)
    result = seed_probe_audit.audit_retained_seed_stages(
        stopped_v2.binding, original_attempt=stopped_v2.original
    )
    assert result["status"] == "retained_seed_stages_audited"
    assert result["fits"] == 0
    assert result["historical_new_fits"] == 4
    assert result["original_aggregate_accepted"] is False
    assert result["pure_seed_effect_claim"] is False
    assert [member["stage"] for member in result["members"]] == list(SEED_STAGES)
    assert "selected_seed" not in result
    assert result["protected_evaluation_authorized"] is False
    assert result["analysis_role"] == "secondary_descriptive_only"
    assert _inventory(stopped_v2.original) == before
    assert stopped_v2.rechecks == [stopped_v2.binding, stopped_v2.binding]


def test_parent_validator_recomputes_the_entire_saved_audit(stopped_v2, monkeypatch):
    result = seed_probe_audit.audit_retained_seed_stages(
        stopped_v2.binding, original_attempt=stopped_v2.original
    )

    def forbidden(*args, **kwargs):
        pytest.fail("parent validation attempted source work or fitting")

    monkeypatch.setattr(secondary_development, "_partition", forbidden)
    monkeypatch.setattr(seed_probe_seeds, "run_seed_stage", forbidden)
    monkeypatch.setattr(seed_probe_seeds, "_fit", forbidden)
    assert (
        seed_probe_audit.validate_audit_summary(
            result,
            binding=stopped_v2.binding,
            original_attempt=stopped_v2.original,
        )
        == result
    )


def test_rejects_missing_reordered_or_extra_seed_stage_as_all_or_none(stopped_v2):
    mutations = []
    missing = deepcopy(stopped_v2.accounting)
    missing["completed_seed_stages"].pop()
    mutations.append(missing)
    reordered = deepcopy(stopped_v2.accounting)
    reordered["completed_seed_stages"][2:4] = reversed(
        reordered["completed_seed_stages"][2:4]
    )
    mutations.append(reordered)
    extra = deepcopy(stopped_v2.accounting)
    extra["completed_seed_stages"].append(deepcopy(extra["completed_seed_stages"][-1]))
    mutations.append(extra)
    for accounting in mutations:
        with pytest.raises(seed_probe_audit.SeedProbeAuditError):
            seed_probe_audit.audit_retained_seed_stages(
                _with_accounting(stopped_v2, accounting),
                original_attempt=stopped_v2.original,
            )


# SP-CORR-04: authenticated root, child, process, stage, and accounting mutations fail.
def test_rejects_root_child_process_and_embedded_summary_hash_mismatch(stopped_v2):
    for location, expected in (
        (("receipt_sha256", "reservation.json"), "original_receipt_hash_mismatch"),
        (
            ("receipt_sha256", "seed_44/outcome.json"),
            "original_receipt_hash_mismatch",
        ),
        (
            ("receipt_sha256", "seed_43-process.json"),
            "original_receipt_hash_mismatch",
        ),
        (
            ("completed_seed_stages", 2, "summary_sha256"),
            "original_seed_summary_mismatch",
        ),
    ):
        accounting = deepcopy(stopped_v2.accounting)
        if len(location) == 2:
            accounting[location[0]][location[1]] = "0" * 64
        else:
            accounting[location[0]][location[1]][location[2]] = "0" * 64
        with pytest.raises(seed_probe_audit.SeedProbeAuditError, match=f"^{expected}$"):
            seed_probe_audit.audit_retained_seed_stages(
                _with_accounting(stopped_v2, accounting),
                original_attempt=stopped_v2.original,
            )


def test_rejects_semantically_invalid_process_after_ledger_rehash(stopped_v2):
    path = stopped_v2.original / "seed_43-process.json"
    process = _load(path)
    process["exit_code"] = 2
    path.write_bytes(seed_probe_runner._json_bytes(process))
    accounting = deepcopy(stopped_v2.accounting)
    accounting["receipt_sha256"]["seed_43-process.json"] = _hash(path)
    assert accounting["receipt_sha256"]["seed_43-process.json"] == _hash(path)
    with pytest.raises(
        seed_probe_audit.SeedProbeAuditError,
        match="^original_process_observation_mismatch$",
    ):
        seed_probe_audit.audit_retained_seed_stages(
            _with_accounting(stopped_v2, accounting),
            original_attempt=stopped_v2.original,
        )


def test_rejects_semantically_invalid_root_receipt_after_ledger_rehash(stopped_v2):
    path = stopped_v2.original / "outcome.json"
    outcome = _load(path)
    outcome["error_type"] = "ValueError"
    path.write_bytes(execution_receipt._json_bytes(outcome, "fixture"))
    accounting = deepcopy(stopped_v2.accounting)
    accounting["receipt_sha256"]["outcome.json"] = _hash(path)
    assert accounting["receipt_sha256"]["outcome.json"] == _hash(path)
    with pytest.raises(
        seed_probe_audit.SeedProbeAuditError,
        match="^original_failure_outcome_mismatch$",
    ):
        seed_probe_audit.audit_retained_seed_stages(
            _with_accounting(stopped_v2, accounting),
            original_attempt=stopped_v2.original,
        )


def test_rejects_nonzero_seed_exit_or_any_probe_row_prefix(stopped_v2):
    nonzero_seed = deepcopy(stopped_v2.accounting)
    nonzero_seed["execution_observation"]["worker_exit_codes"]["seed_45"] = 2
    nonzero_prefix = deepcopy(stopped_v2.accounting)
    nonzero_prefix["failure"]["probe_progress"]["completed_row_prefix"]["1"] = 1
    for accounting in (nonzero_seed, nonzero_prefix):
        with pytest.raises(seed_probe_audit.SeedProbeAuditError):
            seed_probe_audit.audit_retained_seed_stages(
                _with_accounting(stopped_v2, accounting),
                original_attempt=stopped_v2.original,
            )


def test_authenticates_retained_probe_inputs_outside_the_receipt_ledger(stopped_v2):
    assert not any(
        name.startswith("probes/input-")
        for name in stopped_v2.accounting["receipt_sha256"]
    )
    retained = stopped_v2.original / "probes/input-transformer.json"
    retained.write_bytes(retained.read_bytes() + b" ")
    with pytest.raises(
        seed_probe_audit.SeedProbeAuditError, match="original_probe_input_mismatch"
    ):
        seed_probe_audit.audit_retained_seed_stages(
            stopped_v2.binding, original_attempt=stopped_v2.original
        )


def test_holds_every_seed_snapshot_until_final_verification(stopped_v2, monkeypatch):
    verify = seed_probe_runner._verify_stage
    changed = False

    def mutate_after_last(*args, **kwargs):
        nonlocal changed
        result = verify(*args, **kwargs)
        stage = args[3]
        if stage == SEED_STAGES[-1] and not changed:
            changed = True
            retained = stopped_v2.original / SEED_STAGES[0] / "calibration.json"
            retained.write_bytes(retained.read_bytes() + b" ")
        return result

    monkeypatch.setattr(seed_probe_runner, "_verify_stage", mutate_after_last)
    with pytest.raises(seed_probe_audit.SeedProbeAuditError):
        seed_probe_audit.audit_retained_seed_stages(
            stopped_v2.binding, original_attempt=stopped_v2.original
        )
    assert changed is True


def test_parent_rejects_selection_pure_effect_or_partial_member_claims(stopped_v2):
    result = seed_probe_audit.audit_retained_seed_stages(
        stopped_v2.binding, original_attempt=stopped_v2.original
    )
    mutations = []
    selected = deepcopy(result)
    selected["selected_seed"] = 44
    mutations.append(selected)
    pure_effect = deepcopy(result)
    pure_effect["pure_seed_effect_claim"] = True
    mutations.append(pure_effect)
    partial = deepcopy(result)
    partial["members"].pop()
    mutations.append(partial)
    for summary in mutations:
        with pytest.raises(seed_probe_audit.SeedProbeAuditError):
            seed_probe_audit.validate_audit_summary(
                summary,
                binding=stopped_v2.binding,
                original_attempt=stopped_v2.original,
            )


@pytest.mark.parametrize("mutation", ["duplicate_key", "nonfinite_value"])
def test_accounting_parser_rejects_production_shaped_invalid_json(stopped_v2, mutation):
    content = stopped_v2.binding.accounting_bytes
    if mutation == "duplicate_key":
        content = content.replace(
            b'{\n  "schema_version": 1,',
            b'{\n  "schema_version": 1,\n  "schema_version": 1,',
            1,
        )
        assert json.loads(content) == stopped_v2.accounting
    else:
        content = content.replace(
            b'    "bound_report_bytes": 4208,',
            b'    "bound_report_bytes": NaN,',
            1,
        )
        assert math.isnan(json.loads(content)["diagnosis"]["bound_report_bytes"])
    with pytest.raises(
        seed_probe_audit.SeedProbeAuditError,
        match="^retained_seed_audit_failed$",
    ):
        seed_probe_audit.audit_retained_seed_stages(
            replace(stopped_v2.binding, accounting_bytes=content),
            original_attempt=stopped_v2.original,
        )
