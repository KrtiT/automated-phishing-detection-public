"""Static accounting checks for stopped seed/probe execution attempts."""

import json
from hashlib import sha256
from pathlib import Path

from automated_phishing_detection import execution_receipt

ROOT = Path(__file__).resolve().parents[1]
V1_ATTEMPT = ROOT / "reports/secondary-seed-probe-v1-attempt-1.json"
V2_ATTEMPT = ROOT / "reports/secondary-seed-probe-v2-attempt-1.json"
V2_ATTEMPT_SHA256 = "cf1fc0e6e41839464def2955b4475492b4839e637d4e563d4c94324057e74cb4"
CORRECTION_REPORT = ROOT / "reports/secondary-seed-probe-correction-v1-summary.json"
CORRECTION_REPORT_SHA256 = (
    "d63a85792088871cfb667e7e5cbe86c6148dc3a6c7430ca2e4db29d788ab8e23"
)


def _digest(value):
    return sha256(execution_receipt._json_bytes(value, "summary")).hexdigest()


def _contains_key(value, forbidden):
    if type(value) is dict:
        return forbidden in value or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if type(value) is list:
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_stopped_seed_probe_v1_attempt_is_immutable_and_unaccepted():
    assert V1_ATTEMPT.exists(), "missing stopped seed/probe v1 attempt record"
    record = json.loads(V1_ATTEMPT.read_bytes())

    assert set(record) == {
        "schema_version",
        "record_kind",
        "date",
        "status",
        "aggregate_accepted",
        "execution",
        "execution_observation",
        "failure",
        "unattempted_stages",
        "process_observation",
        "receipt_sha256",
        "diagnosis",
        "authorization",
        "interpretation",
    }
    assert record["schema_version"] == 1
    assert record["record_kind"] == "secondary_seed_probe_attempt_accounting"
    assert record["date"] == "2026-09-23"
    assert record["status"] == "stopped_invalid_evidence_json"
    assert record["aggregate_accepted"] is False
    assert record["execution"] == {
        "kind": "secondary_seed_probes",
        "revision": "2ccc56867bf1ce9e963bcd47273925381a241961",
        "profile_sha256": (
            "cf18fa8c35039c63f896cc62c7aaac8b0847a1abf55676ba67ee65b42340381d"
        ),
        "methods_sha256": (
            "eb279404728e498999fc7fd0c7578291373bb80b9816f88b5d7202dfdf637380"
        ),
        "execution_contract_sha256": (
            "887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1"
        ),
        "runtime_sha256": (
            "f60e3c1486996dab72f10090de1ac615806fe53244ceb92dd571a40341803d86"
        ),
        "stages": [
            "seed_42_calibration",
            "seed_43",
            "seed_44",
            "seed_45",
            "seed_46",
            "probes",
        ],
    }
    assert record["execution_observation"] == {
        "start_observed_utc": "2026-09-23T22:12:49Z",
        "exit_observed_utc": "2026-09-23T22:13:12Z",
        "parent_exit_code": 2,
        "worker_exit_code": 2,
        "public_summary_present": False,
        "successful_public_marker_present": False,
        "new_fits": 0,
        "retries": 0,
        "resumes": 0,
        "completed_stages": [],
    }
    assert record["failure"] == {
        "stage": "seed_42_calibration",
        "safe_check": "invalid_evidence_json",
        "root_error_type": "Exception",
    }
    assert record["unattempted_stages"] == [
        "seed_43",
        "seed_44",
        "seed_45",
        "seed_46",
        "probes",
    ]
    assert record["process_observation"] == {
        "record": "seed_42_calibration-process.json",
        "stdout_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "stderr_sha256": (
            "4fc57bd849fdb111eba71f1750238d6252d37efc82297224023973616ed167a2"
        ),
    }
    assert record["receipt_sha256"] == {
        "reservation.json": (
            "73c2f55952fa6e4b7beab7168c67ec00b4487f70022f1914e5dfdf6f44e46285"
        ),
        "finalize.claim": (
            "ebef1a7ae502bbbe7b82280336fc113ba5b4fea1942606ecd63166bd667529a5"
        ),
        "outcome.json": (
            "338f36b085c55c67ec36525d2872392d33c6e16d94406b37755742d77cd6f14a"
        ),
        "failure-details.json": (
            "d5262fcd3b85e0b1a0a58c6af006f30b08132f47f333b83d257bb20663a61828"
        ),
        "seed_42_calibration-process.json": (
            "dc12819722abff0875f01b817c6cff84ebd8f1330d46f1b779cc0ed7c3a07dc1"
        ),
        "seed_42_calibration/reservation.json": (
            "44850820e04518dfe9ff445654aa636ed39dfa2277c5e949b2c8503e0ea7ec96"
        ),
        "seed_42_calibration/finalize.claim": (
            "0f1c6e3f01feeadde96542c180ef8db7492ca03fe254b30edff4f1395b2dce06"
        ),
        "seed_42_calibration/outcome.json": (
            "c51f72b602ebfba26b7beaa7da17e9c808f1785ba1033ed1bd00bf2ec68a252a"
        ),
        "seed_42_calibration/failure-details.json": (
            "74c8143a6f959dc3892f5fd8b0798e9a44d6128899ead790e80eeab99e11441c"
        ),
    }
    assert record["diagnosis"] == {
        "timing": "post_stop",
        "source_canonical_form": "established_ascii_escaped_jsonl",
        "runner_canonical_form": "artifact_utf8_canonical_reserialization",
        "reproduced_with": "invented_unicode_fixture",
        "scope": (
            "This diagnosis explains the evidence-JSON validation failure. It "
            "does not establish whether calibration computation occurred before "
            "the stop."
        ),
    }
    assert record["authorization"] == {
        "profile_status": "exhausted",
        "fresh_execution_authorized": False,
        "required_before_any_fresh_execution": [
            "separate_prospective_correction_profile_frozen",
            "reviewed",
            "published",
            "ci_passed",
        ],
    }
    assert "completion" not in record
    assert "completion_summary_sha256" not in record
    assert record["interpretation"] == [
        "No fit occurred.",
        "No stage was accepted.",
        "This record does not establish whether calibration computation occurred.",
        "No seed/probe research result is accepted from this attempt.",
    ]


def test_stopped_seed_probe_v2_attempt_preserves_preliminary_seed_prefix():
    assert V2_ATTEMPT.exists(), "missing stopped seed/probe v2 attempt record"
    assert sha256(V2_ATTEMPT.read_bytes()).hexdigest() == V2_ATTEMPT_SHA256
    record = json.loads(V2_ATTEMPT.read_bytes())

    assert set(record) == {
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
        "diagnosis",
        "verification",
        "access",
        "authorization",
        "interpretation",
    }
    assert record["schema_version"] == 1
    assert record["record_kind"] == "secondary_seed_probe_attempt_accounting"
    assert record["date"] == "2026-09-23"
    assert record["status"] == "failed_partial_evidence_retained"
    assert record["analysis_stage"] == "development_validation_only"
    assert record["aggregate_accepted"] is False

    stages = [
        "seed_42_calibration",
        "seed_43",
        "seed_44",
        "seed_45",
        "seed_46",
        "probes",
    ]
    completed = stages[:-1]
    assert record["execution"] == {
        "kind": "secondary_seed_probes",
        "revision": "1135b8e0f0750be7bd83ab0e314c6733ca609eb6",
        "profile_sha256": (
            "4da034b1a46baa599ae04226ee2f4d9a26c2b2d639cac73fa576ff9cb7aa8839"
        ),
        "base_profile_sha256": (
            "cf18fa8c35039c63f896cc62c7aaac8b0847a1abf55676ba67ee65b42340381d"
        ),
        "stopped_v1_attempt_sha256": (
            "3be65bd38c32b8bf8aafa06eede3577a0d1acc212f052d2d9f60768183f535c6"
        ),
        "methods_sha256": (
            "eb279404728e498999fc7fd0c7578291373bb80b9816f88b5d7202dfdf637380"
        ),
        "execution_contract_sha256": (
            "887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1"
        ),
        "runtime_sha256": (
            "f60e3c1486996dab72f10090de1ac615806fe53244ceb92dd571a40341803d86"
        ),
        "root_reservation_sha256": (
            "e3963a2a9ea84f37d2f28ec07c0aadbbf6cf98dacdbf0f6e3459fb91ceee79f0"
        ),
        "stages": stages,
        "seed_order": [42, 43, 44, 45, 46],
    }
    assert record["execution_observation"] == {
        "start_observed_utc": "2026-09-24T00:14:29Z",
        "exit_observed_utc": "2026-09-24T07:56:27Z",
        "parent_exit_code": 2,
        "public_summary_present": False,
        "successful_public_marker_present": False,
        "new_fits": 4,
        "retries": 0,
        "resumes": 0,
        "completed_stages": completed,
        "worker_exit_codes": {**dict.fromkeys(completed, 0), "probes": 2},
    }
    assert type(record["execution_observation"]["parent_exit_code"]) is int
    assert all(
        type(value) is int
        for value in record["execution_observation"]["worker_exit_codes"].values()
    )
    assert record["failure"] == {
        "stage": "probes",
        "root_safe_check": "worker_exit_not_successful",
        "worker_safe_check": "probe_metadata",
        "root_error_type": "Exception",
        "worker_error_type": "Exception",
        "probe_progress": {
            "completed_row_prefix": {"0": 0, "1": 0, "2": 0, "3": 0},
            "observed_completed_stream_records": [],
            "unaccepted_row_records_outside_prefix": 0,
            "inventory_incomplete": False,
            "failed_row_position": None,
            "durability_confirmed": False,
        },
    }
    assert record["unattempted_stages"] == []

    summaries = record["completed_seed_stages"]
    assert [entry["summary"]["stage"] for entry in summaries] == completed
    assert [entry["summary"]["result"]["seed"] for entry in summaries] == [
        42,
        43,
        44,
        45,
        46,
    ]
    assert [entry["summary"]["result"]["new_fit"] for entry in summaries] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert sum(entry["summary"]["result"]["new_fit"] for entry in summaries) == 4
    expected_summary_hashes = {
        "seed_42_calibration": (
            "ef8431a5c2308a5023a6e409a22ed388d26745d236047827704ce10b1fc0d5a6"
        ),
        "seed_43": ("4cb3b1080b7484dda827f567598dcd61fc9ad9e97a702639388f92431807d66e"),
        "seed_44": ("89aab1f47dbceb82096948f994843276c13789c923e170432b638cb385783b11"),
        "seed_45": ("85bb4d8063c6de6a143c2aa08eb322998c7b227066af18d4eaf1f1cca4a6f4cd"),
        "seed_46": ("16ebf6c41de836aa124a5b7cbb95b8bb14478d276d356a640b114cf30243c5b5"),
    }
    for entry in summaries:
        summary = entry["summary"]
        assert set(entry) == {"acceptance", "summary_sha256", "summary"}
        assert entry["acceptance"] == (
            "producer_completed_preliminary_not_independently_accepted"
        )
        assert summary["status"] == "completed_secondary_seed_probe_stage"
        assert entry["summary_sha256"] == expected_summary_hashes[summary["stage"]]
        assert _digest(summary) == entry["summary_sha256"]
        assert summary["result"]["primary_artifacts_changed"] is False
        assert summary["result"]["pure_seed_effect_claim"] is False
    assert summaries[0]["summary"]["result"]["best_epoch"] is None
    assert summaries[0]["summary"]["result"]["epochs_completed"] == 0
    assert all(
        type(entry["summary"]["result"]["best_epoch"]) is int
        and entry["summary"]["result"]["best_epoch"] > 0
        and type(entry["summary"]["result"]["epochs_completed"]) is int
        and entry["summary"]["result"]["epochs_completed"] > 0
        for entry in summaries[1:]
    )
    assert not _contains_key(record, "selected_seed")
    assert "completion" not in record
    assert "completion_summary_sha256" not in record
    assert "probe_summary" not in record

    assert record["verification"] == {
        "completed_seed_stage_verifier_exit_code": 0,
        "completed_seed_stage_scope": (
            "saved_evidence_arithmetic_and_receipts_no_refit_or_source_rescoring"
        ),
        "complete_family_verification": "not_eligible_producer_exit_2",
        "complete_family_accepted": False,
    }
    assert record["access"] == {
        "training_partition_accessed": True,
        "development_validation_accessed": True,
        "public_suffix_list_accessed": True,
        "group_test_accessed": False,
        "phishvn_accessed": False,
        "external_source_accessed": False,
        "protected_evaluation_accessed": False,
    }
    assert record["authorization"] == {
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
    assert record["interpretation"] == [
        "The complete v2 seed/probe family is not accepted.",
        "All five seed stages completed and were supervisor-verified before the probe failure, but remain preliminary pending a separate retained-evidence audit.",
        "Seeds 43-46 account for exactly four new fits; seed 42 reused the accepted weights without fitting.",
        "No best seed is selected, no pure causal seed-effect claim is made, and no primary artifact changed.",
        "The probe worker stopped before row preparation or scoring, so no probe scientific result is accepted.",
        "No protected, group-test, PhishVN, external, HTTP, operational, adversarial-success, or replacement-H2 claim follows from this attempt.",
    ]


def test_seed_probe_correction_report_is_exact_accepted_development_evidence():
    assert CORRECTION_REPORT.exists(), "missing seed/probe correction report"
    assert sha256(CORRECTION_REPORT.read_bytes()).hexdigest() == (
        CORRECTION_REPORT_SHA256
    )
    record = json.loads(CORRECTION_REPORT.read_bytes())

    assert list(record) == [
        "schema_version",
        "record_kind",
        "date",
        "status",
        "execution_observation",
        "completion_summary_sha256",
        "completion",
        "receipt_sha256",
        "verification",
        "access",
        "authorization",
        "interpretation",
    ]
    assert record["schema_version"] == 1
    assert record["record_kind"] == "secondary_seed_probe_correction_execution"
    assert record["date"] == "2026-09-24"
    assert record["status"] == "accepted_development_evidence"

    observation = record["execution_observation"]
    assert observation == {
        "producer_start_observed_utc": "2026-09-24T18:12:32Z",
        "producer_exit_observed_utc": "2026-09-24T19:01:50Z",
        "parent_exit_code": 0,
        "verifier_start_observed_utc": "2026-09-24T19:07:17Z",
        "verifier_exit_observed_utc": "2026-09-24T19:12:53Z",
        "verification_exit_code": 0,
        "worker_exit_codes": {"retained_seed_audit": 0, "probes": 0},
        "new_fits": 0,
        "seed_stage_executions": 0,
        "probe_executions": 1,
        "retries": 0,
        "resumes": 0,
    }
    assert type(observation["parent_exit_code"]) is int
    assert type(observation["verification_exit_code"]) is int
    assert all(type(code) is int for code in observation["worker_exit_codes"].values())

    completion = record["completion"]
    assert record["completion_summary_sha256"] == (
        "4b20d8877d59e7c80b384d09596102552bbeae320ab8b01aa32aa3109bf2ab75"
    )
    assert _digest(completion) == record["completion_summary_sha256"]
    assert completion["worker_exit_codes"] == {
        "probes": 0,
        "retained_seed_audit": 0,
    }
    assert all(type(code) is int for code in completion["worker_exit_codes"].values())

    receipts = {
        "reservation.json": (
            "0732c4280d825baecce3f0c5c9d41da929788b0f590a822ef3b82e1d9858f25f"
        ),
        "finalize.claim": (
            "e50120f74b8c5664785c7f385b1daa1e11633f7d84f9752568ea136c696a5b56"
        ),
        "outcome.json": (
            "2f16a57c3ad2dba3df83427752ac8f31168715e209801fc806db8435f2147d96"
        ),
        "retained_seed_audit-process.json": (
            "c0e9ec933e12ed50f751052d9ed73d80687c066e3d5ba02fcb3245aef0a67d8a"
        ),
        "probes-process.json": (
            "dae94b80241458be2a138b55f751b52f75f72ff29fcd8b8402c660e9ac594297"
        ),
        "retained_seed_audit.json": (
            "f4f563149d36975f1d65fd9e751b1f831bfb6eb16c343ce7e5ffbbd726ac7a44"
        ),
        "probes.json": (
            "d5286418c92ac42d038fc2df9b7c54bc2d9e841c4de7e68dd05adbc7d3498f7c"
        ),
        "evidence/stage-summaries.json": (
            "b2d57b1e97d35e99593bae1041ea89178172bf9f962eab4a741782452f154fed"
        ),
    }
    assert record["receipt_sha256"] == receipts
    assert (
        _digest(completion["retained_seed_audit"])
        == receipts["retained_seed_audit.json"]
    )
    assert _digest(completion["probes"]) == receipts["probes.json"]

    assert (
        completion["new_fits"],
        completion["seed_stage_executions"],
        completion["probe_executions"],
        completion["retries"],
    ) == (0, 0, 1, 0)
    assert (
        observation["new_fits"],
        observation["seed_stage_executions"],
        observation["probe_executions"],
        observation["retries"],
    ) == (0, 0, 1, 0)
    assert observation["resumes"] == 0

    audit = completion["retained_seed_audit"]["result"]
    assert type(audit["original_parent_exit_code"]) is int
    assert type(audit["original_probe_exit_code"]) is int
    assert audit["original_parent_exit_code"] == 2
    assert audit["original_probe_exit_code"] == 2
    members = audit["members"]
    expected_member_hashes = {
        "seed_42_calibration": (
            "ef8431a5c2308a5023a6e409a22ed388d26745d236047827704ce10b1fc0d5a6"
        ),
        "seed_43": "4cb3b1080b7484dda827f567598dcd61fc9ad9e97a702639388f92431807d66e",
        "seed_44": "89aab1f47dbceb82096948f994843276c13789c923e170432b638cb385783b11",
        "seed_45": "85bb4d8063c6de6a143c2aa08eb322998c7b227066af18d4eaf1f1cca4a6f4cd",
        "seed_46": "16ebf6c41de836aa124a5b7cbb95b8bb14478d276d356a640b114cf30243c5b5",
    }
    assert [member["stage"] for member in members] == list(expected_member_hashes)
    assert len(members) == 5
    for member in members:
        assert member["checks"] == {
            "refit": False,
            "saved_stage_arithmetic": True,
            "saved_stage_receipts": True,
            "source_reread": False,
            "url_rescoring": False,
        }
        assert member["summary_sha256"] == expected_member_hashes[member["stage"]]
        assert _digest(member["summary"]) == member["summary_sha256"]
    assert not _contains_key(record, "selected_seed")
    assert completion["original_v2_aggregate_accepted"] is False
    assert completion["original_v2_profile_status"] == "exhausted"
    assert audit["original_aggregate_accepted"] is False
    assert audit["fits"] == 0
    assert audit["seed_selection_performed"] is False
    assert audit["pure_seed_effect_claim"] is False

    assert record["verification"] == {
        "root_receipt_and_stage_linkage_verified": True,
        "saved_verifier_exit_code": 0,
        "observed_zero_worker_exits": True,
        "retained_seed_audit_all_or_none": True,
        "retained_seed_stage_count": 5,
        "saved_probe_evidence_verified": True,
        "original_v2_preserved_failed": True,
        "correction_time_refit": False,
        "retained_seed_audit_source_partition_reread": False,
        "retained_seed_audit_url_rescoring": False,
        "probe_development_validation_source_read": True,
        "probe_url_scoring_performed": True,
        "saved_verifier_source_partition_reread": False,
        "saved_verifier_saved_primary_probabilities_reused": True,
        "saved_verifier_retained_url_structural_portable_gmm_rescoring": True,
        "protected_records_accessed": False,
    }
    assert type(record["verification"]["saved_verifier_exit_code"]) is int
    assert record["access"] == {
        "training_partition_accessed": False,
        "development_validation_accessed": True,
        "public_suffix_list_accessed": True,
        "group_test_accessed": False,
        "phishvn_accessed": False,
        "external_source_accessed": False,
        "protected_evaluation_accessed": False,
    }
    assert record["authorization"] == {
        "profile_status": "exhausted",
        "fresh_execution_authorized": False,
        "retry_authorized": False,
        "resume_authorized": False,
        "refit_authorized": False,
        "additional_probe_authorized": False,
        "protected_evaluation_authorized": False,
    }

    stream_list = completion["probes"]["result"]["result"]["streams"]
    assert [stream["name"] for stream in stream_list] == [
        "original",
        "ascii_scheme_host_uppercase",
        "percent_escape_hex_uppercase",
        "first_literal_path_alphanumeric_percent_encode",
    ]
    streams = {stream["name"]: stream for stream in stream_list}
    assert {name: stream["mapping_counts"] for name, stream in streams.items()} == {
        "original": {
            "changed": 0,
            "eligible": 16370,
            "eligible_noop": 16370,
            "ineligible": 0,
        },
        "ascii_scheme_host_uppercase": {
            "changed": 16370,
            "eligible": 16370,
            "eligible_noop": 0,
            "ineligible": 0,
        },
        "percent_escape_hex_uppercase": {
            "changed": 22,
            "eligible": 53,
            "eligible_noop": 31,
            "ineligible": 16317,
        },
        "first_literal_path_alphanumeric_percent_encode": {
            "changed": 1563,
            "eligible": 1563,
            "eligible_noop": 0,
            "ineligible": 14807,
        },
    }

    detector_names = [
        "fixed_cascade",
        "gmm_policy",
        "length",
        "logistic_l1",
        "transformer_42",
    ]
    no_decision_changes = dict.fromkeys(detector_names, {"01": 0, "10": 0})
    for name in (
        "original",
        "ascii_scheme_host_uppercase",
        "percent_escape_hex_uppercase",
    ):
        paired = streams[name]["paired_with_original"]["detectors"]
        assert {
            detector: {
                direction: paired[detector]["decision_transitions"][direction]
                for direction in ("01", "10")
            }
            for detector in detector_names
        } == no_decision_changes
    path_detectors = streams["first_literal_path_alphanumeric_percent_encode"][
        "paired_with_original"
    ]["detectors"]
    assert {
        detector: {
            direction: path_detectors[detector]["decision_transitions"][direction]
            for direction in ("01", "10")
        }
        for detector in detector_names
    } == {
        "fixed_cascade": {"01": 0, "10": 0},
        "gmm_policy": {"01": 105, "10": 63},
        "length": {"01": 48, "10": 0},
        "logistic_l1": {"01": 0, "10": 0},
        "transformer_42": {"01": 14, "10": 0},
    }

    monitor_names = ["gmm", "mmd", "psi"]
    expected_monitor_transitions = {
        "original": {
            "gmm": {"01": 0, "10": 0},
            "mmd": {"01": 0, "10": 0},
            "psi": {"01": 0, "10": 0},
        },
        "ascii_scheme_host_uppercase": {
            "gmm": {"01": 1, "10": 0},
            "mmd": {"01": 7, "10": 0},
            "psi": {"01": 0, "10": 1},
        },
        "percent_escape_hex_uppercase": {
            "gmm": {"01": 0, "10": 0},
            "mmd": {"01": 0, "10": 0},
            "psi": {"01": 0, "10": 0},
        },
        "first_literal_path_alphanumeric_percent_encode": {
            "gmm": {"01": 185, "10": 0},
            "mmd": {"01": 6, "10": 0},
            "psi": {"01": 0, "10": 5},
        },
    }
    for name, expected in expected_monitor_transitions.items():
        monitors = streams[name]["paired_with_original"]["monitors"]
        assert {
            monitor: {
                direction: monitors[monitor]["alert_transitions"][direction]
                for direction in ("01", "10")
            }
            for monitor in monitor_names
        } == expected
    assert record["interpretation"] == {
        "scope": (
            "Secondary descriptive development-validation evidence only; no "
            "group-test, PhishVN, external, protected, operational, or production "
            "claim."
        ),
        "retained_seeds": (
            "The all-or-none retained-stage audit accepts the five retained stage "
            "records without a refit, source-partition reread, URL rescoring, seed "
            "selection, or a pure seed-effect claim. The original v2 execution "
            "remains failed and unaccepted, and its profile remains exhausted."
        ),
        "probes": (
            "The label-free replay is not a correctness, adversarial-success, "
            "semantic-equivalence, production, or H2 result. ASCII scheme/host "
            "uppercasing preserved detector decisions but changed some scores and "
            "monitor outcomes; percent-escape uppercasing changed only 22 rows; "
            "first-literal path encoding changed several detector decisions and "
            "monitor alerts."
        ),
        "preservation": (
            "The failed v2 record is preserved. The separate correction authority "
            "is consumed; no fresh execution, retry, resume, refit, additional "
            "probe, or protected evaluation is authorized."
        ),
    }


def test_seed_probe_correction_verification_scopes_source_reads_and_scoring():
    record = json.loads(CORRECTION_REPORT.read_bytes())
    verification = record["verification"]

    assert "correction_time_source_reread" not in verification
    assert "correction_time_url_rescoring" not in verification
    assert "saved_verifier_url_rescoring" not in verification
    assert verification == {
        "root_receipt_and_stage_linkage_verified": True,
        "saved_verifier_exit_code": 0,
        "observed_zero_worker_exits": True,
        "retained_seed_audit_all_or_none": True,
        "retained_seed_stage_count": 5,
        "saved_probe_evidence_verified": True,
        "original_v2_preserved_failed": True,
        "correction_time_refit": False,
        "retained_seed_audit_source_partition_reread": False,
        "retained_seed_audit_url_rescoring": False,
        "probe_development_validation_source_read": True,
        "probe_url_scoring_performed": True,
        "saved_verifier_source_partition_reread": False,
        "saved_verifier_saved_primary_probabilities_reused": True,
        "saved_verifier_retained_url_structural_portable_gmm_rescoring": True,
        "protected_records_accessed": False,
    }
