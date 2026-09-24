"""Static accounting checks for stopped seed/probe execution attempts."""

import json
from hashlib import sha256
from pathlib import Path

from automated_phishing_detection import execution_receipt

ROOT = Path(__file__).resolve().parents[1]
V1_ATTEMPT = ROOT / "reports/secondary-seed-probe-v1-attempt-1.json"
V2_ATTEMPT = ROOT / "reports/secondary-seed-probe-v2-attempt-1.json"
V2_ATTEMPT_SHA256 = "cf1fc0e6e41839464def2955b4475492b4839e637d4e563d4c94324057e74cb4"


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
