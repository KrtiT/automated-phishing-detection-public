import json
from hashlib import sha256
from pathlib import Path

from automated_phishing_detection import secondary_probes

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "data/secondary-seed-probe-contract-v1.json"


def contract():
    return json.loads(PATH.read_bytes())


def test_prospective_methods_have_fixed_bytes():
    assert sha256(PATH.read_bytes()).hexdigest() == (
        "eb279404728e498999fc7fd0c7578291373bb80b9816f88b5d7202dfdf637380"
    )


def test_supplement_preserves_existing_methods_and_results():
    value = contract()
    assert value["contract_id"] == "secondary-seed-probe-v1"
    assert value["protected_evaluation_ready"] is False
    assert value["research_execution_ready"] is False
    assert value["primary_changes"] is False
    for name, digest in value["public_file_sha256"].items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_seed_family_does_not_refit_or_replace_primary_seed():
    value = contract()["seeds"]
    assert value["report_order"] == [42, 43, 44, 45, 46]
    assert value["new_fit_seeds"] == [43, 44, 45, 46]
    assert value["primary_seed_42"] == "reuse_accepted_weights_and_vocabulary"
    assert value["select_best_seed"] is False
    assert value["primary_entry_seed"] == 42
    assert (
        value["vocabulary"] == "reuse_accepted_training_only_vocabulary_for_all_seeds"
    )
    assert value["runtime_contract"] == "execution-binding-v2"
    assert value["training_batch_size"] == 256
    assert value["checkpoint_scoring_batch_size"] == 512
    assert value["secondary_scoring_batch_size"] == 1
    assert value["maximum_epochs"] == 40
    assert value["patience"] == 5
    assert value["ap_min_delta"] == 0.0001
    assert value["training_generator"] == "one_cpu_torch_generator_seeded_once_per_fit"
    assert value["pure_seed_effect_claim"] is False


def test_secondary_calibration_keeps_primary_stage_one_cutoff():
    value = contract()["seeds"]["calibration"]
    assert value["stage1_model"] == "reuse_accepted_logistic_l1"
    assert value["stage1_cutoff"] == "carry_forward_accepted_historical_cutoff"
    assert value["stage1_cutoff_reselected"] is False
    assert value["transformer_cutoff"] == "per_seed_validation_cp_rule"
    assert value["cascade_band"] == "per_seed_original_inclusive_band_sweep"
    assert value["cascade_maximum_fpr_upper_95"] == 0.01
    assert value["cascade_recall_tolerance"] == 0.02
    assert value["write_primary_artifacts"] is False


def test_probe_streams_keep_positions_and_fixed_monitor_coverage():
    value = contract()["probes"]
    assert value["streams"] == ["original", *secondary_probes.PERTURBATION_OPERATORS]
    assert value["population"] == "original_development_validation_audit_order"
    assert value["operator_composition"] is False
    assert value["retain_ineligible_and_noop_rows"] is True
    assert value["inherit_outcome_labels"] is False
    assert value["detectors"] == [
        "length",
        "logistic_l1",
        "transformer_42",
        "fixed_cascade",
        "gmm_policy",
    ]
    assert value["monitors"] == ["gmm", "mmd", "psi"]
    assert value["window_length"] == 256
    assert value["window_stride"] == 64
    assert value["reset"] == "empty_monitor_and_routing_history_per_stream"
    assert (
        value["references_and_boundaries"]
        == "reuse_accepted_bytes_without_refit_or_recalibration"
    )
    assert value["h2_replacement"] is False
    assert value["http_measurement"] is False
    assert value["adversarial_success_claim"] is False


def test_probe_comparisons_are_paired_and_descriptive():
    value = contract()["probes"]["comparisons"]
    assert value["pairing"] == "original_record_id_and_stream_position"
    assert value["delta"] == "transformed_minus_original"
    assert value["changed_score"] == "exact_float_inequality_without_tolerance"
    assert value["mean"] == "math_fsum_in_input_order_divided_by_count"
    assert value["transition_cells"] == ["00", "01", "10", "11"]
    assert value["inference"] == "descriptive_only_no_pvalue_or_interval"
    assert value["absent_complete_windows"] == "not_estimable_not_zero_alert_success"


def test_new_profile_does_not_extend_completed_execution_permissions():
    value = contract()["execution"]
    assert value["research_input_roles"] == ["train", "validation"]
    assert value["forbidden_roles"] == ["group_test", "external", "PhishVN"]
    assert value["failure_policy"] == "fail_stop_no_retry_no_resume"
    assert value["retain_actual_worker_exit"] is True
    assert value["retain_checkpoint_before_later_checks"] is True
    assert value["retained_attempts_mutable"] is False
    assert value["entry_point"] == "separate_authenticated_runner_required"


def test_implemented_constants_match_declared_methods():
    from automated_phishing_detection import (
        character_transformer,
        probe_replay,
        secondary_transformer,
    )

    value = contract()
    assert (
        list(secondary_transformer.SECONDARY_SEEDS) == value["seeds"]["new_fit_seeds"]
    )
    assert character_transformer.SEED == value["seeds"]["primary_entry_seed"]
    assert (
        character_transformer.TRAIN_BATCH_SIZE == value["seeds"]["training_batch_size"]
    )
    assert (
        character_transformer.VALIDATION_BATCH_SIZE
        == value["seeds"]["checkpoint_scoring_batch_size"]
    )
    assert character_transformer.MAX_EPOCHS == value["seeds"]["maximum_epochs"]
    assert character_transformer.PATIENCE == value["seeds"]["patience"]
    assert character_transformer.MIN_DELTA == value["seeds"]["ap_min_delta"]
    assert probe_replay.CONTRACT_ID == value["contract_id"]
    assert list(probe_replay.STREAM_NAMES) == value["probes"]["streams"]
    assert list(probe_replay.DETECTOR_NAMES) == value["probes"]["detectors"]
    assert list(probe_replay.MONITOR_NAMES) == value["probes"]["monitors"]


def test_reused_primary_weights_and_vocabulary_match_accepted_record():
    accepted = json.loads(
        (ROOT / "reports/rq1-transformer-cascade-v2-summary.json").read_bytes()
    )
    value = contract()["seeds"]
    assert (
        value["primary_weight_sha256"]
        == accepted["artifact_hashes"]["transformer-weights.npz"]
    )
    assert value["vocabulary_sha256"] == accepted["artifact_hashes"]["vocabulary.json"]
    assert (
        "every epoch's batch-512 validation probabilities"
        in value["calibration"]["scoring_outputs"]
    )
