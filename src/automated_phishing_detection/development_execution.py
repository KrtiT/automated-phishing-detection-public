"""Authenticate the narrow secondary-development procedure using public bytes.

The original preflight and protected-evaluation gate are unchanged. This binding
adds two committed contracts and checks their links to accepted development
sources/models; it never inspects a supplied data or model path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from . import baselines, execution_preflight, fixed_cascade, phiusiil
from .execution_preflight import ExecutionBinding
from .execution_preflight import _bind_historical_v2_execution as bind_execution
from .execution_preflight import _recheck_historical_v2_binding as recheck_binding
from .secondary_development import DevelopmentPins

PROFILE_PATH = "data/development-execution-contract-v1.json"
METHODS_PATH = "data/secondary-development-contract-v1.json"
BASE_PROFILE_SHA256 = "887f771381927dfe1b9268a45f4e605baf3e9a7caee2b7005cdfe68b1be516e1"
METHODS_SHA256 = "f592352593ae64b178a468d5800e780267275a62046a05b31b952f69e424f44f"
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
_SOURCE = "data/sources.json"
_PREPARATION = "reports/phiusiil-preparation-summary.json"
_BASELINE = "reports/rq1-baseline-v2-summary.json"
_GMM = "reports/rq2-gmm-development-v1-summary.json"
_POLICY = {
    "schema_version": 1,
    "contract_id": "development-execution-v1",
    "date": "2026-09-22",
    "development_execution_ready": True,
    "protected_evaluation_ready": False,
    "execution_binding_sha256": BASE_PROFILE_SHA256,
    "methods_sha256": METHODS_SHA256,
    "input_roles": ["train", "validation"],
    "steps": list(STEPS),
    "failure_policy": "fail_stop_no_retry_no_resume",
    "input_reads": "once_per_input_after_root_reservation",
    "training_boundary": "freeze_drift_reference_before_reading_validation",
    "child_reservation": "before_member_computation",
    "child_publication": "immediate_immutable_outputs_before_next_member",
    "partial_run": "retain_completed_children_and_failed_or_incomplete_child_unattempted_tail_is_not_failure",
    "aggregate_acceptance": "observed_zero_worker_exit_all_eight_members_verified_and_binding_rechecked",
    "verification": "saved_outputs_only_no_source_reread_refit_or_url_rescoring",
    "primary_changes": False,
    "additional_transformer_fits": 0,
    "perturbation_runs": 0,
}


class DevelopmentExecutionError(ValueError):
    """The prospective development procedure cannot be authenticated."""


@dataclass(frozen=True)
class DevelopmentExecutionBinding:
    base: ExecutionBinding
    profile_sha256: str
    methods_sha256: str
    pins: DevelopmentPins
    preparation_bytes: bytes

    @property
    def protected_evaluation_ready(self) -> bool:
        return False


def _require(condition, reason):
    if not condition:
        raise DevelopmentExecutionError(reason)


def _json(content):
    return json.loads(
        content,
        object_pairs_hook=fixed_cascade._object_without_duplicate_keys,
        parse_constant=fixed_cascade._reject_json_constant,
    )


def _validate_profile(value):
    _require(
        type(value) is dict and set(value) == set(_POLICY) | {"research_scope"},
        "invalid_profile_policy",
    )
    _require(
        fixed_cascade._matches_exactly({key: value[key] for key in _POLICY}, _POLICY)
        and type(value["research_scope"]) is str
        and bool(value["research_scope"].strip()),
        "invalid_profile_policy",
    )


def _read_profile(base, expected_hash):
    content = execution_preflight._read_regular(base.root, PROFILE_PATH)
    _require(sha256(content).hexdigest() == expected_hash, "profile_hash_mismatch")
    try:
        _validate_profile(_json(content))
    except (UnicodeError, ValueError, TypeError) as exc:
        if isinstance(exc, DevelopmentExecutionError):
            raise
        raise DevelopmentExecutionError("invalid_profile_policy") from None
    # Reuse regular Git-mode/committed-byte checks, including all project source.
    execution_preflight._historical_v2_committed_files(
        base.root,
        base.revision,
        {PROFILE_PATH: expected_hash, METHODS_PATH: METHODS_SHA256},
    )
    methods = execution_preflight._read_regular(base.root, METHODS_PATH)
    _require(sha256(methods).hexdigest() == METHODS_SHA256, "methods_hash_mismatch")
    return _json(methods)


def _public_bytes(base, relative):
    digest = dict(base.source_hashes).get(relative)
    _require(digest is not None, "public_input_not_bound")
    content = execution_preflight._read_regular(base.root, relative)
    _require(sha256(content).hexdigest() == digest, "public_input_hash_mismatch")
    return content


def _development_inputs(base, methods):
    hashes = dict(base.source_hashes)
    for relative, digest in methods["public_file_sha256"].items():
        actual = (
            base.contract_sha256
            if relative == execution_preflight._HISTORICAL_V2_CONTRACT_PATH
            else hashes.get(relative)
        )
        _require(actual == digest, "methods_public_chain_mismatch")
    source_bytes = _public_bytes(base, _SOURCE)
    preparation_bytes = _public_bytes(base, _PREPARATION)
    source = phiusiil._load_source_spec(source_bytes)
    preparation = _json(preparation_bytes)
    validated = baselines._validate_preparation_summary(preparation)
    _require(
        preparation["source_spec_sha256"] == hashes[_SOURCE]
        and fixed_cascade._matches_exactly(preparation["declared_sources"], source),
        "source_preparation_chain_mismatch",
    )
    baseline = _json(_public_bytes(base, _BASELINE))
    gmm = _json(_public_bytes(base, _GMM))
    partitions = methods["inputs"]["partitions"]
    accepted = methods["inputs"]["accepted_model_bytes"]
    baseline_contract = hashes["data/rq1-baseline-contract-v2.json"]
    gmm_contract = hashes["data/rq2-gmm-development-contract-v1.json"]
    baseline_inputs = {
        **partitions,
        "contract": baseline_contract,
        "preparation_summary": hashes[_PREPARATION],
    }
    gmm_inputs = {
        **partitions,
        "baseline_contract": baseline_contract,
        "gmm_contract": gmm_contract,
        "logistic_l1_artifact": accepted["logistic_l1"],
        "preparation_summary": hashes[_PREPARATION],
    }
    _require(
        fixed_cascade._matches_exactly(baseline["input_hashes"], baseline_inputs)
        and fixed_cascade._matches_exactly(gmm["input_hashes"], gmm_inputs)
        and baseline["models"]["Logistic-L1"]["artifact_sha256"]
        == accepted["logistic_l1"]
        and gmm["artifact_hashes"]["gmm.json"] == accepted["gmm"],
        "accepted_model_chain_mismatch",
    )
    for split in ("train", "validation"):
        declared = validated["splits"][split]
        _require(
            validated["output_hashes"][f"{split}.jsonl"] == partitions[split]
            and fixed_cascade._matches_exactly(
                baseline["input_counts"][split],
                {"rows": declared["row_count"], **declared["class_counts"]},
            )
            and fixed_cascade._matches_exactly(
                gmm["input_counts"][split],
                {
                    "rows": declared["row_count"],
                    "domain_count": declared["domain_count"],
                    "class_counts": declared["class_counts"],
                },
            ),
            "development_partition_chain_mismatch",
        )
    return DevelopmentPins(
        train_sha256=partitions["train"],
        validation_sha256=partitions["validation"],
        source_csv_sha256=validated["source_csv_sha256"],
        preparation_summary_sha256=hashes[_PREPARATION],
        suffix_rules_sha256=source["public_suffix_list"]["sha256"],
        logistic_l1_artifact_sha256=accepted["logistic_l1"],
        baseline_contract_sha256=baseline_contract,
        gmm_artifact_sha256=accepted["gmm"],
        gmm_contract_sha256=gmm_contract,
    ), preparation_bytes


def bind_development_execution(
    root: Path, *, expected_revision: str, expected_profile_sha256: str
) -> DevelopmentExecutionBinding:
    """Authenticate the fixed development-only profile before data-path access."""
    execution_preflight._exact_hex(
        expected_profile_sha256, 64, "expected_profile_sha256"
    )
    base = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=BASE_PROFILE_SHA256,
    )
    methods = _read_profile(base, expected_profile_sha256)
    try:
        pins, preparation_bytes = _development_inputs(base, methods)
    except DevelopmentExecutionError:
        raise
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise DevelopmentExecutionError("invalid_public_development_chain") from None
    recheck_binding(base)
    _require(
        sha256(execution_preflight._read_regular(base.root, PROFILE_PATH)).hexdigest()
        == expected_profile_sha256,
        "profile_changed_during_binding",
    )
    _require(
        sha256(execution_preflight._read_regular(base.root, METHODS_PATH)).hexdigest()
        == METHODS_SHA256,
        "methods_changed_during_binding",
    )
    return DevelopmentExecutionBinding(
        base, expected_profile_sha256, METHODS_SHA256, pins, preparation_bytes
    )


def recheck_development_binding(binding: DevelopmentExecutionBinding) -> None:
    """Reject changes to either execution binding or the retained source identity."""
    _require(
        type(binding) is DevelopmentExecutionBinding, "invalid_development_binding"
    )
    current = bind_development_execution(
        binding.base.root,
        expected_revision=binding.base.revision,
        expected_profile_sha256=binding.profile_sha256,
    )
    _require(current == binding, "development_binding_changed")
