"""Authenticate the seed/probe run using reviewed public records only.

This binding checks source/runtime identity and derives accepted artifact pins.
It neither opens research inputs nor extends the earlier correction run. The
supervisor must reserve and verify each worker separately; protected access
remains outside this development-only profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from . import bound_models, development_execution, execution_preflight, fixed_cascade
from .execution_preflight import ExecutionBinding
from .execution_preflight import _bind_historical_v2_execution as bind_execution
from .execution_preflight import _recheck_historical_v2_binding as recheck_binding
from .secondary_development import DevelopmentPins
from .source_runner import _json

PROFILE_PATH = "data/seed-probe-execution-contract-v2.json"
PROFILE_SHA256 = "4da034b1a46baa599ae04226ee2f4d9a26c2b2d639cac73fa576ff9cb7aa8839"
BASE_EXECUTION_PROFILE_PATH = "data/seed-probe-execution-contract-v1.json"
BASE_EXECUTION_PROFILE_SHA256 = (
    "cf18fa8c35039c63f896cc62c7aaac8b0847a1abf55676ba67ee65b42340381d"
)
STOPPED_ATTEMPT_PATH = "reports/secondary-seed-probe-v1-attempt-1.json"
STOPPED_ATTEMPT_SHA256 = (
    "3be65bd38c32b8bf8aafa06eede3577a0d1acc212f052d2d9f60768183f535c6"
)
METHODS_PATH = "data/secondary-seed-probe-contract-v1.json"
METHODS_SHA256 = "eb279404728e498999fc7fd0c7578291373bb80b9816f88b5d7202dfdf637380"
ACCEPTED_PATH = "reports/secondary-development-correction-v2-summary.json"
ACCEPTED_SHA256 = "663f1117cd33f949b70c35c56810764193f69cae3a37505004d0db27641d829d"
BASE_PROFILE_SHA256 = development_execution.BASE_PROFILE_SHA256
STAGES = (
    "seed_42_calibration",
    "seed_43",
    "seed_44",
    "seed_45",
    "seed_46",
    "probes",
)
# The canonical policy hash avoids duplicating the contract's prose in code.
_POLICY_SHA256 = "a64b630375d7aeae5601663c255efb2185e30d35fc9c85dddbf70240794cc43a"
_SUPPLEMENT_HASHES = {
    PROFILE_PATH: PROFILE_SHA256,
    BASE_EXECUTION_PROFILE_PATH: BASE_EXECUTION_PROFILE_SHA256,
    STOPPED_ATTEMPT_PATH: STOPPED_ATTEMPT_SHA256,
    METHODS_PATH: METHODS_SHA256,
    development_execution.METHODS_PATH: development_execution.METHODS_SHA256,
    ACCEPTED_PATH: ACCEPTED_SHA256,
}


class SeedProbeExecutionError(ValueError):
    """A safe symbolic failure of the prospective execution binding."""


@dataclass(frozen=True)
class SeedProbeExecutionBinding:
    base: ExecutionBinding
    profile_sha256: str
    base_profile_sha256: str
    stopped_attempt_sha256: str
    methods_sha256: str
    accepted_development_sha256: str
    pins: DevelopmentPins
    preparation_bytes: bytes
    transformer_summary_bytes: bytes
    primary_artifact_hashes: tuple[tuple[str, str], ...]
    public_operating_points_json: str
    retained_drift_summary_json: str
    training_reference_sha256: str
    validation_audit_sha256: str

    @property
    def protected_evaluation_ready(self) -> bool:
        return False


def _require(condition, symbol):
    if not condition:
        raise SeedProbeExecutionError(symbol)


def _canonical(value):
    return execution_preflight._canonical_json(value)


def _digest(value):
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def validate_profile(value) -> None:
    """Freeze the complete policy, including its limits and retention rules."""
    try:
        valid = type(value) is dict and _digest(value) == _POLICY_SHA256
    except (ValueError, TypeError, UnicodeError):
        valid = False
    _require(valid, "invalid_profile_policy")


def _read_supplement(base, expected_hash):
    _require(expected_hash == PROFILE_SHA256, "profile_hash_mismatch")
    execution_preflight._historical_v2_committed_files(
        base.root, base.revision, _SUPPLEMENT_HASHES
    )
    contents = {}
    for relative, digest in _SUPPLEMENT_HASHES.items():
        content = execution_preflight._read_regular(base.root, relative)
        _require(sha256(content).hexdigest() == digest, "supplement_hash_mismatch")
        contents[relative] = _json(content)
    validate_profile(contents[PROFILE_PATH])
    return (
        contents[METHODS_PATH],
        contents[development_execution.METHODS_PATH],
        contents[ACCEPTED_PATH],
    )


def _validate_methods_chain(base, methods):
    hashes = dict(base.source_hashes) | _SUPPLEMENT_HASHES
    hashes[execution_preflight._HISTORICAL_V2_CONTRACT_PATH] = base.contract_sha256
    _require(
        all(
            hashes.get(relative) == digest
            for relative, digest in methods["public_file_sha256"].items()
        ),
        "methods_public_chain_mismatch",
    )


def _primary_inputs(base, methods):
    contents = {
        role: development_execution._public_bytes(base, relative)
        for role, (relative, _) in bound_models._PUBLIC_SUMMARIES.items()
    }
    summaries = {role: _json(content) for role, content in contents.items()}
    bound_models._validate_public(summaries)
    baseline, transformer, gmm = (
        summaries[role] for role in ("baseline", "transformer", "gmm")
    )
    artifacts = {
        "length-only.json": baseline["models"]["length-only"]["artifact_sha256"],
        "logistic-l1.json": baseline["models"]["Logistic-L1"]["artifact_sha256"],
        "gmm.json": gmm["artifact_hashes"]["gmm.json"],
        **transformer["artifact_hashes"],
    }
    _require(
        methods["seeds"]["primary_weight_sha256"]
        == artifacts["transformer-weights.npz"]
        and methods["seeds"]["vocabulary_sha256"] == artifacts["vocabulary.json"],
        "primary_seed_method_chain_mismatch",
    )
    # The other two operating points are omitted from the public summary. The
    # worker must retain their authenticated artifacts, not invent public values.
    points = {
        "length_threshold": baseline["models"]["length-only"]["validation_threshold"][
            "threshold"
        ],
        "stage1_threshold": baseline["models"]["Logistic-L1"]["validation_threshold"][
            "threshold"
        ],
        "monitor_boundary": gmm["threshold"],
    }
    return tuple(sorted(artifacts.items())), _canonical(points), contents["transformer"]


def _retained_drift(report, pins):
    """Select only the drift member from the accepted, hash-bound public report."""
    completion = report["completion"]
    observation = report["execution_observation"]
    _require(
        report["status"] == "accepted_development_evidence"
        and completion["status"] == "completed_secondary_development_correction"
        and report["completion_summary_sha256"] == _digest(completion)
        and type(observation["parent_exit_code"]) is int
        and observation["parent_exit_code"] == 0,
        "accepted_development_status_mismatch",
    )
    zero_exits = {"random_forest": 0, "retained_audit": 0}
    _require(
        fixed_cascade._matches_exactly(completion["worker_exit_codes"], zero_exits)
        and fixed_cascade._matches_exactly(observation["worker_exit_codes"], zero_exits)
        and completion["protected_evaluation_authorized"] is False
        and completion["original_aggregate_accepted"] is False
        and completion["analysis_stage"] == "development_validation_only"
        and fixed_cascade._matches_exactly(
            completion["execution"]["pins"], asdict(pins)
        ),
        "accepted_development_scope_mismatch",
    )
    retained = completion["retained_audit"]
    audit = retained["result"]
    _require(
        retained["stage"] == "retained_audit"
        and retained["status"] == "development_correction_stage_completed"
        and audit["status"] == "retained_development_members_audited"
        and audit["protected_evaluation_authorized"] is False
        and audit["original_aggregate_accepted"] is False
        and audit["analysis_stage"] == "development_validation_only"
        and type(audit["fits"]) is int
        and audit["fits"] == 0
        and fixed_cascade._matches_exactly(audit["input_hashes"], asdict(pins)),
        "retained_audit_scope_mismatch",
    )
    members = audit["members"]
    _require(
        [member["member"] for member in members]
        == list(development_execution.STEPS[:-1]),
        "retained_member_order_mismatch",
    )
    member = members[0]
    summary = member["summary"]
    drift = summary["result"]
    _require(
        member["public_summary_sha256"] == _digest(summary)
        and summary["member"] == "drift"
        and summary["status"] == "development_member_completed"
        and fixed_cascade._matches_exactly(
            member["checks"],
            {
                "authenticated_training_and_validation_membership": True,
                "independent_drift_score_recomputation": False,
                "retained_drift_arithmetic": True,
            },
        ),
        "retained_drift_acceptance_mismatch",
    )
    private_hashes = summary["private_sha256"]
    _require(
        type(private_hashes) is dict
        and set(private_hashes) == {"training-reference.json", "validation-audit.json"}
        and fixed_cascade._matches_exactly(drift["private_sha256"], private_hashes)
        and fixed_cascade._matches_exactly(drift["input_hashes"], asdict(pins))
        and drift["contract_id"] == "secondary-development-v1"
        and drift["analysis_stage"] == "development_validation_only"
        and drift["analysis_role"] == "secondary_descriptive_only"
        and drift["protected_evaluation_authorized"] is False,
        "retained_drift_chain_mismatch",
    )
    for name, digest in private_hashes.items():
        execution_preflight._exact_hex(digest, 64, name)
    return (
        _canonical(drift),
        private_hashes["training-reference.json"],
        private_hashes["validation-audit.json"],
    )


def bind_seed_probe_execution(
    root: Path, *, expected_revision: str, expected_profile_sha256: str
) -> SeedProbeExecutionBinding:
    """Metadata-only binding; accepts no private input or output paths."""
    execution_preflight._exact_hex(
        expected_profile_sha256, 64, "expected_profile_sha256"
    )
    _require(expected_profile_sha256 == PROFILE_SHA256, "profile_hash_mismatch")
    base = bind_execution(
        root,
        expected_revision=expected_revision,
        expected_contract_sha256=BASE_PROFILE_SHA256,
    )
    try:
        methods, development_methods, report = _read_supplement(
            base, expected_profile_sha256
        )
        _validate_methods_chain(base, methods)
        # Reuse source-chain validation, not the old run's authorization profile.
        pins, preparation = development_execution._development_inputs(
            base, development_methods
        )
        artifacts, points, transformer_summary = _primary_inputs(base, methods)
        drift, reference_hash, audit_hash = _retained_drift(report, pins)
    except (SeedProbeExecutionError, execution_preflight.ExecutionPreflightError):
        raise
    except (ValueError, TypeError, KeyError, UnicodeError):
        raise SeedProbeExecutionError("invalid_public_seed_probe_chain") from None
    recheck_binding(base)
    for relative, digest in _SUPPLEMENT_HASHES.items():
        content = execution_preflight._read_regular(base.root, relative)
        _require(sha256(content).hexdigest() == digest, "seed_probe_binding_changed")
    return SeedProbeExecutionBinding(
        base,
        expected_profile_sha256,
        BASE_EXECUTION_PROFILE_SHA256,
        STOPPED_ATTEMPT_SHA256,
        METHODS_SHA256,
        ACCEPTED_SHA256,
        pins,
        preparation,
        transformer_summary,
        artifacts,
        points,
        drift,
        reference_hash,
        audit_hash,
    )


def recheck_seed_probe_binding(binding: SeedProbeExecutionBinding) -> None:
    _require(type(binding) is SeedProbeExecutionBinding, "invalid_seed_probe_binding")
    current = bind_seed_probe_execution(
        binding.base.root,
        expected_revision=binding.base.revision,
        expected_profile_sha256=binding.profile_sha256,
    )
    _require(current == binding, "seed_probe_binding_changed")
