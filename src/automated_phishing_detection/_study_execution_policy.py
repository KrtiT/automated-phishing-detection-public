"""Revision-independent prospective specification, requiring explicit adoption."""

from ._checkpoint_codec import canonical_bytes

POLICY_PATH = "data/study-execution-policy-v1.json"
CONTRACT_SHA256 = "b3847581d73cba2a01d9f25d39548bb6cd96903129b0e382359faa6df97ab7c4"
ROOT_SCRIPT = "scripts/run_adopted_study.py"
CHILD_SCRIPT = "scripts/run_study_child.py"
DEADLINES = {"startup": 300, "shutdown": 180, "terminate": 10, "kill": 10}
METHOD = (
    "Prepare both fixed datasets once in one fresh retained study root before any protected prediction. Hold the entire study on any of the thirteen existing necessary-capacity shortages.",
    "Preserve the original source mappings, quarantine, populations, thresholds, intervals, workloads, hypothesis rules and original H2 audit failure of 28/252. Conditional unavailable secondary outputs remain explicitly unavailable; unrun work is not a measured negative result.",
    "Reuse the original internal then external scientific owners, accepted-input retention, fixed 125-cell schedule and reducers. Do not retry, resume, resample, shorten workloads, reread sources or replace a stopped attempt automatically.",
    "Use identical protective control-plane budgets of 300 seconds startup, 180 seconds graceful shutdown, 10 seconds SIGTERM wait and 10 seconds SIGKILL observation. Preserve the 2000 millisecond request deadline and 60 second controls, with no client-wide workload cutoff or post-adverse-run enlargement.",
    "Use one named operator and one exclusive session on the pinned host/runtime, with AC power, Automatic energy mode, Low Power disabled and sleep disabled. Prohibit concurrent training, benchmarks, tests, builds and voluntary CPU/GPU workloads. Retain pre/post conditions, interference and thermal warnings; any violation preserves actual evidence for review and cannot permit selective deletion or replacement.",
    "Require separate actual method, advisor, exact-profile and protected-access decisions. Method decisions retain their original method or policy scope; final profile and access decisions reference the immutable profile digest. Engineering approval is not scientific adoption or data authorization.",
)
SESSION_REQUIREMENTS = {
    "power": "AC",
    "energy_mode": "Automatic",
    "low_power": "disabled",
    "sleep": "disabled",
    "concurrent_workloads": "none",
    "records": "pre_post_conditions_interference_and_thermal_warnings",
    "violation": "retain_actual_evidence_and_stop_for_review",
}


def policy_projection():
    return {
        "schema_version": 1,
        "policy_id": "study-execution-policy-v1",
        "status": "specified_for_explicit_adoption",
        "base_contract": {"id": "execution-binding-v3", "sha256": CONTRACT_SHA256},
        "component_kinds": ["external-source-candidate-v1", "operational-candidate-v1"],
        "method": list(METHOD),
        "protective_deadlines_seconds": dict(DEADLINES),
        "session_requirements": dict(SESSION_REQUIREMENTS),
        "session_declarations": "prospective_obligations_not_observed_host_conditions",
        "scripts": {"root": ROOT_SCRIPT, "child": CHILD_SCRIPT},
        "admission_roles": ["internal", "external", "service", "client"],
        "decisions": ["method", "advisor", "profile", "access"],
        "trust_boundary": "actual_decision_records_under_trusted_operator_not_cryptographic_consent",
        "legacy_readiness": False,
    }


def policy_bytes():
    return canonical_bytes(policy_projection())
