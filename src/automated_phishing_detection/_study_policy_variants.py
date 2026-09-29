"""Fixed historical policies and a separately adopted scientific amendment."""

from hashlib import sha256

from . import _study_execution_policy as original
from ._checkpoint_codec import canonical_bytes

OPERATOR_POLICY_PATH = "data/study-execution-policy-v2.json"
GOVERNANCE_MODULES = frozenset(
    f"src/automated_phishing_detection/{name}.py"
    for name in (
        "_study_policy_variants",
        "_study_operator_authority",
        "_study_execution_schema",
        "_study_execution_io",
    )
)
GOVERNANCE = (
    "Require actual method, exact-profile and protected-access authorization from "
    "the named operator's retained directive. That directive explicitly waives "
    "fresh advisor concurrence for a governance-only rebind of the retained "
    "reviewed profile. Record the original scope and the agent's effective-profile "
    "binding without claiming advisor approval or personal review of a later "
    "digest. Preserve all scientific and session requirements."
)


def operator_policy_projection():
    value = original.policy_projection()
    value.update(schema_version=2, policy_id="study-execution-policy-v2")
    value["method"][-1] = GOVERNANCE
    value["decisions"] = ["method", "advisor_requirement_waiver", "profile", "access"]
    value["authorization_basis"] = "operator_directive_governance_only_rebind"
    return value


def operator_policy_bytes():
    return canonical_bytes(operator_policy_projection())


def select_policy(expected_sha256):
    from . import _study_urlnorm_policy as amendment

    for path, content in (
        (original.POLICY_PATH, original.policy_bytes()),
        (OPERATOR_POLICY_PATH, operator_policy_bytes()),
        (amendment.POLICY_PATH, amendment.policy_bytes()),
    ):
        if sha256(content).hexdigest() == expected_sha256:
            return path, content
    raise ValueError("unknown_study_execution_policy")
