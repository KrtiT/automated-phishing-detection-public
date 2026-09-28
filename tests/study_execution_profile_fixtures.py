"""Invented profile metadata without real approvals or protected paths."""

import json
from hashlib import sha256

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_cli_protocol import PATH_ARGUMENTS


def _mapping(case):
    mapping = {name: str(case.root.parent / "inputs" / name) for name in PATH_ARGUMENTS}
    for name in (
        "preparation-attempt",
        "internal-attempt",
        "internal-public-summary",
        "external-attempt",
        "external-public-summary",
        "attempt",
        "public-summary",
        "accepted-inputs-dir",
        "cells-dir",
    ):
        mapping[name] = str(case.root.parent / "outputs" / name)
    mapping["repo-root"] = str(case.root)
    return mapping


def _session(case):
    return {
        "operator": "Invented fixture operator",
        "session_id": "fixture-1",
        "requirements": json.loads(case.policy)["session_requirements"],
        "operator_commitment": "exclusive_session_conditions_and_pre_post_records_required",
    }


def profile(case):
    policy = json.loads(case.policy)
    return {
        "schema_version": 1,
        "profile_id": "study-execution-profile-v1",
        "execution": {
            "revision": case.base.revision,
            "contract_sha256": case.base.contract_sha256,
        },
        "policy_sha256": sha256(case.policy).hexdigest(),
        "method_sha256": sha256(canonical_bytes(policy["method"])).hexdigest(),
        "components": {
            "external": case.external.profile_sha256,
            "operational": case.operational.profile_sha256,
        },
        "invocation": {
            "script": "scripts/run_adopted_study.py",
            "arguments": {
                "repo-root": str(case.root),
                "expected-revision": case.base.revision,
            },
        },
        "paths": _mapping(case),
        "source_artifact_scope": dict(case.base.source_hashes),
        "session": _session(case),
    }
