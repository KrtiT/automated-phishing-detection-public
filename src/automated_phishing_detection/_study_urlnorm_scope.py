"""Closed ancestry and the sole permitted source-representation scope change."""

from hashlib import sha256

from . import _study_execution_schema as schema
from . import _study_policy_variants as variants
from ._checkpoint_codec import canonical_bytes
from ._study_urlnorm_policy import REPRESENTATION, policy_projection

OUTPUTS = (
    "preparation-attempt",
    "internal-attempt",
    "internal-public-summary",
    "external-attempt",
    "external-public-summary",
    "attempt",
    "public-summary",
    "accepted-inputs-dir",
    "cells-dir",
)
PIN_FIELDS = (
    "prior_profile_sha256",
    "prior_envelope_sha256",
    "prior_root_reservation_sha256",
    "prior_public_summary_sha256",
    "prior_preparation_reservation_sha256",
    "prior_preparation_complete_sha256",
    "publisher_source_sha256",
    "publisher_summary_sha256",
    "diagnostic_sha256",
)
FIELDS = ("representation", "prior_profile", *PIN_FIELDS)
IMPLEMENTATIONS = frozenset(
    f"src/automated_phishing_detection/{name}.py"
    for name in (
        "_study_urlnorm_policy",
        "_study_urlnorm_scope",
        "_study_urlnorm_authority",
        "_study_urlnorm_completion",
        "_study_urlnorm_preparation",
        "_study_execution_schema",
        "_study_policy_variants",
        "adopted_study_runner",
        "publisher_urlnorm",
        "saved_phishvn_source",
        "retained_study_hold",
        "_retained_study_hold_files",
        "retained_study_derivation",
        "_retained_study_derivation_context",
        "_retained_preparation_records",
    )
)


def digest(value):
    return sha256(canonical_bytes(value)).hexdigest()


def continuation(value):
    schema.closed(value, FIELDS)
    schema.require(value["representation"] == REPRESENTATION)
    for name in PIN_FIELDS:
        schema.digest(value[name])
    prior = value["prior_profile"]
    schema.require(type(prior) is dict and prior.get("schema_version") == 1)
    schema.profile(prior)
    schema.require(value["prior_profile_sha256"] == digest(prior))
    policy = variants.operator_policy_projection()
    schema.require(prior["policy_sha256"] == digest(policy))
    schema.require(prior["method_sha256"] == digest(policy["method"]))
    return prior


def _paths(prior, current):
    previous = {name: schema.lexical_path(prior["paths"][name]) for name in OUTPUTS}
    for name, path in current["paths"].items():
        if name not in OUTPUTS:
            schema.require(path == prior["paths"][name])
            continue
        selected = schema.lexical_path(path)
        for old in previous.values():
            schema.require(not selected.is_relative_to(old))
            schema.require(not old.is_relative_to(selected))


def _session(prior, current):
    before, after = prior["session"], current["session"]
    schema.require(before["session_id"] != after["session_id"])
    schema.require(
        {name: value for name, value in before.items() if name != "session_id"}
        == {name: value for name, value in after.items() if name != "session_id"}
    )


def _implementations(prior, current):
    previous, selected = (
        prior["source_artifact_scope"],
        current["source_artifact_scope"],
    )
    schema.require(previous.keys() <= selected.keys())
    changed = {name for name in selected if selected[name] != previous.get(name)}
    schema.require(changed <= IMPLEMENTATIONS | {"data/study-execution-policy-v3.json"})


def profile_scope(current):
    policy = policy_projection()
    schema.require(current["policy_sha256"] == digest(policy))
    schema.require(current["method_sha256"] == digest(policy["method"]))
    prior = continuation(current["continuation"])
    _paths(prior, current)
    _session(prior, current)
    _implementations(prior, current)
    schema.require(
        prior["execution"]["contract_sha256"] == current["execution"]["contract_sha256"]
    )
