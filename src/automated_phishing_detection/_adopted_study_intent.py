"""Authenticate retained root intent against independent reviewed expectations."""

from hashlib import sha256

from . import _adopted_study_records as records
from . import _study_run_public as original
from . import _study_run_schema as schema
from .study_execution import validate_study_execution_bytes

_AUTHORIZATION = ("policy_bytes", "profile_bytes", "envelope_bytes")


def authenticate_intent(content, execution, profile_pin, envelope_pin):
    value = records.unwrap(
        content, execution, {*_AUTHORIZATION, "scientific_intent_bytes"}
    )
    schema.require(value["status"] == "intent")
    schema.require(execution["study_profile_sha256"] == profile_pin)
    schema.require(execution["adoption_envelope_sha256"] == envelope_pin)
    payloads = tuple(records.decoded(value[name]) for name in _AUTHORIZATION)
    profile = validate_study_execution_bytes(
        *payloads,
        expected_profile_sha256=profile_pin,
        expected_envelope_sha256=envelope_pin,
    )
    _profile_execution(profile, execution, payloads[0])
    _scientific_intent(value, execution, payloads[0], profile)
    return profile


def _scientific_intent(value, execution, policy_bytes, profile):
    content = records.decoded(value["scientific_intent_bytes"])
    original._intent(content, records.scientific_execution(execution))
    scientific = schema.load(content)
    schema.same(
        scientific["protective_deadlines_seconds"],
        schema.load(policy_bytes)["protective_deadlines_seconds"],
    )
    schema.same(
        scientific["operational_profile"]["bound_file_sha256"],
        profile["source_artifact_scope"],
    )


def _profile_execution(profile, execution, policy_bytes):
    schema.require(execution["study_policy_sha256"] == sha256(policy_bytes).hexdigest())
    schema.require(profile["execution"]["revision"] == execution["revision"])
    schema.require(
        profile["execution"]["contract_sha256"]
        == execution["execution_contract_sha256"]
    )
    schema.require(
        profile["components"]["operational"] == execution["operational_profile_sha256"]
    )


def validate_source_profile(content, profile):
    source = schema.load(content)["accepted_inputs"]["external"]["execution"]
    schema.require(source["source_profile_sha256"] == profile["components"]["external"])
