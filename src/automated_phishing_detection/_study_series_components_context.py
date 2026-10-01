"""Exact public base/candidate identities without IO or live preflight rechecks."""

from hashlib import sha256

from . import _external_source_profile as external
from . import _operational_profile as operational
from . import _study_execution_schema as schema
from . import _study_series_adoption_profile as series
from . import _study_series_policy as policy
from . import execution_preflight as preflight
from ._checkpoint_codec import canonical_bytes


def _base(base, profile):
    pins = external._pins(base)
    schema.require(base.source_hashes == tuple(sorted(pins.items())))
    schema.require(pins == profile["source_artifact_scope"])
    root = schema.lexical_path(profile["paths"]["repo_root"])
    schema.require(type(base.root) is type(root) and base.root == root)
    execution = external._execution(base, pins)
    schema.require(
        profile["execution"]
        == {
            "revision": base.revision,
            "contract_sha256": base.contract_sha256,
            "runtime_sha256": sha256(base.runtime_json.encode()).hexdigest(),
        }
    )
    schema.require(
        profile["scientific_pins"]["source_spec_sha256"] == pins[external._SOURCE]
    )
    return execution, pins


def _external(value, execution, pins):
    expected = {
        "schema_version": 1,
        "profile_id": "external-source-candidate-v1",
        "status": "specified_closed_candidate",
        "protected_evaluation_ready": False,
        "protected_evaluation_authorized": False,
        "execution": execution,
        "publisher": external._publisher(),
        "retention": external._retention(),
        "implementation_sha256": {
            name: pins[name] for name in external._IMPLEMENTATIONS
        },
        "public_suffix_list": value["public_suffix_list"],
    }
    schema.require(type(value["public_suffix_list"]) is dict)
    schema.digest(value["public_suffix_list"]["sha256"])
    schema.require(canonical_bytes(value) == canonical_bytes(expected))


def joined(base, source, workload, profile_bytes, expected_profile_sha256):
    schema.require(type(base) is preflight.ExecutionBinding)
    schema.require(type(source) is external.CandidateExternalProfile)
    schema.require(type(workload) is operational.CandidateOperationalProfile)
    profile = schema.parse(profile_bytes, expected_profile_sha256)
    series.profile(profile, policy.policy_projection())
    execution, pins = _base(base, profile)
    first = schema.parse(
        source.canonical_bytes, profile["components"]["current_external"]
    )
    second = schema.parse(
        workload.canonical_bytes, profile["components"]["current_operational"]
    )
    _external(first, execution, pins)
    schema.require(
        canonical_bytes(second) == canonical_bytes(operational._projection(base))
    )
    return profile, first, second
