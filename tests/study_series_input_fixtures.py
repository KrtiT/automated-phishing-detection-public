"""Invented source objects and series metadata, never observed research evidence."""

import json
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
from external_completion_lineage_fixtures import _overlap
from operational_input_external_fixtures import external_case
from operational_input_fixtures import build, candidates, manifests, source_case
from study_series_adoption_fixtures import digest
from study_series_adoption_profile_fixtures import profile

from automated_phishing_detection import _study_series_policy as policy
from automated_phishing_detection import execution_receipt, operational_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._study_execution_policy import CONTRACT_SHA256
from automated_phishing_detection.operational_cell_inputs import bind_cell_descriptor

__all__ = ["candidates", "manifests", "series_case"]


def api():
    name = "automated_phishing_detection.study_series_inputs"
    assert find_spec(name), "missing pure series computational input adapter"
    return import_module(name)


def original_case(manifests):
    case = source_case(manifests)
    payloads = dict(case.internal.snapshot.payloads)
    execution = case.internal.public_summary["execution"]
    execution["execution_contract_sha256"] = CONTRACT_SHA256
    payloads["public-summary.json"] = execution_receipt._json_bytes(
        {"execution": execution}, "invented"
    )
    payloads["attempt/checkpoints/source-overlap.json"] = _overlap(execution)
    case.internal = replace(
        case.internal,
        snapshot=replace(
            case.internal.snapshot, payloads=tuple(sorted(payloads.items()))
        ),
    )
    case.external = external_case(case.internal, 1001, case.external.worker)
    case.binding = replace(case.binding, contract_sha256=CONTRACT_SHA256)
    return case


def joined_profile(case, metadata):
    value = profile(policy.policy_projection())
    prior = value["origin"]["profile"]
    execution = metadata["execution"]
    prior["execution"]["revision"] = execution["revision"]
    prior["invocation"]["arguments"]["expected-revision"] = execution["revision"]
    prior["components"] = {
        "external": metadata["external"]["execution"]["source_profile_sha256"],
        "operational": metadata["operational_profile_sha256"],
    }
    scope = _source_scope(case, prior)
    value["origin"].update(
        profile_sha256=digest(prior),
        revision=execution["revision"],
        root_reservation_sha256=metadata["root_reservation_sha256"],
        preparation_reservation_sha256="d" * 64,
        preparation_complete_sha256="e" * 64,
    )
    value["execution"]["runtime_sha256"] = execution["runtime_sha256"]
    value["transition"]["unchanged_sha256"] = scope.copy()
    value["source_artifact_scope"] = scope | value["transition"]["added_sha256"]
    value["components"].update(
        original_external=prior["components"]["external"],
        original_operational=prior["components"]["operational"],
    )
    _scientific_pins(value, case, metadata)
    return value


def _source_scope(case, prior):
    scope = prior["source_artifact_scope"] | dict(case.binding.source_hashes)
    prior["source_artifact_scope"] = scope
    prior["continuation"]["prior_profile"]["source_artifact_scope"] = scope.copy()
    prior["continuation"]["prior_profile_sha256"] = digest(
        prior["continuation"]["prior_profile"]
    )
    return scope


def _scientific_pins(value, case, metadata):
    value["scientific_pins"].update(
        source_spec_sha256=metadata["execution"]["source_spec_sha256"],
        primary_metadata_sha256=digest(metadata["primary"]),
        internal_bindings_sha256=digest(
            case.internal.snapshot.payload("attempt/evidence/bindings.json")
        ),
        external_bindings_sha256=digest(
            case.external.snapshot.payload("attempt/evidence/bindings.json")
        ),
    )


@pytest.fixture(scope="module")
def series_case(manifests):
    case = original_case(manifests)
    origin_bytes = build(operational_inputs, case).metadata_bytes
    selected = joined_profile(case, json.loads(origin_bytes))
    return SimpleNamespace(
        original=case,
        origin_bytes=origin_bytes,
        profile=selected,
        internal=case.internal.snapshot,
        external=case.external.snapshot,
    )


def metadata(case, **changes):
    arguments = dict(
        expected_origin_sha256=digest(case.origin_bytes),
        expected_profile_sha256=digest(case.profile),
        series_reservation_sha256="1" * 64,
        segment_reservation_sha256="2" * 64,
    )
    return api().build_series_input_metadata(
        case.origin_bytes,
        canonical_bytes(case.profile),
        case.internal,
        case.external,
        **(arguments | changes),
    )


def descriptor(case, ordinal=73, *, content=None, profile_value=None):
    content = metadata(case) if content is None else content
    selected = case.profile if profile_value is None else profile_value
    return api().build_series_cell_descriptor(
        content,
        canonical_bytes(selected),
        case.internal,
        case.external,
        ordinal,
        expected_metadata_sha256=digest(content),
        expected_profile_sha256=digest(selected),
    )


def restore(case, ordinal=73, **changes):
    content = metadata(case)
    payloads = descriptor(case, ordinal, content=content)
    binding = bind_cell_descriptor(
        payloads.descriptor_bytes, cell_reservation_sha256="3" * 64
    )
    arguments = dict(
        metadata_bytes=content,
        profile_bytes=canonical_bytes(case.profile),
        internal=case.internal,
        external=case.external,
        descriptor_bytes=payloads.descriptor_bytes,
        binding_bytes=binding,
        manifest_bytes=payloads.manifest_bytes,
        expected_metadata_sha256=digest(content),
        expected_profile_sha256=digest(case.profile),
        expected_binding_sha256=digest(binding),
        expected_cell_reservation_sha256="3" * 64,
    )
    return api().restore_series_cell_inputs(**(arguments | changes))
