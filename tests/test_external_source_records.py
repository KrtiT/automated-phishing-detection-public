"""Byte-only outer records from invented verified internal completions."""

import importlib
import importlib.util
import json
from dataclasses import FrozenInstanceError, fields
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_internal_external_handoff as internal_fixtures
from phishvn_source_fixtures import decode, members
from test_external_source_records_fixtures import (
    composition_summary,
    profile_projection,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection._external_checkpoint_protocol import PROTOCOL
from automated_phishing_detection._external_provenance_payloads import PROVENANCE_NAMES
from automated_phishing_detection._external_source_profile import (
    CandidateExternalProfile,
)
from automated_phishing_detection._saved_external_bindings import PRIVATE_OUTPUTS
from automated_phishing_detection.bound_drift import DriftArtifactPaths
from automated_phishing_detection.bound_models import ArtifactPaths
from automated_phishing_detection.bound_secondary import SecondaryArtifactPaths
from automated_phishing_detection.execution_receipt import _json_bytes

inputs = internal_fixtures.inputs
published = internal_fixtures.published
runner = internal_fixtures.runner
verifier = internal_fixtures.verifier
observed_worker = internal_fixtures.observed_worker
completion = internal_fixtures.completion
handoff_api = internal_fixtures.handoff_api
_PUBLIC_FIELDS = {
    "schema_version",
    "status",
    "source_binding",
    "protected_evaluation_authorized",
    "execution",
    "source_profile",
    "publisher",
    "composition",
    "checkpoint_sha256",
    "private_sha256",
}


def api():
    name = "automated_phishing_detection._external_source_records"
    assert importlib.util.find_spec(name), "missing byte-only external records"
    return importlib.import_module(name)


def digest(content):
    return sha256(content).hexdigest()


@pytest.fixture
def records_case(completion, published, handoff_api):
    handoff = handoff_api.build_internal_handoff(completion)
    internal = json.loads(handoff.handoff_bytes)["execution"]
    binding = published[0]
    publisher = decode(members())
    projection = profile_projection(internal, publisher)
    outputs = {name: name.encode() for name in PROVENANCE_NAMES | PRIVATE_OUTPUTS}
    outputs.update(publisher.private_outputs)
    outputs["publisher-summary.json"] = canonical_bytes(publisher.public_summary)
    outputs["internal-source-handoff.json"] = handoff.handoff_bytes
    outputs["internal-source-overlap.json"] = handoff.overlap_bytes
    composition = composition_summary(outputs)
    return SimpleNamespace(
        binding=binding,
        handoff=handoff,
        projection=projection,
        profile=CandidateExternalProfile(canonical_bytes(projection)),
        outputs=outputs,
        composition=composition,
        reservation="f" * 64,
        internal=internal,
    )


def build(case, **changes):
    identity = api().external_identity(case.binding, case.profile, case.handoff)
    arguments = dict(
        binding=case.binding,
        profile=case.profile,
        identity=identity,
        reservation_sha256=case.reservation,
        private_outputs=case.outputs,
        composition=case.composition,
    )
    return api().build_external_public(**(arguments | changes))


def test_external_paths_are_frozen_and_have_exact_path_roles():
    record_type = api().ExternalRunPaths
    names = (
        "archive",
        "suffix_rules",
        "artifacts",
        "secondary_artifacts",
        "drift_artifacts",
        "attempt",
        "public_summary",
    )
    assert tuple(field.name for field in fields(record_type)) == names
    artifacts = ArtifactPaths(*(Path("invented") for unused in fields(ArtifactPaths)))
    secondary = SecondaryArtifactPaths(
        *(Path("invented") for unused in fields(SecondaryArtifactPaths))
    )
    drift = DriftArtifactPaths(Path("training"), Path("validation"))
    paths = record_type(
        Path("archive"),
        Path("suffix"),
        artifacts,
        secondary,
        drift,
        Path("attempt"),
        Path("public"),
    )
    assert paths.artifacts is artifacts and paths.drift_artifacts is drift
    with pytest.raises(FrozenInstanceError):
        paths.archive = Path("changed")


def test_identity_binds_exact_source_profile_and_internal_parent_links(records_case):
    case = records_case
    expected = {
        "kind": "external_evaluation",
        "source_interface": "publisher_archive_reconstruction_v1",
        "checkpoint_protocol": PROTOCOL,
        "source_profile_sha256": case.profile.profile_sha256,
        **case.projection["execution"],
        **case.projection["publisher"]["expected_format"],
        "suffix_rules_sha256": case.profile.suffix_rules_sha256,
        "internal_handoff_sha256": digest(case.handoff.handoff_bytes),
        "internal_overlap_sha256": digest(case.handoff.overlap_bytes),
        "internal_reservation_sha256": case.internal["reservation_sha256"],
    }
    assert api().external_identity(case.binding, case.profile, case.handoff) == expected


def test_public_envelope_keeps_exact_inner_summary_and_matching_hash_maps(records_case):
    public = build(records_case)
    hashes = {name: digest(content) for name, content in records_case.outputs.items()}
    assert set(public) == _PUBLIC_FIELDS
    assert (
        public["schema_version"] == 1
        and public["status"] == "external_evidence_published"
    )
    assert (
        public["source_binding"]
        == "authenticated_publisher_with_parent_declared_internal_handoff"
    )
    assert public["protected_evaluation_authorized"] is False
    assert public["execution"]["reservation_sha256"] == records_case.reservation
    assert public["source_profile"] == records_case.projection
    assert public["publisher"] == json.loads(
        records_case.outputs["publisher-summary.json"]
    )
    assert public["composition"] == records_case.composition
    assert public["composition"] is not records_case.composition
    assert public["checkpoint_sha256"] == public["private_sha256"] == hashes
    assert len(hashes) == 36 and json.loads(_json_bytes(public, "fixture")) == public


def test_public_projections_cannot_mutate_input_records(records_case):
    public = build(records_case)
    public["composition"]["invented_nested"]["unchanged"].clear()
    public["source_profile"]["execution"]["revision"] = "changed"
    public["checkpoint_sha256"].clear()
    assert records_case.composition["invented_nested"]["unchanged"] == [1, None, False]
    assert records_case.profile.projection() == records_case.projection
    assert len(public["private_sha256"]) == 36
