"""Authenticate study-only scope while every original candidate remains closed.

Decision extracts are trusted operator records, not cryptographic consent.
Session declarations bind prospective obligations, not observed host conditions.
"""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from . import _external_source_profile as external
from . import _operational_profile as operational
from . import _study_execution_io as admission_io
from . import _study_execution_schema as schema
from . import execution_preflight as preflight
from ._checkpoint_codec import canonical_bytes
from ._study_execution_paths import project_paths
from ._study_execution_policy import CONTRACT_SHA256, DEADLINES
from ._study_execution_schema import StudyExecutionError
from ._study_run_context import StudyRunPaths


@dataclass(frozen=True)
class StudyExecutionBinding:
    base: preflight.ExecutionBinding
    external: external.CandidateExternalProfile
    operational: operational.CandidateOperationalProfile
    paths: StudyRunPaths
    policy_bytes: bytes = field(repr=False)
    profile_bytes: bytes = field(repr=False)
    envelope_bytes: bytes = field(repr=False)
    envelope_path: Path

    @property
    def profile_sha256(self):
        return sha256(self.profile_bytes).hexdigest()

    @property
    def envelope_sha256(self):
        return sha256(self.envelope_bytes).hexdigest()

    @property
    def policy_sha256(self):
        return sha256(self.policy_bytes).hexdigest()

    @property
    def deadlines(self):
        return dict(DEADLINES)


def validate_study_execution_bytes(
    policy_bytes,
    profile_bytes,
    envelope_bytes,
    *,
    expected_profile_sha256,
    expected_envelope_sha256,
):
    """Verify only retained byte/scope consistency against independent expectations."""
    try:
        return schema.joined(
            policy_bytes,
            profile_bytes,
            envelope_bytes,
            expected_profile_sha256,
            expected_envelope_sha256,
        )
    except Exception:
        raise StudyExecutionError("invalid_study_execution_authorization") from None


def _components(base, value):
    selected_external = external.resolve_external_source_profile(base)
    selected_operational = operational.resolve_operational_profile(base)
    schema.require(type(selected_external) is external.CandidateExternalProfile)
    schema.require(
        type(selected_operational) is operational.CandidateOperationalProfile
    )
    schema.require(
        value["components"]
        == {
            "external": selected_external.profile_sha256,
            "operational": selected_operational.profile_sha256,
        }
    )
    schema.require(value["source_artifact_scope"] == dict(base.source_hashes))
    return selected_external, selected_operational


def _base(root, revision, value):
    schema.require(value["execution"]["revision"] == revision)
    schema.require(schema.lexical_path(str(root)) == Path(value["paths"]["repo-root"]))
    base = preflight.bind_execution(
        root, expected_revision=revision, expected_contract_sha256=CONTRACT_SHA256
    )
    schema.require(type(base) is preflight.ExecutionBinding)
    schema.require(base.root == Path(value["paths"]["repo-root"]))
    schema.require(
        base.revision == revision and base.contract_sha256 == CONTRACT_SHA256
    )
    return base


def _bind(root, revision, path, pin):
    schema.digest(revision, 40)
    schema.digest(pin)
    content = admission_io.read_envelope(path)
    value = schema.envelope(content, pin)
    base = _base(root, revision, value)
    policy = admission_io.committed_policy(base, value["policy_sha256"])
    payload = canonical_bytes(value)
    schema.joined(policy, payload, content, sha256(payload).hexdigest(), pin)
    selected_external, selected_operational = _components(base, value)
    paths = project_paths(base, value["paths"])
    preflight.recheck_binding(base)
    schema.require(admission_io.read_envelope(path) == content)
    schema.require(
        admission_io.committed_policy(base, value["policy_sha256"]) == policy
    )
    return StudyExecutionBinding(
        base,
        selected_external,
        selected_operational,
        paths,
        policy,
        payload,
        content,
        Path(path),
    )


def bind_study_execution(
    root, *, expected_revision, envelope_path, expected_envelope_sha256
):
    """Authenticate a complete exact-profile envelope before protected path inspection."""
    try:
        return _bind(root, expected_revision, envelope_path, expected_envelope_sha256)
    except Exception:
        raise StudyExecutionError("invalid_study_execution_authorization") from None


def recheck_study_execution(binding):
    """Reauthenticate unchanged candidates, policy, envelope and reviewed scope."""
    schema.require(type(binding) is StudyExecutionBinding)
    current = bind_study_execution(
        binding.base.root,
        expected_revision=binding.base.revision,
        envelope_path=binding.envelope_path,
        expected_envelope_sha256=binding.envelope_sha256,
    )
    schema.require(current == binding)
