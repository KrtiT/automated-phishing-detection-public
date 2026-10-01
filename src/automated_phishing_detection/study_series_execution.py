"""Bind public series identity without historical access or execution authority."""

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from . import _external_source_profile as external
from . import _operational_profile as operational
from . import _study_execution_schema as schema
from . import _study_series_execution_io as admission_io
from . import _study_series_policy as policy
from . import execution_preflight as preflight
from ._checkpoint_codec import canonical_bytes
from ._study_execution_policy import CONTRACT_SHA256, DEADLINES
from .study_series_adoption import validate_series_adoption_header


class SeriesPublicBindingError(ValueError):
    """The supplied adoption metadata and public execution identity do not join."""


@dataclass(frozen=True)
class SeriesPublicBinding:
    """Public consistency only; caller construction never grants a capability."""

    base: preflight.ExecutionBinding
    external: external.CandidateExternalProfile
    operational: operational.CandidateOperationalProfile
    components: object
    policy_bytes: bytes = field(repr=False)
    profile_bytes: bytes = field(repr=False)
    envelope_bytes: bytes = field(repr=False)
    envelope_path: Path
    authorizes_execution: bool = field(default=False, init=False)

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


def _header(path, profile_pin, envelope_pin):
    content = admission_io.read_envelope(path)
    value = schema.parse(content, envelope_pin)
    payload = canonical_bytes(value["profile"])
    selected_policy = policy.policy_bytes()
    validate_series_adoption_header(
        selected_policy,
        payload,
        content,
        expected_profile_sha256=profile_pin,
        expected_envelope_sha256=envelope_pin,
    )
    return value["profile"], payload, content, selected_policy


def _base(root, revision, value):
    schema.require(value["execution"]["revision"] == revision)
    schema.require(schema.lexical_path(str(root)) == Path(value["paths"]["repo_root"]))
    base = preflight.bind_execution(
        root, expected_revision=revision, expected_contract_sha256=CONTRACT_SHA256
    )
    schema.require(type(base) is preflight.ExecutionBinding)
    schema.require(base.root == Path(value["paths"]["repo_root"]))
    schema.require(
        base.revision == revision and base.contract_sha256 == CONTRACT_SHA256
    )
    return base


def _components(base, payload, profile_pin):
    from .study_series_components import verify_series_component_transition

    selected_external = external.resolve_external_source_profile(base)
    selected_operational = operational.resolve_operational_profile(base)
    audit = preflight._read_regular(
        base.root, "reports/rq2-gmm-development-v1-summary.json"
    )
    comparison = verify_series_component_transition(
        base,
        selected_external,
        selected_operational,
        payload,
        audit,
        expected_profile_sha256=profile_pin,
    )
    return selected_external, selected_operational, comparison


def _bind(root, revision, path, profile_pin, envelope_pin):
    schema.digest(revision, 40)
    schema.digest(profile_pin)
    schema.digest(envelope_pin)
    path = schema.lexical_path(str(path))
    value, payload, content, expected_policy = _header(path, profile_pin, envelope_pin)
    base = _base(root, revision, value)
    committed = admission_io.committed_policy(base, value["policy_sha256"])
    schema.require(committed == expected_policy)
    selected_external, selected_operational, comparison = _components(
        base, payload, profile_pin
    )
    preflight.recheck_binding(base)
    schema.require(admission_io.read_envelope(path) == content)
    schema.require(
        admission_io.committed_policy(base, value["policy_sha256"]) == committed
    )
    return SeriesPublicBinding(
        base,
        selected_external,
        selected_operational,
        comparison,
        committed,
        payload,
        content,
        path,
    )


def bind_series_public_execution(
    root,
    *,
    expected_revision,
    expected_profile_sha256,
    envelope_path,
    expected_envelope_sha256,
):
    """Authenticate public-only code and metadata before any history path access."""
    try:
        return _bind(
            root,
            expected_revision,
            envelope_path,
            expected_profile_sha256,
            expected_envelope_sha256,
        )
    except Exception:
        raise SeriesPublicBindingError("invalid_series_public_binding") from None


def recheck_series_public_execution(binding):
    """Rerun the public stage only; never repeat historical reconstruction here."""
    try:
        schema.require(type(binding) is SeriesPublicBinding)
        current = bind_series_public_execution(
            binding.base.root,
            expected_revision=binding.base.revision,
            expected_profile_sha256=binding.profile_sha256,
            envelope_path=binding.envelope_path,
            expected_envelope_sha256=binding.envelope_sha256,
        )
        schema.require(current == binding)
    except Exception:
        raise SeriesPublicBindingError("invalid_series_public_binding") from None
