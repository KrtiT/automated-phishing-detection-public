"""Authenticate one historical whole-study hold for a separately authorized study."""

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256

from . import _adopted_study_records as adopted
from . import _study_run_schema as schema
from ._adopted_study_verification import verify_saved_adopted_authorization
from ._checkpoint_codec import canonical_bytes
from ._prepared_failure_context import carry_failure_context
from ._retained_study_hold_files import hold_snapshot
from .study_preparation_context import PREPARATION, SOURCE, bound_preparation_context

_FIELDS = {
    "representation",
    "prior_profile",
    "prior_profile_sha256",
    "prior_envelope_sha256",
    "prior_root_reservation_sha256",
    "prior_public_summary_sha256",
    "prior_preparation_reservation_sha256",
    "prior_preparation_complete_sha256",
    "publisher_source_sha256",
    "publisher_summary_sha256",
    "diagnostic_sha256",
}


class RetainedStudyHoldError(ValueError):
    """Symbolic historical-root rejection without private diagnostics."""


@dataclass(frozen=True)
class PriorStudyHold:
    profile_bytes: bytes = field(repr=False)
    barrier_bytes: bytes = field(repr=False)
    identity_bytes: bytes = field(repr=False)
    source_spec_bytes: bytes = field(repr=False)
    preparation_summary_bytes: bytes = field(repr=False)

    @property
    def profile(self):
        return json.loads(self.profile_bytes)

    @property
    def barrier(self):
        return json.loads(self.barrier_bytes)

    @property
    def expected_identity(self):
        return json.loads(self.identity_bytes)


def _continuation(value):
    schema.keys(value, _FIELDS)
    schema.require(value["representation"] == "publisher_url_norm_v1")
    for name in _FIELDS - {"representation", "prior_profile"}:
        schema.operational.digest(value[name])
    content = canonical_bytes(value["prior_profile"])
    schema.require(sha256(content).hexdigest() == value["prior_profile_sha256"])
    return json.loads(content), content


def _authenticate(snapshot, continuation, profile_bytes):
    schema.require(
        sha256(snapshot.payload("public-summary.json")).hexdigest()
        == continuation["prior_public_summary_sha256"]
    )
    execution = verify_saved_adopted_authorization(
        snapshot,
        expected_profile_sha256=continuation["prior_profile_sha256"],
        expected_envelope_sha256=continuation["prior_envelope_sha256"],
    )
    intent = schema.load(snapshot.payload("attempt/study-intent.json"))
    schema.require(adopted.decoded(intent["profile_bytes"]) == profile_bytes)
    public = json.loads(snapshot.payload("public-summary.json"))
    barrier_bytes = snapshot.payload("attempt/prediction-barrier.json")
    barrier = schema.load(barrier_bytes)
    schema.require(public["status"] == barrier["status"] == "whole_study_hold")
    schema.require(barrier["predictions_started"] is False)
    for suffix in ("reservation", "complete"):
        schema.require(
            barrier[f"study_preparation_{suffix}_sha256"]
            == continuation[f"prior_preparation_{suffix}_sha256"]
        )
    return execution, barrier_bytes


def _historical_identity(binding, profile, execution):
    identity, unused, buffers, unused_profile = bound_preparation_context(binding)
    pins = dict(binding.source_hashes)
    for path, name in (
        (SOURCE, "source_spec_sha256"),
        (PREPARATION, "preparation_summary_sha256"),
    ):
        schema.require(
            profile["source_artifact_scope"][path]
            == pins[path]
            == identity[name]
            == sha256(buffers[path]).hexdigest()
        )
    for name in ("execution_contract_sha256", "runtime_sha256", "source_spec_sha256"):
        schema.require(execution[name] == identity[name])
    schema.require(identity["revision"] == binding.revision)
    historical = identity | {
        "revision": profile["execution"]["revision"],
        "source_profile_sha256": profile["components"]["external"],
    }
    return canonical_bytes(historical), buffers


@contextmanager
def hold_prior_study_hold(binding, continuation):
    """Keep authenticated old root identities held through all caller cleanup."""
    original = None
    try:
        profile, profile_bytes = _continuation(continuation)
        with hold_snapshot(
            profile, continuation["prior_root_reservation_sha256"]
        ) as snapshot:
            execution, barrier = _authenticate(snapshot, continuation, profile_bytes)
            identity, buffers = _historical_identity(binding, profile, execution)
            try:
                yield PriorStudyHold(
                    profile_bytes,
                    barrier,
                    identity,
                    buffers[SOURCE],
                    buffers[PREPARATION],
                )
            except BaseException as error:
                original = error
                raise
    except BaseException as error:
        carry_failure_context(error, original)
        if not isinstance(error, Exception) or error is original:
            raise
        rejected = RetainedStudyHoldError("invalid_retained_study_hold")
        carry_failure_context(rejected, error)
        raise rejected from None
