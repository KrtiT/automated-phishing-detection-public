"""Fresh preparation from authenticated retained buffers, never original inputs."""

from hashlib import sha256

from . import _study_preparation_body as body
from . import source_checkpoints, study_preparation_inputs
from . import study_preparation_runner as original
from ._checkpoint_codec import canonical_bytes
from ._exception_cleanup import CleanupStack
from ._retained_study_derivation_context import preflight
from ._study_preparation_records import PreparationState, PreparedStudySnapshot
from ._study_urlnorm_completion import derivation
from .publisher_urlnorm import derive_publisher_url_norm
from .study_preparation_retention import retain_study_preparation


def _internal(state, prior):
    state.stage = "suffix_rules"
    suffix = prior.payload("suffix-rules.dat")
    body.retain(state, "suffix-rules.dat", suffix)
    state.stage = "internal_retention"
    outputs = source_checkpoints._contents(
        state.attempt, state.identity, prior.reconstructed_internal
    )
    for name in (
        "group_test.jsonl",
        "source-overlap.json",
        "source-reconstruction.json",
    ):
        body.retain(state, name, outputs[name])
    return suffix


def _external(state, prior, suffix):
    state.stage = "external_decode"
    decoded = derive_publisher_url_norm(prior.publisher)
    state.stage = "publisher_retention"
    body.retain(
        state, "publisher-source.json", decoded.private_outputs["publisher-source.json"]
    )
    body.retain(
        state, "publisher-summary.json", canonical_bytes(decoded.public_summary)
    )
    state.stage = "external_preparation"
    prepared = study_preparation_inputs.prepare_external_inputs(
        decoded, suffix, overlap_domains=prior.overlap_domains
    )
    state.stage = "external_retention"
    for name in ("retained-test.jsonl", "quarantine.jsonl", "inventory.json"):
        body.retain(state, name, prepared.private_outputs[name])
    body.retain(
        state, "preparation-summary.json", canonical_bytes(prepared.public_summary)
    )
    return prepared


def _complete(state, continuation):
    state.stage = "preparation_completion"
    content = canonical_bytes(
        {
            "schema_version": 2,
            "protocol": "study-preparation-derived-v1",
            "status": "preparation_only",
            "protected_evaluation_authorized": False,
            "scoring_authorized": False,
            "execution": state.identity,
            "reservation_sha256": state.attempt.reservation_sha256,
            "input_sha256": {
                name: sha256(value).hexdigest() for name, value in state.outputs.items()
            },
            "derivation": derivation(continuation),
        }
    )
    body.retain(state, "preparation-complete.json", content)
    return state.writer.finish()


def _run(state, prior, continuation):
    original._recheck(state.binding)
    preflight(state, prior, continuation)
    with CleanupStack() as cleanup:
        with body.deferred_io():
            state.writer = cleanup.enter_context(
                retain_study_preparation(state.attempt, identity=state.identity)
            )
        suffix = _internal(state, prior)
        external = _external(state, prior, suffix)
        body.assess(state, prior.internal, external)
        state.stage = "final_binding"
        original._recheck(state.binding)
        payloads = _complete(state, continuation)
        state.stage = "retention_finalization"
    return PreparedStudySnapshot(state.attempt.reservation_sha256, payloads)


def run_retained_study_preparation(binding, paths, *, prior_preparation, continuation):
    """Derive once under the adopted parent; this helper grants no standalone access."""
    state = PreparationState(binding, paths)
    try:
        return _run(state, prior_preparation, continuation)
    except BaseException as error:
        raise original._failure(state, error) from None
