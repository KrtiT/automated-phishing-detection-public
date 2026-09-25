"""Closed immutable file inventories and known context reject before science."""

import builtins
from dataclasses import replace
from pathlib import Path

import pytest
import test_external_completion_records as fixtures

from automated_phishing_detection import _external_source_records as records
from automated_phishing_detection import execution_preflight, saved_external_evidence
from automated_phishing_detection._external_completion_files import ExternalFileSnapshot

inputs = fixtures.inputs
published = fixtures.published
runner = fixtures.runner
verifier = fixtures.verifier
observed_worker = fixtures.observed_worker
completion = fixtures.completion
handoff_api = fixtures.handoff_api
records_case = fixtures.records_case
completion_case = fixtures.completion_case


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "duplicate",
        "list",
        "entry_list",
        "nonbytes",
        "nonstring",
        "short_entry",
    ],
)
def test_retained_snapshot_requires_exact_immutable_inventory(completion_case, change):
    payloads = fixtures.snapshot(completion_case).payloads
    mutations = {
        "missing": payloads[:-1],
        "extra": (*payloads, ("unknown", b"extra")),
        "duplicate": (*payloads, payloads[0]),
        "list": list(payloads),
        "entry_list": (list(payloads[0]), *payloads[1:]),
        "nonbytes": ((payloads[0][0], bytearray(payloads[0][1])), *payloads[1:]),
        "nonstring": ((1, payloads[0][1]), *payloads[1:]),
        "short_entry": ((payloads[0][0],), *payloads[1:]),
    }
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(
            completion_case, files=ExternalFileSnapshot(mutations[change])
        )


@pytest.mark.parametrize("files", [None, {}, (), b"private-canary"])
def test_untyped_snapshot_is_not_metadata_evidence(completion_case, files):
    case = completion_case
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.api().authenticate_external_records(
            files,
            case.attempt,
            binding=case.records.binding,
            profile=case.records.profile,
            handoff=case.records.handoff,
        )


@pytest.mark.parametrize("directory", ["checkpoints", "evidence"])
@pytest.mark.parametrize(
    "name",
    ["publisher-source.json", "internal-source-handoff.json", "all-scores.jsonl"],
)
def test_all_checkpoint_and_evidence_copies_must_agree(
    completion_case, directory, name
):
    changed = fixtures.snapshot(
        completion_case, **{f"attempt/{directory}/{name}": b"changed"}
    )
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(completion_case, files=changed)


@pytest.mark.parametrize("field", ["binding", "profile", "handoff"])
def test_independent_context_types_are_required(completion_case, field):
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(completion_case, **{field: None})


def test_independent_context_is_not_replaced_by_saved_claims(completion_case):
    binding = replace(completion_case.records.binding, revision="a" * 40)
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(completion_case, binding=binding)


@pytest.mark.parametrize(
    "attempt",
    [
        Path("/invented/wrong"),
        Path("/invented/../attempt"),
        "/invented/external-attempt",
    ],
)
def test_expected_attempt_is_lexically_pinned(completion_case, attempt):
    case = completion_case
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.api().authenticate_external_records(
            fixtures.snapshot(case),
            attempt,
            binding=case.records.binding,
            profile=case.records.profile,
            handoff=case.records.handoff,
        )


def test_bad_reservation_stops_before_outer_public_reconstruction(
    completion_case, monkeypatch
):
    changed = fixtures.snapshot(
        completion_case, **{"attempt/reservation.json": b"private-canary"}
    )
    monkeypatch.setattr(
        records,
        "build_external_public",
        lambda *args, **kwargs: pytest.fail(
            "public work before reservation authentication"
        ),
    )
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError) as rejected:
        fixtures.authenticate(completion_case, files=changed)
    assert "private-canary" not in str(rejected.value)


def test_metadata_authentication_never_reads_or_recomputes_science(
    completion_case, monkeypatch
):
    module = fixtures.api()

    def forbidden(*args, **kwargs):
        pytest.fail("retained metadata authentication attempted external work")

    for owner, name in (
        (builtins, "open"),
        (Path, "open"),
        (Path, "read_bytes"),
        (Path, "stat"),
        (execution_preflight, "recheck_binding"),
        (saved_external_evidence, "reconstruct_external_evidence"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    result = module.authenticate_external_records(
        fixtures.snapshot(completion_case),
        completion_case.attempt,
        binding=completion_case.records.binding,
        profile=completion_case.records.profile,
        handoff=completion_case.records.handoff,
    )
    assert result.public["protected_evaluation_authorized"] is False
