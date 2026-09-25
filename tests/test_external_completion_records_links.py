"""Consistently rehashed outer forgeries still fail expected metadata checks."""

import json

import pytest
import test_external_completion_records as fixtures

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
_META = (
    "attempt/reservation.json",
    "attempt/finalize.claim",
    "attempt/outcome.json",
    "public-summary.json",
)


@pytest.mark.parametrize("name", _META)
@pytest.mark.parametrize(
    "change",
    ["duplicate", "extra", "missing", "bool_version", "float_version", "noncanonical"],
)
def test_every_outer_record_has_exact_canonical_layout(completion_case, name, change):
    contents = completion_case.contents.copy()
    value = json.loads(contents[name])
    if change == "extra":
        value["private-canary"] = True
    elif change == "missing":
        value.pop("schema_version")
    elif change == "bool_version":
        value["schema_version"] = True
    elif change == "float_version":
        value["schema_version"] = 1.0
    content = fixtures.encoded(value)
    if change == "duplicate":
        content = content.replace(
            b'"schema_version":1', b'"schema_version":1,"schema_version":1'
        )
    elif change == "noncanonical":
        content += b"\n"
    contents[name] = content
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(
            completion_case, files=ExternalFileSnapshot(tuple(contents.items()))
        )


@pytest.mark.parametrize(
    "field,value",
    [("directory", "/invented/other"), ("status", "failed"), ("identity", {})],
)
def test_rehashed_reservation_cannot_change_independent_expectations(
    completion_case, field, value
):
    contents = completion_case.contents.copy()
    reservation = json.loads(contents["attempt/reservation.json"])
    reservation[field] = value
    contents["attempt/reservation.json"] = fixtures.encoded(reservation)
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(
            completion_case, files=fixtures.relink(completion_case, contents)
        )


@pytest.mark.parametrize(
    "field,value", [("operation", "failure"), ("reservation_sha256", "a" * 64)]
)
def test_claim_requires_exact_completion_operation_and_reservation(
    completion_case, field, value
):
    claim = json.loads(completion_case.contents["attempt/finalize.claim"])
    claim[field] = value
    files = fixtures.snapshot(
        completion_case, **{"attempt/finalize.claim": fixtures.encoded(claim)}
    )
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(completion_case, files=files)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "failure"),
        ("reservation_sha256", "a" * 64),
        ("public_summary_sha256", "a" * 64),
        ("private_sha256", {}),
    ],
)
def test_outcome_requires_exact_prepared_public_and_private_links(
    completion_case, field, value
):
    outcome = json.loads(completion_case.contents["attempt/outcome.json"])
    outcome[field] = value
    files = fixtures.snapshot(
        completion_case, **{"attempt/outcome.json": fixtures.encoded(outcome)}
    )
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(completion_case, files=files)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "accepted"),
        ("protected_evaluation_authorized", True),
        ("source_binding", "authorized"),
        ("source_profile", {}),
        ("publisher", {}),
    ],
)
def test_rehashed_public_fields_must_equal_rebuilt_outer_projection(
    completion_case, field, value
):
    contents = completion_case.contents.copy()
    public = json.loads(contents["public-summary.json"])
    public[field] = value
    contents["public-summary.json"] = fixtures.encoded(public)
    with pytest.raises(fixtures.api().ExternalCompletionVerificationError):
        fixtures.authenticate(
            completion_case, files=fixtures.relink(completion_case, contents)
        )


def test_coherently_changed_science_is_left_to_independent_scientific_verifier(
    completion_case,
):
    contents = completion_case.contents.copy()
    for directory in ("checkpoints", "evidence"):
        contents[f"attempt/{directory}/all-scores.jsonl"] = b"not scientific evidence"
    public = json.loads(contents["public-summary.json"])
    public["composition"]["private_sha256"]["all-scores.jsonl"] = fixtures.digest(
        b"not scientific evidence"
    )
    contents["public-summary.json"] = fixtures.encoded(public)
    result = fixtures.authenticate(
        completion_case, files=fixtures.relink(completion_case, contents)
    )
    assert result.private_outputs["all-scores.jsonl"] == b"not scientific evidence"
