"""Pure root envelopes preserve existing identities without paths or new science."""

import inspect
import json
from dataclasses import replace
from hashlib import sha256

import pytest
from operational_input_fixtures import candidates, manifests
from operational_profile_fixtures import profile_case, resolve
from study_run_record_fixtures import api, attempt

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["profile_case", "candidates", "manifests"]


def test_study_identity_is_closed_and_fresh(profile_case):
    profile = resolve(profile_case)
    identity = api().study_identity(profile_case.binding, profile)
    assert identity == {
        "kind": "whole_study",
        "protocol": "study-root-v1",
        **profile.projection()["execution"],
        "operational_profile_sha256": profile.profile_sha256,
    }
    identity["kind"] = "mutated"
    assert api().study_identity(profile_case.binding, profile)["kind"] == "whole_study"


def test_intent_binds_actual_reservation_profile_and_explicit_deadlines(profile_case):
    profile = resolve(profile_case)
    identity = api().study_identity(profile_case.binding, profile)
    reserved = attempt(identity)
    deadlines = dict(startup=10.0, shutdown=5.0, terminate=2.0, kill=1.0)
    content = api().study_intent(profile_case.binding, profile, reserved, deadlines)
    value = json.loads(content)
    assert value == {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "intent",
        "execution": identity | {"reservation_sha256": reserved.reservation_sha256},
        "operational_profile": profile.projection(),
        "protective_deadlines_seconds": deadlines,
    }
    assert content == canonical_bytes(value) and content.isascii()
    assert b"/invented" not in content and b"directory" not in content


@pytest.mark.parametrize(
    "change", ["profile", "mutable", "reservation", "deadline", "extra"]
)
def test_intent_rejects_unbound_or_implicit_context(profile_case, change):
    profile = resolve(profile_case)
    identity = api().study_identity(profile_case.binding, profile)
    reserved = attempt(identity)
    deadlines = dict(startup=10.0, shutdown=5.0, terminate=2.0, kill=1.0)
    if change in ("profile", "mutable"):
        content = b"{}\n" if change == "profile" else bytearray(profile.canonical_bytes)
        profile = replace(profile, canonical_bytes=content)
    elif change == "reservation":
        reserved = replace(reserved, reservation_sha256="f" * 64)
    elif change == "deadline":
        deadlines["startup"] = False
    else:
        deadlines["retry"] = 1
    with pytest.raises(ValueError):
        api().study_intent(profile_case.binding, profile, reserved, deadlines)


def test_source_results_preserve_original_metadata_and_digest(manifests):
    from operational_cell_runner_fixtures import accepted_source

    source, profile, accepted = accepted_source(manifests)
    identity = api().study_identity(source.binding, profile)
    execution = identity | {"reservation_sha256": source.reservation}
    content = api().source_results(accepted, execution=execution)
    value = json.loads(content)
    assert value == {
        "schema_version": 1,
        "protocol": "study-root-v1",
        "status": "sources_accepted",
        "execution": execution,
        "accepted_inputs": json.loads(accepted.metadata_bytes),
        "accepted_inputs_sha256": sha256(accepted.metadata_bytes).hexdigest(),
    }
    assert canonical_bytes(value["accepted_inputs"]) == accepted.metadata_bytes


def test_source_results_reject_cross_root_before_new_projection(manifests):
    from operational_cell_runner_fixtures import accepted_source

    source, profile, accepted = accepted_source(manifests)
    execution = api().study_identity(source.binding, profile) | {
        "reservation_sha256": "1" * 64
    }
    with pytest.raises(ValueError):
        api().source_results(accepted, execution=execution)


def test_builder_signatures_have_no_authority_override():
    assert tuple(inspect.signature(api().prediction_barrier).parameters) == (
        "preparation",
        "execution",
    )
    assert tuple(inspect.signature(api().root_public_summary).parameters) == (
        "execution",
        "checkpoints",
        "reduced",
    )
