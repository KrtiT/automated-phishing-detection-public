"""Invented old/new profile identities with byte-preserved scientific source."""

from pathlib import Path
from types import SimpleNamespace

from study_urlnorm_fixtures import digest
from study_urlnorm_fixtures import profile as origin_profile

from automated_phishing_detection import _study_execution_policy as original


def origin():
    component = SimpleNamespace(profile_sha256=digest(b"original-component"))
    case = SimpleNamespace(
        root=Path("/invented/repository"),
        policy=original.policy_bytes(),
        external=component,
        operational=component,
        base=SimpleNamespace(
            revision="a" * 40,
            contract_sha256=original.CONTRACT_SHA256,
            source_hashes=(
                ("src/automated_phishing_detection/kernel.py", digest(b"kernel")),
            ),
        ),
    )
    return origin_profile(case)


def origin_record(prior):
    return {
        "attempt_ordinal": 4,
        "profile": prior,
        "revision": "a" * 40,
        "profile_sha256": digest(prior),
        "policy_sha256": prior["policy_sha256"],
        "method_sha256": prior["method_sha256"],
        **{
            name: digest(name.encode())
            for name in (
                "envelope_sha256",
                "root_reservation_sha256",
                "preparation_reservation_sha256",
                "preparation_complete_sha256",
            )
        },
    }


def metadata(policy, prior):
    additions = {
        name: digest(name.encode()) for name in policy["permitted_added_paths"]
    }
    unchanged = prior["source_artifact_scope"]
    return {
        "schema_version": 1,
        "profile_id": "study-series-profile-v1",
        "series_id": "invented-series",
        "operator": prior["session"]["operator"],
        "policy_sha256": digest(policy),
        "amendment_sha256": digest(policy["amendment"]),
        "execution": {
            "revision": "b" * 40,
            "contract_sha256": original.CONTRACT_SHA256,
            "runtime_sha256": digest(b"invented-runtime"),
        },
        "origin": origin_record(prior),
        "components": components(prior),
        "source_artifact_scope": unchanged | additions,
        "transition": {"unchanged_sha256": unchanged.copy(), "added_sha256": additions},
        "scientific_pins": scientific_pins(),
    }


def components(prior):
    return {
        f"{period}_{kind}": prior["components"][kind]
        if period == "original"
        else digest(f"{period}-{kind}".encode())
        for period in ("original", "current")
        for kind in ("external", "operational")
    }


def scientific_pins():
    return {
        name: digest(name.encode())
        for name in (
            "source_spec_sha256",
            "primary_metadata_sha256",
            "internal_bindings_sha256",
            "external_bindings_sha256",
            "schedule_sha256",
            "original_h2_audit_sha256",
        )
    }


def paths():
    return {"repo_root": "/invented/repository"} | {
        name: f"/invented/fresh/{name}"
        for name in (
            "series_attempt",
            "series_public_summary",
            "segment_attempt",
            "segment_public_summary",
            "historical_inputs_dir",
            "cells_dir",
            "physical_records_dir",
        )
    }


def segment():
    return {
        "ordinal": 2,
        "session_id": "invented-next-session",
        "start_ordinal": 73,
        "end_ordinal": 125,
        "predecessor_accounting_sha256": digest(b"accounting"),
        "session_requirements": dict(original.SESSION_REQUIREMENTS),
        "operator_commitment": "exclusive_session_conditions_and_pre_post_records_required",
    }


def profile(policy):
    value = metadata(policy, origin())
    value["history"] = {
        f"{name}_{suffix}": f"/invented/history/{name}.json"
        if suffix == "path"
        else digest(name.encode())
        for name in ("index", "eligible_prefix_review", "exposure_record")
        for suffix in ("path", "sha256")
    }
    value["paths"], value["segment"] = paths(), segment()
    value["invocation"] = {
        "script": "scripts/run_study_series.py",
        "arguments": {
            "repo-root": value["paths"]["repo_root"],
            "expected-revision": "b" * 40,
        },
    }
    return value
