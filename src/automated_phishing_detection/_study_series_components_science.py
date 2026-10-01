"""Hash-projected original science and the independently pinned public audit."""

from hashlib import sha256

from . import _operational_input_schema as inputs
from . import _operational_profile as operational
from . import _study_execution_schema as schema
from . import bound_models, gmm_monitor
from ._checkpoint_codec import canonical_bytes

AUDIT_SUMMARY_PATH = "reports/rq2-gmm-development-v1-summary.json"


def _original(profile, value, kind):
    original = value | {
        "execution": value["execution"] | {"revision": profile["origin"]["revision"]}
    }
    if kind == "operational":
        original["bound_file_sha256"] = profile["transition"]["unchanged_sha256"]
    content = canonical_bytes(original)
    schema.require(
        sha256(content).hexdigest() == profile["components"][f"original_{kind}"]
    )
    return content


def components(profile, external, workload):
    schedule = canonical_bytes(workload["schedule"])
    schema.require(schedule == canonical_bytes(operational._schedule()))
    schema.require(
        sha256(schedule).hexdigest() == profile["scientific_pins"]["schedule_sha256"]
    )
    return (
        _original(profile, external, "external"),
        _original(profile, workload, "operational"),
    )


def audit(profile, content):
    expected = profile["transition"]["unchanged_sha256"][AUDIT_SUMMARY_PATH]
    schema.require(type(content) is bytes and sha256(content).hexdigest() == expected)
    summary = inputs.loads(content, canonical=False)
    gmm_monitor._validate_public_summary(summary)
    bound_models._validate_gmm_audit(summary)
    schema.require(summary["status"] == "completed_development_validation")
    schema.require(summary["analysis_stage"] == "development_validation_only")
    schema.require(
        type(summary["schema_version"]) is int and summary["schema_version"] == 1
    )
    schema.require(
        summary["audit_alert_count"] == 28 and summary["audit_window_count"] == 252
    )
    schema.digest(summary["artifact_hashes"]["validation-audit.json"])
    schema.require(
        summary["artifact_hashes"]["validation-audit.json"]
        == profile["scientific_pins"]["original_h2_audit_sha256"]
    )
