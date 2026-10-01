"""Full invented public components; construction never establishes preflight."""

from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

from study_series_adoption_profile_fixtures import origin, origin_record, profile
from study_urlnorm_fixtures import digest

from automated_phishing_detection import _external_source_profile as external
from automated_phishing_detection import _operational_profile as operational
from automated_phishing_detection import _study_series_policy as policy
from automated_phishing_detection import execution_preflight as preflight
from automated_phishing_detection._checkpoint_codec import canonical_bytes

AUDIT_PATH = "reports/rq2-gmm-development-v1-summary.json"


def api():
    name = "automated_phishing_detection.study_series_components"
    assert find_spec(name) is not None, "missing pure series component transition"
    return import_module(name)


def audit_summary():
    return {
        "schema_version": 1,
        "status": "completed_development_validation",
        "analysis_stage": "development_validation_only",
        "contract": {"id": "rq2-gmm-development-v1", "sha256": digest(b"contract")},
        "hypothesis_status": {"H2": "not_supported"},
        "access": {"protected_evaluation_ready": False},
        "input_hashes": {},
        "artifact_hashes": {
            "validation-audit.json": digest(b"invented saved audit"),
            "gmm.json": digest(b"invented GMM artifact"),
        },
        "input_counts": {"audit": {"complete_windows": 252}},
        "candidates": [{"components": 1, "bic": 100.0, "converged": True, "n_iter": 2}],
        "selected_component_count": 1,
        "threshold": 2.0,
        "calibration_window_count": 252,
        "audit_alert_count": 28,
        "audit_window_count": 252,
        "audit_alert_fraction": 28 / 252,
        "false_alert_gate_met": False,
        "software_versions": {},
        "numpy_build_configuration": "invented",
        "configuration": {},
        "warnings": [],
    }


def external_value(base):
    pins = dict(base.source_hashes)
    return {
        "schema_version": 1,
        "profile_id": "external-source-candidate-v1",
        "status": "specified_closed_candidate",
        "protected_evaluation_ready": False,
        "protected_evaluation_authorized": False,
        "execution": external._execution(base, pins),
        "publisher": external._publisher(),
        "public_suffix_list": {
            "url": "https://invented.invalid/public-suffix-list",
            "upstream_url": "https://invented.invalid/upstream",
            "sha256": digest(b"invented suffix rules"),
        },
        "retention": external._retention(),
        "implementation_sha256": {
            name: pins[name] for name in external._IMPLEMENTATIONS
        },
    }


def original_scope(audit):
    names = {*external._IMPLEMENTATIONS, *operational._REQUIRED}
    names |= {external._SOURCE, "reports/phiusiil-preparation-summary.json", AUDIT_PATH}
    return {name: digest(name.encode()) for name in names} | {AUDIT_PATH: digest(audit)}


def original_profile(scope, original_external, original_operational):
    value = origin()
    value["source_artifact_scope"] = scope.copy()
    value["components"] = {
        "external": digest(original_external),
        "operational": digest(original_operational),
    }
    prior = value["continuation"]["prior_profile"]
    prior["source_artifact_scope"] = scope.copy()
    prior["components"] = value["components"].copy()
    value["continuation"]["prior_profile_sha256"] = digest(prior)
    return value


def current_base(value, scope):
    current = scope | value["transition"]["added_sha256"]
    return preflight.ExecutionBinding(
        Path(value["paths"]["repo_root"]),
        "b" * 40,
        value["execution"]["contract_sha256"],
        tuple(sorted(current.items())),
        '{"invented_runtime":true}',
    )


def joined_profile(value, base, scope, before, after):
    value["origin"] = origin_record(original_profile(scope, *before))
    value["source_artifact_scope"] = dict(base.source_hashes)
    value["transition"]["unchanged_sha256"] = scope.copy()
    value["execution"]["runtime_sha256"] = digest(base.runtime_json.encode())
    value["components"] = {
        f"{period}_{kind}": digest(content)
        for period, contents in (("original", before), ("current", after))
        for kind, content in zip(("external", "operational"), contents, strict=True)
    }
    value["scientific_pins"].update(
        source_spec_sha256=scope[external._SOURCE],
        schedule_sha256=digest(operational._schedule()),
        original_h2_audit_sha256=audit_summary()["artifact_hashes"][
            "validation-audit.json"
        ],
    )
    return value


def component_bytes(base):
    return (
        canonical_bytes(external_value(base)),
        canonical_bytes(operational._projection(base)),
    )


def case():
    audit = canonical_bytes(audit_summary())
    scope = original_scope(audit)
    value = profile(policy.policy_projection())
    base = current_base(value, scope)
    original = replace(
        base, revision="a" * 40, source_hashes=tuple(sorted(scope.items()))
    )
    before, after = component_bytes(original), component_bytes(base)
    value = joined_profile(value, base, scope, before, after)
    return SimpleNamespace(
        base=base, profile=value, audit=audit, before=before, after=after
    )


def arguments(selected):
    content = canonical_bytes(selected.profile)
    return {
        "base": selected.base,
        "external": external.CandidateExternalProfile(selected.after[0]),
        "operational": operational.CandidateOperationalProfile(selected.after[1]),
        "profile_bytes": content,
        "audit_summary_bytes": selected.audit,
        "expected_profile_sha256": digest(content),
    }


def rewrite_component(selected, kind, value):
    content = canonical_bytes(value)
    index = ("external", "operational").index(kind)
    changed = list(selected.after)
    changed[index] = content
    selected.after = tuple(changed)
    selected.profile["components"][f"current_{kind}"] = digest(content)


def repin_audit(selected, content):
    selected.audit = content
    scope = original_scope(content)
    selected.base = current_base(selected.profile, scope)
    original = replace(
        selected.base, revision="a" * 40, source_hashes=tuple(sorted(scope.items()))
    )
    selected.before = component_bytes(original)
    selected.after = component_bytes(selected.base)
    joined_profile(
        selected.profile, selected.base, scope, selected.before, selected.after
    )
