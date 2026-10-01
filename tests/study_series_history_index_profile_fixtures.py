"""Invented pinned profiles with public-source identities for locator joins."""

from study_series_adoption_fixtures import digest
from study_series_adoption_profile_fixtures import profile

from automated_phishing_detection import _study_series_policy as policy


def series_profile(prefix):
    value = profile(policy.policy_projection())
    prior = value["origin"]["profile"]
    scope = prior["source_artifact_scope"] | {
        name: digest(name.encode())
        for name in ("data/sources.json", "reports/phiusiil-preparation-summary.json")
    }
    prior["source_artifact_scope"] = scope
    ancestor = prior["continuation"]["prior_profile"]
    ancestor["source_artifact_scope"] = scope.copy()
    prior["continuation"]["prior_profile_sha256"] = digest(ancestor)
    value["origin"]["profile_sha256"] = digest(prior)
    value["transition"]["unchanged_sha256"] = scope.copy()
    value["source_artifact_scope"] = scope | value["transition"]["added_sha256"]
    value["scientific_pins"]["source_spec_sha256"] = scope["data/sources.json"]
    value["segment"]["start_ordinal"] = prefix + 1
    return value
