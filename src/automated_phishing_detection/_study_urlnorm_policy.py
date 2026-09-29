"""One explicit retained-data scientific amendment; no implicit retry authority."""

from . import _study_execution_policy as original
from ._checkpoint_codec import canonical_bytes

POLICY_PATH = "data/study-execution-policy-v3.json"
REPRESENTATION = "publisher_url_norm_v1"
SCOPE = "publisher_url_norm_retained_data_continuation"


def policy_projection():
    value = original.policy_projection()
    value.update(
        schema_version=3,
        policy_id="study-execution-policy-v3",
        authorization_basis=SCOPE,
        representation=REPRESENTATION,
        decisions=["method", "profile", "access"],
    )
    value["method"][0] = (
        "Authenticate one prior prediction-free whole-study hold and its retained "
        "preparation. Create one new retained-data preparation without reopening "
        "original sources. Reuse the unchanged internal population and assess all "
        "thirteen necessary-capacity conditions before predictions; hold the "
        "entire study on any shortage."
    )
    value["method"][1] += (
        " The sole external representation amendment uses exact publisher "
        "url_norm uniformly for parsing and model inputs, retaining original raw "
        "cells and provenance. No fallback, local repair, refit or recalibration "
        "is permitted. Disclose prior preparation and label-count exposure."
    )
    value["method"][-1] = (
        "Require the named operator's explicit retained-data URL amendment "
        "directive and exact effective profile, ancestry and access binding. "
        "Preserve the prior operator-authorized hold and advisor-concurrence "
        "waiver without claiming advisor approval or personal review of future "
        "implementation hashes. The original hold is never resumed or replaced."
    )
    return value


def policy_bytes():
    return canonical_bytes(policy_projection())
