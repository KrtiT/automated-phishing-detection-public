"""Authenticate declared origin, additive scope and prospective metadata only."""

from . import _study_execution_policy as original
from . import _study_execution_schema as legacy
from . import _study_series_adoption_schema as schema
from . import _study_urlnorm_policy as urlnorm


def _origin(value):
    origin = value["origin"]
    schema.closed(origin, "origin")
    schema.ordinal(origin["attempt_ordinal"], 4, 4)
    schema.pins(origin)
    prior = origin["profile"]
    legacy.profile(prior)
    legacy.require(prior["profile_id"] == "study-urlnorm-profile-v1")
    legacy.require(schema.digest(prior) == origin["profile_sha256"])
    legacy.require(origin["revision"] == prior["execution"]["revision"])
    policy = urlnorm.policy_projection()
    legacy.require(
        origin["policy_sha256"] == schema.digest(policy) == prior["policy_sha256"]
    )
    legacy.require(
        origin["method_sha256"]
        == schema.digest(policy["method"])
        == prior["method_sha256"]
    )
    legacy.require(value["operator"] == prior["session"]["operator"])
    return prior


def _execution(value, policy, prior):
    execution = value["execution"]
    schema.closed(execution, "execution")
    legacy.digest(execution["revision"], 40)
    schema.pins(execution)
    legacy.require(execution["revision"] != prior["execution"]["revision"])
    legacy.require(execution["contract_sha256"] == original.CONTRACT_SHA256)
    legacy.require(value["policy_sha256"] == schema.digest(policy))
    legacy.require(value["amendment_sha256"] == schema.digest(policy["amendment"]))
    legacy.require(value["amendment_sha256"] != value["origin"]["method_sha256"])
    components = value["components"]
    schema.closed(components, "components")
    for kind in ("external", "operational"):
        before, after = components[f"original_{kind}"], components[f"current_{kind}"]
        legacy.digest(before)
        legacy.digest(after)
        legacy.require(before == prior["components"][kind] and after != before)


def _scope(value, policy, prior):
    transition = value["transition"]
    schema.closed(transition, "transition")
    unchanged, added = transition["unchanged_sha256"], transition["added_sha256"]
    for mapping in (unchanged, added, value["source_artifact_scope"]):
        schema.file_map(mapping)
    legacy.require(unchanged == prior["source_artifact_scope"])
    legacy.require(not unchanged.keys() & added.keys())
    legacy.require(set(added) == set(policy["permitted_added_paths"]))
    legacy.require(value["source_artifact_scope"] == unchanged | added)


def _segment(value, prior):
    segment = value["segment"]
    schema.closed(segment, "segment")
    schema.ordinal(segment["ordinal"], 2, 2)
    schema.ordinal(segment["start_ordinal"], 2, 125)
    schema.ordinal(segment["end_ordinal"], 125, 125)
    legacy.text(segment["session_id"])
    legacy.require(segment["session_id"] != prior["session"]["session_id"])
    legacy.require(segment["session_requirements"] == original.SESSION_REQUIREMENTS)
    legacy.require(
        segment["operator_commitment"]
        == "exclusive_session_conditions_and_pre_post_records_required"
    )
    schema.pins(segment)


def _paths(value, prior):
    schema.closed(value["paths"], "paths")
    paths = {name: legacy.lexical_path(path) for name, path in value["paths"].items()}
    history = value["history"]
    schema.closed(history, "history")
    schema.pins(history)
    ancestors = (prior, prior["continuation"]["prior_profile"])
    protected = [
        legacy.lexical_path(path)
        for ancestor in ancestors
        for path in ancestor["paths"].values()
    ]
    protected += [
        legacy.lexical_path(path)
        for name, path in history.items()
        if name.endswith("_path")
    ]
    protected.append(paths["repo_root"])
    outputs = [path for name, path in paths.items() if name != "repo_root"]
    for index, path in enumerate(outputs):
        for other in (*protected, *outputs[:index]):
            schema.disjoint(path, other)


def _invocation(value, policy):
    invocation = value["invocation"]
    schema.closed(invocation, "invocation")
    schema.closed(invocation["arguments"], "arguments")
    legacy.require(invocation["script"] == policy["scripts"]["root"])
    legacy.require(
        invocation["arguments"]
        == {
            "repo-root": value["paths"]["repo_root"],
            "expected-revision": value["execution"]["revision"],
        }
    )


def profile(value, policy):
    schema.closed(value, "profile")
    schema.version(value["schema_version"])
    legacy.require(value["profile_id"] == "study-series-profile-v1")
    for name in ("series_id", "operator"):
        legacy.text(value[name])
    schema.pins(value)
    prior = _origin(value)
    _execution(value, policy, prior)
    _scope(value, policy, prior)
    _segment(value, prior)
    _paths(value, prior)
    _invocation(value, policy)
    schema.closed(value["scientific_pins"], "scientific_pins")
    schema.pins(value["scientific_pins"])
