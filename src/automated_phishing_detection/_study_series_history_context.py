"""Bind reviewed complete archives before reconstructing selected history."""

from pathlib import Path

from . import _study_execution_schema as schema
from . import _study_series_index_refs as refs
from .study_series_history_index import validate_series_history_index


def payloads(reader, members):
    return {
        name: reader.read(value["path"], value["sha256"])
        for name, value in members.items()
    }


def pins(members):
    return {name: value["sha256"] for name, value in members.items()}


def reference(reader, value):
    refs.reference(value)
    return reader.read(value["path"], value["sha256"])


def _review(reader, profile, index):
    history = profile["history"]
    content = reader.read(
        history["eligible_prefix_review_path"], history["eligible_prefix_review_sha256"]
    )
    review = schema.parse(content, history["eligible_prefix_review_sha256"])
    _review_identity(review, profile, index)
    _review_result(review)
    for name in ("custody_review", "scientific_report"):
        reference(reader, review[name])


def _review_identity(review, profile, index):
    schema.closed(
        review,
        "schema_version review_id index_sha256 selected_root_reservation_sha256 accepted_ordinals stopped_ordinal source_bindings_sha256 custody_review scientific_report interruption_cause physical_evidence_scope limitations status authorizes_execution".split(),
    )
    schema.require(
        type(review["schema_version"]) is int and review["schema_version"] == 1
    )
    schema.require(review["review_id"] == "study-series-eligible-prefix-review-v1")
    schema.require(review["index_sha256"] == index.index_sha256)
    schema.require(
        review["selected_root_reservation_sha256"]
        == profile["origin"]["root_reservation_sha256"]
    )
    _review_prefix(review, profile, index)


def _review_prefix(review, profile, index):
    schema.require(type(review["accepted_ordinals"]) is list)
    schema.require(all(type(ordinal) is int for ordinal in review["accepted_ordinals"]))
    schema.require(review["accepted_ordinals"] == list(index.accepted_ordinals))
    schema.require(
        type(review["stopped_ordinal"]) is int
        and review["stopped_ordinal"] == index.stopped_ordinal
    )
    expected = {
        kind: profile["scientific_pins"][kind + "_bindings_sha256"]
        for kind in ("internal", "external")
    }
    schema.require(review["source_bindings_sha256"] == expected)


def _review_result(review):
    schema.require(
        review["status"] == "eligible_entire_prefix"
        and review["authorizes_execution"] is False
    )
    schema.require(
        review["interruption_cause"] == "supervisor_ac_power_loss_after_prefix"
    )
    schema.require(
        review["physical_evidence_scope"]
        == "sampled_prefix_with_next_child_witness_not_continuous_power"
    )
    schema.require(type(review["limitations"]) is list and review["limitations"])
    for limitation in review["limitations"]:
        schema.text(limitation)


def _exposure(reader, profile, index):
    history = profile["history"]
    content = reader.read(
        history["exposure_record_path"], history["exposure_record_sha256"]
    )
    value = schema.parse(content, history["exposure_record_sha256"])
    schema.closed(
        value,
        "schema_version record_id index_sha256 decision_record verification_review exposed_retained_results blinding_claimed original_failures_preserved prior_results_selection_rule session_confounding_disclosed notes".split(),
    )
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["record_id"] == "study-series-exposure-v1")
    schema.require(value["index_sha256"] == index.index_sha256)
    schema.require(
        value["exposed_retained_results"] is True and value["blinding_claimed"] is False
    )
    schema.require(value["original_failures_preserved"] is True)
    schema.require(value["session_confounding_disclosed"] is True)
    schema.require(
        value["prior_results_selection_rule"]
        == "entire_eligible_attempt4_prefix_no_metric_selection"
    )
    schema.text(value["notes"])
    for name in ("decision_record", "verification_review"):
        reference(reader, value[name])


def _catalog(reader, entry, *, reviewed):
    content = reference(reader, entry["inventory"])
    value = schema.parse(content, entry["inventory"]["sha256"])
    schema.closed(value, ("schema_version", "inventory_id", "files"))
    schema.require(
        type(value["schema_version"]) is int and value["schema_version"] == 1
    )
    schema.require(value["inventory_id"] == "study-series-retained-inventory-v1")
    schema.require(type(value["files"]) is dict and value["files"])
    original = schema.parse(
        reference(reader, entry["profile"]), entry["profile"]["sha256"]
    )
    reference(reader, entry["envelope"])
    root = Path(original["paths"]["attempt"]).parent
    paths = _catalog_paths(value["files"], root, Path(entry["profile"]["path"]).parent)
    if reviewed:
        review = schema.parse(
            reference(reader, entry["interruption_review"]),
            entry["interruption_review"]["sha256"],
        )
        schema.require(review["catalog_sha256"] == entry["inventory"]["sha256"])
        schema.require(review["catalog_files"] == sorted(paths))
        schema.require(
            review["root_reservation_sha256"] == entry["root_reservation_sha256"]
        )
        schema.require(review["disposition"] == entry["disposition"])
    for member in value["files"].values():
        reader.read(member["path"], member["sha256"], retain=False)
    reader.tree(root, tuple(path for path in paths if Path(path).is_relative_to(root)))


def _catalog_paths(values, root, sidecar_parent):
    paths = []
    suffixes = (
        "pre.json",
        "launch.json",
        "conditions.jsonl",
        "post.json",
        "sleep-cleanup.json",
        "stdout.log",
        "stderr.log",
    )
    for name, member in values.items():
        refs.reference(member)
        schema.require(type(name) is str and "/" in name)
        kind, relative = name.split("/", 1)
        selected = schema.lexical_path(member["path"])
        if kind == "tree":
            schema.require(
                selected == root / relative and selected.is_relative_to(root)
            )
        else:
            schema.require(kind == "sidecars" and "/" not in relative)
            schema.require(
                relative.startswith("study-session-") and relative.endswith(suffixes)
            )
            schema.require(selected == sidecar_parent / relative)
        paths.append(str(selected))
    schema.require(len(paths) == len(set(paths)))
    return paths


def archives(reader, public):
    profile = schema.parse(public.profile_bytes, public.profile_sha256)
    history = profile["history"]
    content = reader.read(history["index_path"], history["index_sha256"])
    index = validate_series_history_index(
        content,
        public.profile_bytes,
        expected_index_sha256=history["index_sha256"],
        expected_profile_sha256=public.profile_sha256,
    )
    value = schema.parse(content, index.index_sha256)
    _review(reader, profile, index)
    _exposure(reader, profile, index)
    for entry in value["attempts"]:
        _catalog(reader, entry, reviewed=True)
    _catalog(reader, value["original_hold"], reviewed=False)
    return index, value, profile
