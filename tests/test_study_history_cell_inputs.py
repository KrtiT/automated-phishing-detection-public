"""Source science and exact workload selections rejoin original accepted bytes."""

import json
from dataclasses import replace

import pytest
from study_history_cell_fixtures import (
    api,
    candidates,
    case,
    history,
    manifests,
    restore,
)
from study_history_cell_mutation_fixtures import (
    rebind_context,
    rebind_source_payloads,
    reordered_manifest,
)

from automated_phishing_detection._checkpoint_codec import canonical_bytes

__all__ = ["candidates", "case", "history", "manifests"]


@pytest.mark.parametrize("role", ("internal", "external"))
@pytest.mark.parametrize("kind", ("missing", "extra", "duplicate", "changed", "list"))
def test_source_inventory_is_exact_and_bound(history, role, kind):
    name = f"{role}_snapshot"
    snapshot = history.arguments[name]
    pairs = snapshot.payloads
    variants = {
        "missing": pairs[:-1],
        "extra": (*pairs, ("extra", b"invented")),
        "duplicate": (*pairs, pairs[0]),
        "list": list(pairs),
        "changed": ((pairs[0][0], pairs[0][1] + b"\n"), *pairs[1:]),
    }
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, **{name: replace(snapshot, payloads=variants[kind])})


@pytest.mark.parametrize("field", ("half_width", "logistic_l1", "monitor_boundary"))
def test_rejoined_metadata_cannot_substitute_primary_policy(history, field):
    metadata = json.loads(history.arguments["accepted_metadata_bytes"])
    metadata["primary"]["thresholds"][field] += 0.001
    changed = rebind_context(history, metadata=metadata)
    with pytest.raises(api().HistoricalCellScienceError):
        restore(changed)


def test_rehashed_internal_execution_must_match_metadata(history):
    first = dict(history.arguments["internal_snapshot"].payloads)
    second = dict(history.arguments["external_snapshot"].payloads)
    public = json.loads(first["public-summary.json"])
    public["execution"]["revision"] = "0" * 40
    first["public-summary.json"] = canonical_bytes(public)
    changed = rebind_source_payloads(history, first, second)
    with pytest.raises(api().HistoricalCellScienceError):
        restore(changed)


def test_both_rehashed_primary_bindings_must_match_original_metadata(history):
    snapshots = [
        dict(history.arguments[f"{role}_snapshot"].payloads)
        for role in ("internal", "external")
    ]
    for values in snapshots:
        binding = json.loads(values["attempt/evidence/bindings.json"])
        binding["thresholds"]["logistic_l1"] += 0.001
        values["attempt/evidence/bindings.json"] = canonical_bytes(binding)
    changed = rebind_source_payloads(history, *snapshots)
    with pytest.raises(api().HistoricalCellScienceError):
        restore(changed)


def test_external_profile_remains_bound_to_original_execution(history):
    snapshot = history.arguments["external_snapshot"]
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, external_snapshot=replace(snapshot, profile_bytes=b"{}"))


@pytest.mark.parametrize("field", ("manifest_sha256", "cell"))
def test_rejoined_descriptor_cannot_choose_other_manifest_or_cell(history, field):
    descriptor = json.loads(history.arguments["descriptor_bytes"])
    if field == "manifest_sha256":
        descriptor[field] = "0" * 64
    else:
        descriptor[field]["run_index"] += 1
        descriptor[field]["ordinal"] += 1
    changed = rebind_context(history, descriptor=descriptor)
    with pytest.raises(api().HistoricalCellScienceError):
        restore(changed)


@pytest.mark.parametrize("kind", ("reorder", "missing", "capacity", "wrong_prevalence"))
def test_exact_verified_manifest_outcome_and_external_row_order(history, kind):
    if history.working.inputs.cell.workload == "shift_period":
        snapshot = history.arguments["external_snapshot"]
        rows = snapshot.rows[::-1] if kind == "reorder" else snapshot.rows[:999]
        changes = {"external_snapshot": replace(snapshot, rows=rows)}
    else:
        snapshot = history.arguments["internal_snapshot"]
        outcomes = dict(snapshot.manifest_outcomes)
        prevalence = history.working.inputs.cell.prevalence_basis_points
        outcome = outcomes[prevalence]
        if kind == "missing":
            del outcomes[prevalence]
        elif kind == "capacity":
            outcomes[prevalence] = replace(outcome, status="insufficient_capacity")
        elif kind == "reorder":
            outcomes[prevalence] = replace(
                outcome, manifest=reordered_manifest(outcome.manifest)
            )
        else:
            outcomes[prevalence] = replace(outcome, manifest=outcomes[10].manifest)
        changes = {
            "internal_snapshot": replace(
                snapshot, manifest_outcomes=tuple(outcomes.items())
            )
        }
    with pytest.raises(api().HistoricalCellScienceError):
        restore(history, **changes)
