"""Full-size original workload fixtures with constructed source/exit context only."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace

from operational_cell_shift_fixtures import scored_source
from operational_input_fixtures import build
from operational_summary_fixtures import _phase, counters, http_run
from shift_run_codec_fixtures import _complete_case, outcome
from study_evidence_fixtures import evidence
from study_operational_fixtures import bound, compact, digest

from automated_phishing_detection import execution_receipt, operational_inputs
from automated_phishing_detection._checkpoint_codec import canonical_bytes
from automated_phishing_detection.external_source_handoff import _Population
from automated_phishing_detection.http_replay import summarize_run
from automated_phishing_detection.http_run_codec import encode_http_run
from automated_phishing_detection.operational_cell_acceptance import cell_identity
from automated_phishing_detection.operational_cell_inputs import build_cell_descriptor
from automated_phishing_detection.operational_schedule import planned_cells
from automated_phishing_detection.shift_replay import summarize_shift_run
from automated_phishing_detection.shift_run_codec import encode_shift_run
from automated_phishing_detection.shift_schema import ShiftPlan


def _external_population(snapshot, external):
    return replace(
        snapshot,
        populations=tuple(
            (name, _Population(value.records, tuple(sorted(value.predictions.items()))))
            for name, value in sorted(external.populations.items())
        ),
        control_ids=external.controls.record_ids,
        control_columns=tuple(sorted(external.controls.predictions.items())),
        external_windows=external.external_windows,
        policy=external.replay,
        role_counts=tuple(sorted(external.role_counts.items())),
    )


def _sources(case):
    case, supplied = scored_source(case), evidence()
    population = supplied["internal"]
    internal = replace(
        case.internal,
        snapshot=replace(
            case.internal.snapshot,
            records=population.records,
            prediction_columns=tuple(sorted(population.predictions.items())),
        ),
    )
    snapshot = _external_population(case.external.snapshot, supplied["external"])
    return SimpleNamespace(
        **(
            vars(case)
            | {
                "internal": internal,
                "external": replace(case.external, snapshot=snapshot),
            }
        )
    )


def _http_phase(run, selected, phase, digest_value, reference):
    rows = []
    for position, row in enumerate(getattr(run, phase)):
        identity = row.request_id.replace(run.manifest_sha256, digest_value)
        response = row.response
        if response is not None:
            changes = {"request_id": identity}
            if reference and phase == "measured":
                changes["admission_sequence"] = 1001
            response = response.model_copy(update=changes)
        rows.append(
            replace(
                row,
                record_id=selected.requests[position].record_id,
                request_id=identity,
                response=response,
            )
        )
    return tuple(rows)


def _http(cell, selected):
    original = http_run(cell)
    digest_value = digest(selected.manifest_bytes)
    phases = {
        phase: _http_phase(original, selected, phase, digest_value, cell.ordinal == 1)
        for phase in ("warmup", "measured")
    }
    if cell.ordinal == 1:
        phases.update(
            after_warmup=counters(1000, 400), after_measured=counters(11000, 4400)
        )
    result = replace(original, manifest_sha256=digest_value, **phases)
    _phase.cache_clear()
    return result


def _shift(cell, selected):
    plan = ShiftPlan(digest(selected.manifest_bytes), cell.run_index, selected.requests)
    warmup = tuple(outcome(plan, "warmup", position) for position in range(1000))
    measured = tuple(
        outcome(plan, "measured", position, "timeout" if position in (0, 999) else None)
        for position in range(len(plan.requests))
    )
    run = _complete_case(plan, warmup, measured)[0]
    rows = []
    for row in run.trace.rows:
        window = (
            None
            if row.window is None
            else row.window.model_copy(update={"score": -100.0})
        )
        rows.append(row.model_copy(update={"monitor_nll": -100.0, "window": window}))
    return replace(run, trace=run.trace.model_copy(update={"rows": rows}))


def _record(api, accepted, selected, cell, run):
    original = compact(api, cell.ordinal)
    descriptor = canonical_bytes(
        json.loads(selected.descriptor_bytes) | {"cell": asdict(cell)}
    )
    binding = json.loads(original.binding_bytes)
    reservation = binding["cell_reservation_sha256"]
    is_shift = cell.workload == "shift_period"
    content = (encode_shift_run if is_shift else encode_http_run)(run)
    summary = (summarize_shift_run if is_shift else summarize_run)(run)
    hashes = dict(original.snapshot_sha256)
    for prefix in ("attempt", "attempt/evidence"):
        hashes[f"{prefix}/run.json"] = digest(content)
    public = json.loads(original.public_bytes)
    public.update(
        execution=cell_identity(accepted, descriptor)
        | {"reservation_sha256": reservation},
        summary=summary,
    )
    public["private_sha256"]["run.json"] = digest(content)
    public_bytes = execution_receipt._json_bytes(public, "fixture")
    hashes["public-summary.json"] = digest(public_bytes)
    return replace(
        original,
        descriptor_bytes=descriptor,
        binding_bytes=bound(descriptor, reservation),
        run_bytes=content,
        public_bytes=public_bytes,
        snapshot_sha256=tuple(sorted(hashes.items())),
    )


def full_case(api, source):
    accepted = build(operational_inputs, _sources(source))
    pool, records, runs = {}, [], []
    for cell in planned_cells():
        key = cell.prevalence_basis_points
        if key not in pool:
            pool[key] = build_cell_descriptor(accepted, cell)
        selected = pool[key]
        run = (
            _shift(cell, selected)
            if cell.workload == "shift_period"
            else _http(cell, selected)
        )
        records.append(_record(api, accepted, selected, cell, run))
        runs.append(run)
    return SimpleNamespace(
        accepted=accepted,
        slots=api.freeze_cell_accounting(tuple(records)),
        runs=tuple(runs),
    )
