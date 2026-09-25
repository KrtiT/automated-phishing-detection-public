"""Complete invented shift rows with matching saved offline probabilities and NLLs."""

from dataclasses import replace

from operational_cell_acceptance_fixtures import cell_case
from shift_run_codec_fixtures import _complete_case, outcome, trace_row

from automated_phishing_detection.primary_scores import PrimaryURLScores
from automated_phishing_detection.selective_inference import InferenceCounts
from automated_phishing_detection.shift_run_codec import encode_shift_run
from automated_phishing_detection.shift_schema import ShiftPlan


def scored_source(source):
    primary = PrimaryURLScores(
        (),
        0.2,
        0.2,
        0.01,
        0.2,
        0,
        0,
        0,
        0,
        False,
        0.2,
        -100.0,
        "{}",
        "{}",
        InferenceCounts(1, 1, 1, 0),
    )
    rows = tuple(replace(row, primary=primary) for row in source.external.snapshot.rows)
    external = replace(
        source.external, snapshot=replace(source.external.snapshot, rows=rows)
    )
    return type(source)(**(vars(source) | {"external": external}))


def shift_case(api, source):
    case = cell_case(api, scored_source(source), ordinal=121)
    plan = ShiftPlan(case.inputs.manifest_sha256, 1, case.inputs.requests)
    warmup = tuple(outcome(plan, "warmup", position) for position in range(1000))
    measured = tuple(
        outcome(plan, "measured", position) for position in range(len(plan.requests))
    )
    run, first, second = _complete_case(plan, warmup, measured)
    rows = []
    for position in range(len(plan.requests)):
        row = trace_row(plan, position)
        window = row.window
        if window is not None:
            window = window.model_copy(update={"score": -100.0})
        rows.append(row.model_copy(update={"monitor_nll": -100.0, "window": window}))
    run = replace(run, trace=run.trace.model_copy(update={"rows": rows}))
    case.payloads.update(
        {
            "run.json": encode_shift_run(run),
            "warmup.json": first,
            "measured.json": second,
        }
    )
    return case
