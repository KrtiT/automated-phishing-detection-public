"""Bounded scoring paths, owned by the post-review admitted parent."""

import math
import os
import time
from statistics import median

import numpy as np
from revision_analysis import routing_decisions
from revision_population import _bytes, _write
from routing_replay import THRESHOLD, gate_mask

PATHS = (
    "structural",
    "transformer",
    "original",
    "symmetric",
    "directional",
    "symmetric_control",
)


class RoutingEngine:
    def __init__(
        self, structural, transformer, *, transformer_threshold, forward_counts=None
    ):
        self.structural = structural
        self.transformer = transformer
        self.transformer_threshold = transformer_threshold
        self.forward_counts = forward_counts
        self.counts = dict.fromkeys(
            (
                "stage1_started",
                "stage1_completed",
                "stage2_started",
                "stage2_completed",
                "requests_completed",
                "requests_failed",
            ),
            0,
        )

    def _score(self, raw, stage, function):
        self.counts[stage + "_started"] += 1
        probability = function(raw)
        if (
            type(probability) not in (float, int)
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("invalid_stage_probability")
        self.counts[stage + "_completed"] += 1
        return probability

    def scan(self, raw, path, *, control_selected=False):
        if path not in PATHS or type(control_selected) is not bool:
            raise ValueError("unknown_scoring_path")
        structural = expert = None
        selected = path == "transformer"
        try:
            if path != "transformer":
                structural = self._score(raw, "stage1", self.structural)
                if path in {"original", "symmetric", "directional"}:
                    selected = gate_mask(
                        [structural],
                        "below" if path == "directional" else "symmetric",
                        0 if path == "original" else 0.20,
                    )[0]
                elif path == "symmetric_control":
                    selected = control_selected
            if selected:
                expert = self._score(raw, "stage2", self.transformer)
            decision = (
                int(expert >= self.transformer_threshold)
                if selected
                else int(structural >= THRESHOLD)
            )
            self.counts["requests_completed"] += 1
            return {
                "decision": decision,
                "selected": selected,
                "stage1_probability": structural,
                "transformer_probability": expert,
            }
        except BaseException:
            self.counts["requests_failed"] += 1
            raise


def _verify_result(row, result, decision, selected):
    if result["decision"] != decision or result["selected"] != selected:
        raise ValueError("timed_decision_or_membership_differs_from_R")
    for key in ("stage1_probability", "transformer_probability"):
        if result[key] is not None and result[key] != row[key]:
            raise ValueError("timed_score_differs_from_R")


def _run_arm(
    rows, engine, path, repetition, root, require_admission, decisions, mask, warmups
):
    require_admission()
    for number in range(warmups):
        index = number % len(rows)
        require_admission()
        result = engine.scan(rows[index]["raw_url"], path, control_selected=mask[index])
        _verify_result(rows[index], result, decisions[index], mask[index])
    before = dict(engine.counts)
    forwards_before = engine.forward_counts() if engine.forward_counts else None
    durations = []
    arm_started = time.perf_counter_ns()
    descriptor = os.open(
        root / f"arm-{repetition}-{path}.jsonl",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb") as stream:
        for index, row in enumerate(rows):
            require_admission()
            started = time.perf_counter_ns()
            try:
                result = engine.scan(row["raw_url"], path, control_selected=mask[index])
            except BaseException as error:
                stream.write(
                    _bytes(
                        {
                            "record_id": row["record_id"],
                            "elapsed_ns": time.perf_counter_ns() - started,
                            "error_type": type(error).__name__,
                            "counts": engine.counts,
                        }
                    )
                )
                stream.flush()
                os.fsync(stream.fileno())
                raise
            elapsed = time.perf_counter_ns() - started
            stream.write(
                _bytes({"record_id": row["record_id"], "elapsed_ns": elapsed, **result})
            )
            stream.flush()
            durations.append(elapsed)
            _verify_result(row, result, decisions[index], mask[index])
        os.fsync(stream.fileno())
    wall_elapsed = time.perf_counter_ns() - arm_started
    require_admission()
    counts = {name: value - before[name] for name, value in engine.counts.items()}
    expected_stage1 = 0 if path == "transformer" else len(rows)
    if counts != {
        "stage1_started": expected_stage1,
        "stage1_completed": expected_stage1,
        "stage2_started": sum(mask),
        "stage2_completed": sum(mask),
        "requests_completed": len(rows),
        "requests_failed": 0,
    }:
        raise ValueError("actual_stage_counts_differ")
    forward_delta = None
    if forwards_before is not None:
        forward_delta = {
            name: value - forwards_before[name]
            for name, value in engine.forward_counts().items()
        }
        if forward_delta != {"started": sum(mask), "completed": sum(mask)}:
            raise ValueError("underlying_transformer_forward_counts_differ")
    return {
        "repetition": repetition,
        "path": path,
        "counts": counts,
        "underlying_transformer_forwards": forward_delta,
        "warmup_requests_excluded": warmups,
        "summed_scoring_seconds": sum(durations) / 1e9,
        "whole_arm_seconds": wall_elapsed / 1e9,
        "latency_ms": dict(
            zip(
                ("p50", "p95", "p99"),
                (
                    np.quantile(durations, [0.5, 0.95, 0.99], method="linear") / 1e6
                ).tolist(),
                strict=True,
            )
        ),
        "implied_serial_scoring_requests_per_second": len(rows) * 1e9 / sum(durations),
    }


def run_cost(rows, engine, root, require_admission, *, warmups=100):
    require_admission()
    decisions, masks = routing_decisions(rows, engine.transformer_threshold)
    schedule = [
        {"repetition": repetition, "path": path}
        for repetition in range(5)
        for path in PATHS[repetition:] + PATHS[:repetition]
    ]
    _write(
        root / "cost-started.json",
        _bytes({"schedule": schedule, "warmups_per_arm": warmups, "rows": len(rows)}),
    )
    arms = []
    active = None
    try:
        _write(
            root / "cost-membership.json",
            _bytes(
                {
                    "record_ids": [row["record_id"] for row in rows],
                    "masks": {name: masks[name] for name in PATHS},
                }
            ),
        )
        for active in schedule:
            arm = _run_arm(
                rows,
                engine,
                active["path"],
                active["repetition"],
                root,
                require_admission,
                decisions[active["path"]],
                masks[active["path"]],
                warmups,
            )
            _write(
                root / f"arm-{active['repetition']}-{active['path']}-summary.json",
                _bytes(arm),
            )
            arms.append(arm)
        ratios = {}
        for path in PATHS:
            paired = [
                next(
                    arm["summed_scoring_seconds"]
                    for arm in arms
                    if arm["path"] == path and arm["repetition"] == repetition
                )
                / next(
                    arm["summed_scoring_seconds"]
                    for arm in arms
                    if arm["path"] == "structural" and arm["repetition"] == repetition
                )
                for repetition in range(5)
            ]
            ratios[path] = {"paired_ratios": paired, "median": median(paired)}
        summary = {
            "arms": arms,
            "elapsed_ratios_to_structural": ratios,
            "http_service_measurement": False,
            "order_design": "five_cyclic_rotations_not_fully_position_balanced",
            "run_level_confidence_interval": None,
            "requests_per_arm": len(rows),
        }
        require_admission()
        _write(root / "cost-summary.json", _bytes(summary))
        return summary
    except BaseException as error:
        _write(
            root / "cost-failure.json",
            _bytes(
                {
                    "error_type": type(error).__name__,
                    "completed_arms": len(arms),
                    "active_arm": active,
                    "unstarted_arms": schedule[len(arms) + 1 :]
                    if active is not None
                    else schedule,
                    "actual_counts_including_warmups": engine.counts,
                }
            ),
        )
        raise
