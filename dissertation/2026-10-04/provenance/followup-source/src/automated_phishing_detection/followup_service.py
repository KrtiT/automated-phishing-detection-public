"""Bounded service comparison; never replacement evidence for original H3."""

from contextlib import contextmanager

import numpy as np
import threadpoolctl

from . import fixed_cascade, http_replay
from .selective_inference import InferenceCounts, RequestScores

PROTOCOL = "bounded-followup-service-v1"
WORKLOADS = ("no_model", "structural_detector")


def schedule():
    return [
        {
            "workload": workload,
            "concurrency": concurrency,
            "pair": pair,
            "client": client,
        }
        for workload in WORKLOADS
        for concurrency in (1, 64)
        for pair in range(1, 11)
        for client in (("shared", "worker") if pair % 2 else ("worker", "shared"))
    ]


def synthetic_requests():
    return tuple(
        http_replay.ReplayRequest(
            f"synthetic-{index:05d}",
            f"{'https' if index % 2 else 'http'}://host{index % 257}.example/path/{index}?token={index % 97}",
        )
        for index in range(10000)
    )


class StructuralScorer:
    def __init__(self, model):
        self.model = model
        self.completed = 0
        self.failed = 0

    @property
    def counts(self):
        return InferenceCounts(0, 0, self.completed, self.failed)

    def scan(self, raw_url, *, drift_override=False):
        try:
            if drift_override:
                raise ValueError("follow-up has no drift override")
            if self.model is None:
                probability, decision, audit = 0.25, 0, {}
            else:
                scores, audit = fixed_cascade.score_logistic_l1_authoritative(
                    self.model, [raw_url]
                )
                probability = scores[0]
                decision = int(
                    probability >= self.model.validation_threshold_record["threshold"]
                )
            result = RequestScores(
                probability, None, decision, decision, False, False, False, False, audit
            )
            self.completed += 1
            return result
        except BaseException:
            self.failed += 1
            raise


@contextmanager
def scorer_session(workload, baseline_path):
    if workload not in WORKLOADS:
        raise ValueError("unknown follow-up workload")
    with threadpoolctl.threadpool_limits(limits=1):
        if any(pool["num_threads"] != 1 for pool in threadpoolctl.threadpool_info()):
            raise ValueError("one numerical thread not established")
        model = (
            fixed_cascade.load_logistic_l1_artifact(baseline_path)
            if workload == "structural_detector"
            else None
        )
        yield StructuralScorer(model)


def latency_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("invalid latency observations")
    quantiles = (
        np.quantile(values, [0.5, 0.95, 0.99], method="linear").tolist()
        if values.size
        else [None] * 3
    )
    return dict(
        zip(
            ("count", "p50_ms", "p95_ms", "p99_ms"),
            (len(values), *quantiles),
            strict=True,
        )
    )


def describe_run(run):
    summary = http_replay.summarize_run(run)
    for name in ("prevalence_basis_points", "workload", "run_index"):
        summary.pop(name)
    for name, rows in (
        (
            "success_latency",
            [row.elapsed_ms for row in run.measured if row.error is None],
        ),
        (
            "failure_latency",
            [row.elapsed_ms for row in run.measured if row.error is not None],
        ),
        ("all_latency", [row.elapsed_ms for row in run.measured]),
    ):
        summary[name] = latency_summary(rows)
    summary["error_types"] = {
        kind: sum(row.error == kind for row in run.measured)
        for kind in sorted(http_replay.ERRORS)
    }
    return summary


def response_agreement(shared, worker):
    if tuple(row.record_id for row in shared.measured) != tuple(
        row.record_id for row in worker.measured
    ):
        raise ValueError("paired record order changed")
    both, exact, predictions = 0, 0, 0
    for left, right in zip(shared.measured, worker.measured, strict=True):
        if left.error is not None or right.error is not None:
            continue
        both += 1
        left_response = left.response.model_dump(exclude={"request_id"})
        right_response = right.response.model_dump(exclude={"request_id"})
        exact += left_response == right_response
        left_response.pop("admission_sequence")
        right_response.pop("admission_sequence")
        predictions += left_response == right_response
    return {
        "requested": len(shared.measured),
        "both_successful": both,
        "exact_except_request_id": exact,
        "prediction_agreement": predictions,
        "noncomparable_errors": len(shared.measured) - both,
    }


def primary_summary(pairs, successful_latencies, *, errors, exact_agreement):
    if len(pairs) != 10 or {row["pair"] for row in pairs} != set(range(1, 11)):
        raise ValueError("all ten unique prespecified pairs are required")
    if (
        type(errors) is not int
        or not 0 <= errors <= 100000
        or len(successful_latencies) + errors != 100000
    ):
        raise ValueError("all 100,000 measured worker requests are required")
    if type(exact_agreement) is not int or not 0 <= exact_agreement <= 100000:
        raise ValueError("invalid response agreement count")
    ordered = sorted(pairs, key=lambda row: row["pair"])
    ratios, undefined = [], []
    for row in ordered:
        shared, worker = row["shared_p95_ms"], row["worker_p95_ms"]
        if shared is None or worker is None or shared == 0:
            ratios.append(None)
            undefined.append(row["pair"])
        elif (
            not np.isfinite(shared)
            or not np.isfinite(worker)
            or shared < 0
            or worker < 0
        ):
            raise ValueError("invalid paired p95")
        else:
            ratios.append(worker / shared)
    median, interval = None, None
    if not undefined:
        random = np.random.Generator(np.random.PCG64(20261002))
        values = np.asarray(ratios)
        median = float(np.median(values))
        bootstrap = np.median(values[random.integers(0, 10, size=(10000, 10))], axis=1)
        interval = np.quantile(bootstrap, [0.0125, 0.9875], method="linear").tolist()
    latencies = latency_summary(successful_latencies)
    requirements = {
        "median_ratio_at_most_point_eight": median is not None and median <= 0.8,
        "ratio_interval_upper_below_one": interval is not None and interval[1] < 1,
        "pooled_worker_success_p95_at_most_200ms": latencies["p95_ms"] is not None
        and latencies["p95_ms"] <= 200,
        "errors_below_one_per_thousand": errors < 100,
        "exact_paired_response_agreement": exact_agreement == 100000,
    }
    return {
        "ratios": ratios,
        "median_ratio": median,
        "ratio_interval_97_5": interval,
        "undefined_pairs": undefined,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261002,
        "worker_success_latency": latencies,
        "worker_errors": errors,
        "worker_error_rate": errors / 100000,
        "request_denominator": 100000,
        "exact_paired_agreement": exact_agreement,
        "requirements": requirements,
        "S_requirement_met": all(requirements.values()),
    }
