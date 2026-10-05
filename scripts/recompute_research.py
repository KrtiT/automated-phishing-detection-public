"""Recompute saved research results; never fit models, predict, or contact URL hosts."""

import argparse
import hashlib
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "dissertation/2026-10-04"
SCRIPTS = PACKAGE / "analysis-scripts/dissertation"
ORIGIN = "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4"
SERIES = "gwu_working/study-series-v1-20261001-segment-2"
FOLLOWUP = "dissertation/followup-20261001"
CATALOG = ROOT / "research-archive/2026-10-04/catalog.json"


class ArchiveInputs:
    def __init__(self, root, catalog_path=CATALOG):
        self.root = Path(root).resolve()
        self.entries = {}
        self.checked = set()
        catalog_bytes = Path(catalog_path).read_bytes()
        families = json.loads(catalog_bytes)["families"]
        self.authentication = {
            "catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
            "inventory_sha256": {},
        }
        directory = self.root / ".archive-inventories"
        expected = {name + ".jsonl": record for name, record in families.items()}
        if (
            not directory.is_dir()
            or directory.is_symlink()
            or {path.name for path in directory.iterdir()} != set(expected)
        ):
            raise ValueError("Inventory set differs from the committed catalog")
        for name, record in expected.items():
            inventory = directory / name
            if inventory.is_symlink():
                raise ValueError("Inventory cannot be a symlink")
            payload = inventory.read_bytes()
            identity = hashlib.sha256(payload).hexdigest()
            if identity != record["inventory_sha256"]:
                raise ValueError("Inventory hash differs from the committed catalog")
            self.authentication["inventory_sha256"][name] = identity
            for line in payload.splitlines():
                entry = json.loads(line)
                name = entry["path"]
                path = PurePosixPath(name)
                if path.is_absolute() or ".." in path.parts or name in self.entries:
                    raise ValueError("Invalid or duplicate inventoried path")
                self.entries[name] = entry
        if not self.entries:
            raise ValueError(
                "No archive inventory; use research_archive.py verify --materialize first"
            )

    def bytes(self, name, exact=True):
        if name not in self.entries:
            raise ValueError("Input is not inventoried: " + name)
        entry = self.entries[name]
        if entry["status"] == "hash_only" or (exact and entry["status"] != "exact"):
            raise ValueError("Scientific input must be exact: " + name)
        path = self.root / name
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Input escapes archive root")
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry["public_sha256"]:
            raise ValueError("Input hash differs: " + name)
        self.checked.add(name)
        return payload

    def read(self, name, exact=True):
        return json.loads(self.bytes(name, exact=exact))

    def rows(self, name):
        return [json.loads(line) for line in self.bytes(name).splitlines()]


class RetainedPath:
    def __init__(self, inputs, name):
        self.inputs, self.name = inputs, name

    def __truediv__(self, component):
        return RetainedPath(self.inputs, self.name + "/" + component)

    def __fspath__(self):
        return str(self.inputs.root / self.name)

    def open(self):
        return io.StringIO(self.inputs.bytes(self.name).decode("utf-8"))


def require_secondary_coverage(rows, bins, names):
    if rows != 153 or bins != 430 or sorted(names) != ["gmm", "mmd", "psi"]:
        raise ValueError("Secondary coverage differs from the published study")


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_operations(inputs, verifier):
    import numpy as np

    operational = inputs.read(
        SERIES + "/series-attempt/evidence/operational-summary.json"
    )
    verifier.require(len(operational["groups"]) == 25, "25 operational groups")
    covered, live, total_errors, total_requests = [], {}, 0, 0
    for group in operational["groups"]:
        verifier.require(group["run_indices"] == [1, 2, 3, 4, 5], "five repeats")
        latencies, counters, errors = [], Counter(), 0
        for summary in group["run_summaries"]:
            ordinal = summary["cell_ordinal"]
            covered.append(ordinal)
            location = f"{ORIGIN}/cells" if ordinal <= 72 else f"{SERIES}/cells-dir"
            saved = inputs.read(
                f"{location}/cell-{ordinal:03d}-attempt/evidence/run.json"
            )
            run = saved.get("run", saved)
            count = 8701 if ordinal >= 121 else 10000
            verifier.require(
                len(run["measured"]) == count and len(run["warmup"]) == 1000,
                "cell coverage",
            )
            verifier.require(
                len({row["request_id"] for row in run["measured"]}) == count,
                "unique request IDs",
            )
            latencies.extend(row["elapsed_ms"] for row in run["measured"])
            run_errors = sum(
                row["error"] is not None or row["status_code"] != 200
                for row in run["measured"]
            )
            verifier.require(run_errors == summary["request_errors"], "cell errors")
            errors += run_errors
            for name in (
                "admitted_requests",
                "completed_requests",
                "failed_requests",
                "successful_transformer_scores",
                "transformer_forward_attempts",
            ):
                delta = run["after_measured"][name] - run["after_warmup"][name]
                verifier.require(delta == summary[name], "physical counter")
                counters[name] += delta
            verifier.close(
                count * 1000 / run["measured_elapsed_ms"],
                summary["client_attempts_per_second"],
                "throughput",
            )
            verifier.close(
                (count - run_errors) * 1000 / run["measured_elapsed_ms"],
                summary["successful_responses_per_second"],
                "successful throughput",
            )
            if ordinal >= 121:
                live[ordinal] = run
        verifier.require(
            len(latencies) == group["request_count"]
            and errors == group["request_errors"],
            "pooled population",
        )
        for name, value in counters.items():
            verifier.require(value == group[name], "pooled counters")
        for name, probability in (("p50_ms", 0.5), ("p95_ms", 0.95), ("p99_ms", 0.99)):
            verifier.close(
                float(np.quantile(latencies, probability, method="linear")),
                group[name],
                name,
            )
        verifier.close(
            errors / len(latencies), group["request_error_rate"], "error rate"
        )
        verifier.close(
            counters["transformer_forward_attempts"] / len(latencies),
            group["physical_invocation_fraction"],
            "invocation fraction",
        )
        total_errors += errors
        total_requests += len(latencies)
    verifier.require(sorted(covered) == list(range(1, 126)), "exactly 125 unique cells")
    return live, {
        "cells": len(covered),
        "groups": 25,
        "requests": total_requests,
        "errors": total_errors,
    }


def original_science(inputs, verifier, live):
    verifier.CONTEXT = inputs.root
    verifier.ORIGIN = RetainedPath(inputs, ORIGIN)
    verifier.RUN = inputs.root / SERIES
    verifier.read = lambda path: inputs.read(
        Path(path).relative_to(inputs.root).as_posix(),
        exact=Path(path).name != "source-results.json",
    )
    evidence, gates = verifier.verify_science(live)
    service = load("verify_followup_service_20261002")
    service.same(
        evidence,
        json.loads((PACKAGE / "aggregate-data/primary-results.json").read_bytes()),
    )
    secondary = load("export_final_secondary_20261001")
    published_secondary = json.loads(
        (PACKAGE / "aggregate-data/complete-secondary-results.json").read_bytes()
    )
    columns, calibration_bins = 0, 0
    for source in ("internal", "external"):
        saved = inputs.read(f"{ORIGIN}/{source}/evidence/secondary.json")
        service.same(saved, published_secondary[source])
        outcomes = secondary.load_outcomes(source)
        populations = (
            {"internal": saved["metrics"]}
            if source == "internal"
            else {
                name: value["detectors"] for name, value in saved["populations"].items()
            }
        )
        for population, detectors in populations.items():
            rows = secondary.select_population(outcomes, population)
            for name, metric in detectors.items():
                secondary.verify_metric(rows, name, metric)
                columns += 1
                calibration_bins += len(metric.get("calibration_bins", []))
    monitors = inputs.read(ORIGIN + "/external/evidence/monitors.json")
    service.same(monitors, published_secondary["monitors"])
    require_secondary_coverage(
        columns, calibration_bins, [monitor["name"] for monitor in monitors]
    )
    for monitor in monitors:
        verifier.require(len(monitor["windows"]) == 132, "monitor coverage")
        for window in monitor["windows"]:
            verifier.require(
                window["alert"] == (window["score"] > monitor["threshold"]),
                "monitor threshold",
            )
    return {
        "primary_checks": len(gates),
        "decisions": dict(Counter(gate["status"] for gate in gates)),
        "hypotheses": {
            key: value["decision"]
            for key, value in evidence["primary"]["hypotheses"].items()
        },
        "secondary_population_detector_rows": columns,
        "calibration_bins": calibration_bins,
        "monitor_windows": 132 * len(monitors),
    }


def detection(inputs):
    verifier = load("verify_followup_detection_20261001")
    rows = inputs.rows(FOLLOWUP + "/detection-evaluate-v1/predictions.jsonl")
    population = inputs.rows(
        FOLLOWUP + "/population-admission-v1-attempt-2/retained.jsonl"
    )
    frozen = inputs.read(FOLLOWUP + "/detection-fit-v1/frozen-models.json")
    saved = inputs.read(FOLLOWUP + "/detection-evaluate-v1/completion.json")
    if len(rows) != len(population) or len(rows) != 8622:
        raise ValueError("D population coverage")
    for row, reference in zip(rows, population, strict=True):
        if any(
            row[key] != reference[key]
            for key in ("record_id", "registrable_domain", "is_phishing")
        ):
            raise ValueError("D row identity, label or order")
    actual = verifier.recompute(
        rows, {name: frozen[name + "_threshold"] for name in ("baseline", "candidate")}
    )
    for key, value in actual.items():
        verifier.compare(saved[key], value, key)
    return {
        "records": len(rows),
        "D_requirement_met": actual["D_requirement_met"],
        "scheme_invariance": actual["scheme_invariance"],
    }


def service(inputs):
    verifier = load("verify_followup_service_20261002")
    prefix = FOLLOWUP + "/service-comparison-v2"
    completed = inputs.read(prefix + "/completion.json")
    manifest_hash = hashlib.sha256(
        inputs.bytes(prefix + "/synthetic-manifest.json")
    ).hexdigest()
    verifier.require(
        completed["arm_count"] == 80 and not completed["interrupted_schedule_pooled"],
        "S schedule",
    )
    ratios, latencies, errors, exact, predictions = [], [], 0, 0, 0
    all_errors, arm_count, pair_count = 0, 0, 0
    for workload in ("no_model", "structural_detector"):
        for concurrency in (1, 64):
            for pair in range(1, 11):
                runs, summaries = {}, {}
                for client in (
                    ("shared", "worker") if pair % 2 else ("worker", "shared")
                ):
                    arm = {
                        "workload": workload,
                        "concurrency": concurrency,
                        "pair": pair,
                        "client": client,
                    }
                    name = f"{prefix}/{workload}-c{concurrency}-pair{pair:02d}-{client}"
                    record = inputs.read(name + "/run.json")
                    verifier.same(arm, record["arm"])
                    run = record["legacy_transport_record"]["run"]
                    summary = verifier.summarize(run, manifest_hash)
                    verifier.same(
                        {"summary": summary, **arm}, completed["arms"][arm_count]
                    )
                    all_errors += summary["request_errors"]
                    arm_count += 1
                    runs[client] = run["measured"]
                    summaries[client] = summary
                agreement = verifier.agreement(runs["shared"], runs["worker"])
                verifier.same(
                    {
                        "workload": workload,
                        "concurrency": concurrency,
                        "pair": pair,
                        "response_agreement": agreement,
                    },
                    completed["pairs"][pair_count],
                )
                pair_count += 1
                if workload == "structural_detector" and concurrency == 64:
                    ratios.append(
                        summaries["worker"]["success_latency"]["p95_ms"]
                        / summaries["shared"]["success_latency"]["p95_ms"]
                    )
                    latencies.extend(
                        row["elapsed_ms"]
                        for row in runs["worker"]
                        if row["error"] is None
                    )
                    errors += summaries["worker"]["request_errors"]
                    exact += agreement["exact_except_request_id"]
                    predictions += agreement["prediction_agreement"]
    primary = verifier.primary(ratios, latencies, errors=errors, exact=exact)
    verifier.same(primary, completed["primary"])
    return {
        "arms": arm_count,
        "pairs": pair_count,
        "requests": arm_count * 10000,
        "errors": all_errors,
        "primary_prediction_agreement": predictions,
        "primary": primary,
        "interrupted_schedule_pooled": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise ValueError("Output exists; do not overwrite evidence")
    inputs = ArchiveInputs(arguments.archive_root)
    verifier = load("verify_final_evidence_20261001")
    live, operations = original_operations(inputs, verifier)
    report = {
        "scope": "Read-only arithmetic from public retained records; no model execution, new timing, execution authorization revalidation or institutional approval."
    }
    report["original_operational"] = operations
    report["original_science"] = original_science(inputs, verifier, live)
    del live
    report["D"] = detection(inputs)
    report["S"] = service(inputs)
    report["authenticated_input_files"] = len(inputs.checked)
    report["authentication"] = inputs.authentication
    report["status"] = "verified"
    with arguments.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
