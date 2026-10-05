"""Read-only verification of retained evidence; no inference, fitting or source access."""

import base64
import csv
import hashlib
import json
import math
import operator
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

import numpy as np
from scipy.stats import beta

CONTEXT = Path(__file__).resolve().parent.parent
RUN = CONTEXT / "gwu_working/study-series-v1-20261001-segment-2"
ORIGIN = CONTEXT / "gwu_working/study-urlnorm-v1-2026-09-30-attempt-4"
HISTORY = CONTEXT / "study-series-history-package-2026-10-01T035808.737927_0000"
OUTPUT = CONTEXT / "dissertation/final-evidence-20261001"
HASHES = {}
CHECKS = Counter()


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    value = hasher.hexdigest()
    HASHES[str(path.relative_to(CONTEXT))] = value
    return value


def require(condition, message):
    if not condition:
        raise ValueError(message)
    CHECKS["assertions"] += 1


def close(actual, expected, name):
    require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), name)


def check_hash(path, expected):
    require(digest(path) == expected, f"hash mismatch: {path}")
    CHECKS["file_hash_comparisons"] += 1


def authenticate_tree(value):
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            check_hash(value["path"], value["sha256"])
        for child in value.values():
            authenticate_tree(child)
    elif isinstance(value, list):
        for child in value:
            authenticate_tree(child)


def receipt(attempt, summary_path):
    outcome = read(attempt / "outcome.json")
    require(outcome["status"] == "completion_prepared", str(attempt))
    check_hash(attempt / "reservation.json", outcome["reservation_sha256"])
    check_hash(summary_path, outcome["public_summary_sha256"])
    summary = read(summary_path)
    require(summary["private_sha256"] == outcome["private_sha256"], "receipt map")
    for name, expected in outcome["private_sha256"].items():
        check_hash(attempt / "evidence" / name, expected)
    return summary


def process_pair(pair):
    require(pair["status"] == "observed" and pair["failure"] is None, "pair status")
    require(not pair["record_failures"], "pair retention failures")
    for role in ("client", "service"):
        process = pair[role]
        require(process["exit_observed"] and process["exit_code"] == 0, "owned exit")
        require(not process["forced"] and not process["signals"], "forced shutdown")
        CHECKS["successful_owned_operational_exits"] += 1


def verify_authorization():
    profile_path = CONTEXT / "study-series-profile-20261001-segment-2.json"
    envelope_path = CONTEXT / "study-series-envelope-20261001-segment-2.json"
    check_hash(profile_path, "92f6c954f5cb1c7d0a699a9e0965c3aefa98435fcf64ad7800f84396bac24714")
    check_hash(envelope_path, "11cf63c6a01a6661964bd1fad17e49a9d8ee8d67e7752a0ff560532badf890c7")
    profile, envelope = read(profile_path), read(envelope_path)
    require(envelope["profile"] == profile and not envelope["revoked"], "authority")
    for decision in envelope["decisions"].values():
        require(decision["status"] == "authorized_by_explicit_series_directive", "decision")
    for field in ("index", "eligible_prefix_review", "exposure_record"):
        check_hash(profile["history"][field + "_path"], profile["history"][field + "_sha256"])
    for mapping in profile["transition"].values():
        for name, expected in mapping.items():
            check_hash(Path(profile["paths"]["repo_root"]) / name, expected)
    index = read(HISTORY / "index.json")
    authenticate_tree(index)
    require(index["selected_attempt_ordinal"] == 4, "selected historical attempt")
    require([cell["ordinal"] for cell in index["accepted_cells"]] == list(range(1, 73)), "prefix")
    eligibility = read(HISTORY / "eligibility.json")
    require(eligibility["accepted_ordinals"] == list(range(1, 73)), "eligibility ordinals")
    return profile, index


def verify_conditions():
    records = RUN / "physical-records-dir"
    post = read(records / "post.json")
    require(post["root_exit_code"] == 0 and post["session_violation"] is None, "root exit")
    require(post["supervisor_error_type"] is None and not post["root_shutdown_escalated"], "supervisor")
    require((records / "stderr.log").stat().st_size == 0, "root stderr")
    pre = read(records / "pre.json")
    sample_count = 0
    with (records / "conditions.jsonl").open() as source:
        for sample in chain([pre], (json.loads(line) for line in source if line.strip()), [post]):
            require("Now drawing from 'AC Power'" in sample["battery"], "AC observation")
            require(not sample["competing_known_workload_pids"], "known competing workload")
            require("No thermal warning level" in sample["thermal"], "thermal observation")
            require("No performance warning level" in sample["thermal"], "performance observation")
            require("PreventSystemSleep             1" in sample["assertions"], "sleep inhibition")
            sample_count += 1
    for path in records.iterdir():
        if path.is_file():
            digest(path)
    return {"samples": sample_count, "started_at": pre["observed_at"],
            "ended_at": post["ended_at"], "root_exit_code": 0,
            "limitation": "Sampled known-command checks are not proof of absence of every possible workload."}


def verify_cells(index):
    root = receipt(RUN / "series-attempt", RUN / "series-public-summary")
    receipt(RUN / "segment-attempt", RUN / "segment-public-summary")
    segment = read(RUN / "segment-attempt/segment-accounting.json")
    series = read(RUN / "series-attempt/series-accounting.json")
    require(segment["status"] == series["status"] == "complete", "accounting complete")
    ledger = segment["ledger"]
    require([cell["ordinal"] for cell in ledger["cells"]] == list(range(73, 126)), "fresh ordinals")
    require(len(ledger["admissions"]) == 106, "admission coverage")
    for admission in ledger["admissions"]:
        require(admission["accepted"] and admission["issued"] and admission["observation_recorded"], "admission")
        require(admission["exit_observed"] and admission["exit_code"] == 0, "admission exit")
        require(hashlib.sha256(base64.b64decode(admission["frame_bytes"], validate=True)).hexdigest() == admission["frame_sha256"], "frame")
    paths = {}
    for cell in index["accepted_cells"]:
        paths[cell["ordinal"]] = Path(cell["payloads"]["public-summary.json"]["path"])
    for cell in ledger["cells"]:
        ordinal = cell["ordinal"]
        prefix = RUN / f"cells-dir/cell-{ordinal:03d}"
        require(cell["status"] == "accepted" and cell["holders_closed"], "accepted ledger cell")
        for name, expected in cell["snapshot_sha256"].items():
            path = Path(str(prefix) + "-summary.json") if name == "public-summary.json" else Path(str(prefix) + "-" + name)
            check_hash(path, expected)
        paths[ordinal] = Path(str(prefix) + "-summary.json")
        observation = base64.b64decode(cell["observation_bytes"], validate=True)
        require(observation == Path(str(prefix) + "-attempt/evidence/process-pair.json").read_bytes(), "ledger observation")
    runs = {}
    for ordinal, path in sorted(paths.items()):
        attempt = Path(str(path).replace("-summary.json", "-attempt"))
        summary = receipt(attempt, path)
        require(summary["status"] == "operational_evidence_published", "cell publication")
        require(summary["cell"]["ordinal"] == ordinal, "cell identity")
        process_pair(read(attempt / "evidence/process-pair.json"))
        saved = read(attempt / "evidence/run.json")
        run = saved.get("run", saved)
        require(len(run["warmup"]) == 1000, "warmup count")
        count = 8701 if ordinal >= 121 else 10000
        require(len(run["measured"]) == count, "measured count")
        require(len({request["request_id"] for request in run["measured"]}) == count, "unique requests")
        runs[ordinal] = run
    operational = read(RUN / "series-attempt/evidence/operational-summary.json")
    require(root["operational"] == operational, "root operational join")
    require(len(operational["groups"]) == 25, "group count")
    covered = []
    for group in operational["groups"]:
        require(group["run_indices"] == [1, 2, 3, 4, 5], "all repeats")
        latencies, errors, counters = [], 0, Counter()
        for summary in group["run_summaries"]:
            ordinal = summary["cell_ordinal"]
            covered.append(ordinal)
            run = runs[ordinal]
            latencies.extend(request["elapsed_ms"] for request in run["measured"])
            run_errors = sum(request["error"] is not None or request["status_code"] != 200 for request in run["measured"])
            require(run_errors == summary["request_errors"], "run errors")
            errors += run_errors
            for name in ("admitted_requests", "completed_requests", "failed_requests", "successful_transformer_scores", "transformer_forward_attempts"):
                observed = run["after_measured"][name] - run["after_warmup"][name]
                require(observed == summary[name], f"measured counter {name}")
                counters[name] += observed
            close(len(run["measured"]) * 1000 / run["measured_elapsed_ms"], summary["client_attempts_per_second"], "throughput")
            close((len(run["measured"]) - run_errors) * 1000 / run["measured_elapsed_ms"], summary["successful_responses_per_second"], "success throughput")
        require(len(latencies) == group["request_count"] and errors == group["request_errors"], "group count/error")
        for name, value in counters.items():
            require(value == group[name], f"pooled counter {name}")
        for name, quantile in (("p50_ms", .5), ("p95_ms", .95), ("p99_ms", .99)):
            close(float(np.quantile(latencies, quantile, method="linear")), group[name], name)
        close(errors / len(latencies), group["request_error_rate"], "group error rate")
        close(counters["transformer_forward_attempts"] / len(latencies), group["physical_invocation_fraction"], "physical fraction")
    require(sorted(covered) == list(range(1, 126)), "125 unique ordinals")
    return operational, runs


def verify_science(runs):
    evidence = read(RUN / "series-attempt/evidence/study-evidence.json")
    primary = evidence["primary"]
    metadata = read(ORIGIN / "study-root/source-results.json")["accepted_inputs"]
    for source in ("internal", "external"):
        observation = metadata[source]["worker"]["exit"]
        require(observation["exit_observed"] and observation["exit_code"] == 0, "source exit")
    populations = defaultdict(list)
    for source in ("internal", "external"):
        with (ORIGIN / source / "evidence/predictions.jsonl").open() as saved:
            for line in saved:
                row = json.loads(line)
                record, scores = row["record"], row.get("primary", row)
                outcome = {"domain": record["registrable_domain"], "label": record["is_phishing"],
                           "length_only": scores["length_decision"], "logistic_l1": scores["stage1_decision"],
                           "cascade": scores["cascade_decision"], "transformer": scores["transformer_decision"]}
                if source == "external":
                    outcome["policy"] = row["policy_decision"]
                populations["internal" if source == "internal" else record["role"]].append(outcome)
    for name, metric in primary["metrics"].items():
        population, detector = name.split(".")
        rows = populations[population]
        if population == "tranco":
            close(sum(row[detector] for row in rows) / len(rows), metric["estimate"], name)
            continue
        positives = [row for row in rows if row["label"] == 1]
        negatives = [row for row in rows if row["label"] == 0]
        true_positive = sum(row[detector] for row in positives)
        false_positive = sum(row[detector] for row in negatives)
        for field, count in (("true_positives", true_positive), ("false_positives", false_positive),
                             ("false_negatives", len(positives) - true_positive), ("true_negatives", len(negatives) - false_positive)):
            require(metric[field] == count, name + field)
        for field, numerator, denominator in (("recall", true_positive, len(positives)), ("fpr", false_positive, len(negatives))):
            if denominator:
                close(metric[field]["estimate"], numerator / denominator, name + field)
                upper = 1.0 if numerator == denominator else float(beta.ppf(.95, numerator + 1, denominator - numerator))
                close(metric[field]["upper_95"], upper, name + " CP bound")
    for name, contrast in primary["contrasts"].items():
        population, comparison = name.split(".")
        candidate, reference = comparison.split("_minus_")
        rows = [row for row in populations[population] if row["label"] == 1]
        grouped = defaultdict(lambda: [0, 0])
        for row in rows:
            grouped[row["domain"]][0] += int(row[candidate]) - int(row[reference])
            grouped[row["domain"]][1] += 1
        require(len(rows) == contrast["positive_count"] and len(grouped) == contrast["domain_count"], "paired population")
        require(sum(row[candidate] for row in rows) == contrast["candidate_true_positives"], "candidate TP")
        require(sum(row[reference] for row in rows) == contrast["reference_true_positives"], "reference TP")
        clusters = np.array([grouped[domain] for domain in sorted(grouped)], dtype=np.int64)
        rng = np.random.Generator(np.random.PCG64(20260816))
        estimates = []
        for replicate in range(2000):
            sampled = clusters[rng.integers(0, len(clusters), size=len(clusters), dtype=np.int64)].sum(axis=0)
            estimates.append(sampled[0] / sampled[1])
        lower, upper = np.quantile(estimates, [.025, .975], method="linear")
        close(lower, contrast["lower"], name + " lower")
        close(upper, contrast["upper"], name + " upper")
        close(clusters[:, 0].sum() / len(rows), contrast["estimate"], name + " difference")
        require(contrast["bootstrap_replicates"] == 2000, "replicates")
    gates = []
    operations = {"<=": operator.le, "<": operator.lt, ">": operator.gt, ">=": operator.ge}
    for hypothesis, result in primary["hypotheses"].items():
        require(result["complete"], "complete hypothesis")
        for gate in result["gates"]:
            passed = operations[gate["operator"]](gate["estimate"], gate["threshold"])
            require(gate["status"] == ("pass" if passed else "fail"), "gate decision")
            if gate["denominator"]:
                close(gate["estimate"], gate["numerator"] / gate["denominator"], "gate ratio")
            if gate["name"] in primary["contrasts"]:
                close(gate["estimate"], primary["contrasts"][gate["name"]]["lower"], "lower-bound gate")
            gates.append({"hypothesis": hypothesis, **gate})
        require(result["decision"] == "not_supported" and any(gate["status"] == "fail" for gate in result["gates"]), "hypothesis conclusion")
    require(Counter(gate["hypothesis"] for gate in gates) == {"H1": 10, "H2": 4, "H3": 8}, "22 gates")
    routing = read(ORIGIN / "external/evidence/routing.json")
    require(len(routing["rows"]) == 8701 and len(routing["windows"]) == 132, "stream coverage")
    require(sum(window["alert"] for window in routing["windows"]) == 115, "external alerts")
    for ordinal in range(121, 126):
        trace = runs[ordinal]["trace"]
        require(trace["complete"] and not trace["broken"], "live complete")
        require(len(trace["rows"]) == len(routing["rows"]), "trace size")
        for actual, expected in zip(trace["rows"], routing["rows"], strict=True):
            for actual_key, expected_key in (("band_selected", "logical_band"), ("fixed_decision", "fixed_decision"),
                                             ("drift_override", "drift_override"), ("decision", "policy_decision"),
                                             ("stage2_invoked", "logical_stage2_mask")):
                require(actual[actual_key] == expected[expected_key], "live/offline trace agreement")
    return evidence, gates


def write_csv(name, rows):
    columns = list(dict.fromkeys(field for row in rows for field in row))
    with (OUTPUT / name).open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    profile, index = verify_authorization()
    print("Authorization, frozen code and historical file identities verified.", flush=True)
    conditions = verify_conditions()
    operational, runs = verify_cells(index)
    print("All 125 cells, 250 owned exits and 25 pooled groups verified.", flush=True)
    evidence, gates = verify_science(runs)
    OUTPUT.mkdir(exist_ok=True)
    write_csv("primary-gates.csv", gates)
    write_csv("paired-contrasts.csv", [{"contrast": name, **value} for name, value in evidence["primary"]["contrasts"].items()])
    write_csv("operational-groups.csv", [{name: value for name, value in group.items() if name != "run_summaries"} for group in operational["groups"]])
    write_csv("operational-runs.csv", [run for group in operational["groups"] for run in group["run_summaries"]])
    (OUTPUT / "primary-results.json").write_text(json.dumps(evidence, indent=2) + "\n")
    report = {"status": "verified", "verified_at": datetime.now(timezone.utc).isoformat(),
              "scope": "Independent read-only hashes, receipts, exits, recorded conditions, arithmetic, pooled terminal latencies, saved-outcome bootstrap and live/offline traces; no model execution.",
              "measurement_revision": profile["execution"]["revision"], "checks": dict(CHECKS),
              "retained_cells": 72, "new_cells": 53, "operational_cells": 125, "operational_groups": 25,
              "hypothesis_gates": 22, "conditions": conditions, "file_sha256": HASHES}
    report["verifier_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (OUTPUT / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "file_sha256"}, indent=2))


if __name__ == "__main__":
    main()
