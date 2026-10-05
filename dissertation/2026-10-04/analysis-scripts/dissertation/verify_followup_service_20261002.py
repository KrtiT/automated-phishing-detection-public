"""Recompute retained service measurements without importing scientific reducers."""

import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "followup-20261001"
ERRORS = ("correlation", "http_status", "invalid_json", "invalid_schema", "timeout", "transport")
COUNTERS = ("admitted_requests", "completed_requests", "failed_requests", "successful_transformer_scores", "transformer_forward_attempts")


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def percentile(ordered, probability):
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def quantiles(values):
    ordered = sorted(values)
    require(all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in ordered), "invalid latency")
    return {"count":len(ordered), **{name:percentile(ordered, probability) if ordered else None
            for name, probability in (("p50_ms", .5), ("p95_ms", .95), ("p99_ms", .99))}}


def same(left, right, path="result"):
    if isinstance(left, dict):
        require(isinstance(right, dict) and left.keys() == right.keys(), f"keys differ: {path}")
        for key in left:
            same(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), f"length differs: {path}")
        for index, (first, second) in enumerate(zip(left, right)):
            same(first, second, f"{path}.{index}")
    elif type(left) is float:
        require(type(right) in (float, int) and math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9), f"number differs: {path}")
    else:
        require(left == right and type(left) is type(right), f"value differs: {path}")


def agreement(shared, worker):
    require([row["record_id"] for row in shared] == [row["record_id"] for row in worker], "paired order differs")
    both, exact, predictions = 0, 0, 0
    for left, right in zip(shared, worker):
        if left["error"] is not None or right["error"] is not None:
            continue
        both += 1
        first = {key:value for key,value in left["response"].items() if key != "request_id"}
        second = {key:value for key,value in right["response"].items() if key != "request_id"}
        exact += first == second
        first.pop("admission_sequence")
        second.pop("admission_sequence")
        predictions += first == second
    return {"requested":len(shared), "both_successful":both, "exact_except_request_id":exact,
            "prediction_agreement":predictions, "noncomparable_errors":len(shared)-both}


def primary(ratios, success_latencies, *, errors, exact):
    require(len(ratios) == 10 and len(success_latencies) + errors == 100000, "incomplete primary comparison")
    require(type(errors) is int and 0 <= errors <= 100000 and type(exact) is int and 0 <= exact <= 100000, "invalid denominator")
    undefined = [index+1 for index, ratio in enumerate(ratios) if ratio is None]
    median, interval = None, None
    if not undefined:
        require(all(math.isfinite(value) and value >= 0 for value in ratios), "invalid ratio")
        median = statistics.median(ratios)
        generator = np.random.Generator(np.random.PCG64(20261002))
        estimates = sorted(statistics.median(ratios[index] for index in generator.integers(0, 10, size=10)) for _ in range(10000))
        interval = [percentile(estimates, .0125), percentile(estimates, .9875)]
    latency = quantiles(success_latencies)
    requirements = {
        "median_ratio_at_most_point_eight":median is not None and median <= .8,
        "ratio_interval_upper_below_one":interval is not None and interval[1] < 1,
        "pooled_worker_success_p95_at_most_200ms":latency["p95_ms"] is not None and latency["p95_ms"] <= 200,
        "errors_below_one_per_thousand":errors < 100,
        "exact_paired_response_agreement":exact == 100000,
    }
    return {"ratios":ratios,"median_ratio":median,"ratio_interval_97_5":interval,
            "undefined_pairs":undefined,"bootstrap_replicates":10000,"bootstrap_seed":20261002,
            "worker_success_latency":latency,"worker_errors":errors,"worker_error_rate":errors/100000,
            "request_denominator":100000,"exact_paired_agreement":exact,"requirements":requirements,
            "S_requirement_met":all(requirements.values())}


def validate_phase(run, phase, before, after, count, manifest_hash):
    rows = run[phase]
    require(len(rows) == count, "incomplete phase")
    delta = {key:after[key]-before[key] for key in COUNTERS}
    require(all(type(value) is int and value >= 0 for value in delta.values()), "invalid counter delta")
    require(delta["admitted_requests"] <= count and delta["completed_requests"]+delta["failed_requests"] == delta["admitted_requests"], "undrained counters")
    require(delta["successful_transformer_scores"] == delta["transformer_forward_attempts"] == 0, "unexpected transformer work")
    sequences = set()
    success = 0
    for index, row in enumerate(rows):
        expected = f'{manifest_hash}.{run["concurrency"]}.{run["run_index"]}.{phase}.{index}'
        require(row["record_id"] == f"synthetic-{index:05d}" and row["request_id"] == expected, "request identity/order mismatch")
        quantiles([row["elapsed_ms"]])
        if row["error"] is not None:
            require(row["error"] in ERRORS and row["response"] is None, "invalid error outcome")
            continue
        response = row["response"]
        sequence = response["admission_sequence"]
        require(row["status_code"] == 200 and row["elapsed_ms"] <= 2000, "invalid successful response")
        require(response["request_id"] == expected and sequence not in sequences and before["admitted_requests"] < sequence <= after["admitted_requests"], "invalid response correlation")
        require(response["stage2_invoked"] is False and math.isfinite(response["probability"]) and 0 <= response["probability"] <= 1, "invalid prediction fields")
        sequences.add(sequence)
        success += 1
    require(success <= delta["completed_requests"], "more successes than physical completions")
    return delta


def summarize(run, manifest_hash):
    require(set(run["initial"]) == set(COUNTERS) and not any(run["initial"].values()), "nonfresh counters")
    validate_phase(run, "warmup", run["initial"], run["after_warmup"], 1000, manifest_hash)
    counters = validate_phase(run, "measured", run["after_warmup"], run["after_measured"], 10000, manifest_hash)
    rows = run["measured"]
    elapsed = run["measured_elapsed_ms"]
    drain = run["measured_drain_ms"]
    require(math.isfinite(elapsed) and elapsed > 0 and math.isfinite(drain) and drain > 0, "invalid phase times")
    minimum = max(max(row["elapsed_ms"] for row in rows), math.fsum(row["elapsed_ms"] for row in rows)/run["concurrency"])
    require(elapsed >= minimum or math.isclose(elapsed,minimum,rel_tol=1e-12,abs_tol=1e-6), "phase shorter than occupied work")
    errors = sum(row["error"] is not None for row in rows)
    all_latency = quantiles([row["elapsed_ms"] for row in rows])
    return {"concurrency":run["concurrency"],"manifest_sha256":manifest_hash,
            "request_count":10000,"request_errors":errors,"request_error_rate":errors/10000,
            **{key:value for key,value in all_latency.items() if key != "count"},
            "measured_elapsed_ms":elapsed,"measured_drain_ms":drain,
            "client_attempts_per_second":10000*1000/elapsed,"successful_responses_per_second":(10000-errors)*1000/elapsed,
            **counters,"all_latency":all_latency,
            "success_latency":quantiles([row["elapsed_ms"] for row in rows if row["error"] is None]),
            "failure_latency":quantiles([row["elapsed_ms"] for row in rows if row["error"] is not None]),
            "error_types":{name:sum(row["error"] == name for row in rows) for name in ERRORS}}


def check_conditions(path, inhibitor_pid):
    observations = [json.loads(line) for line in path.read_text().splitlines()]
    require(bool(observations), "missing condition observations")
    for row in observations:
        require(row["violation"] is None and not row["competing_known_workload_pids"], "condition violation")
        require("Now drawing from 'AC Power'" in row["battery"], "non-AC sample")
        expected = {"Note: No thermal warning level has been recorded", "Note: No performance warning level has been recorded", "Note: No CPU power status has been recorded"}
        require(set(row["thermal"].strip().splitlines()) == expected, "non-normal thermal sample")
        settings = row["power_settings"].split("AC Power:\n")
        require(len(settings) == 2 and re.findall(r"^\s*powermode\s+([0-9]+)\s*$",settings[1],re.M) == ["0"], "nonautomatic energy mode")
        require(not re.search(r"^\s*lowpowermode\s+[1-9]",settings[1],re.M), "low power mode")
        for kind in ("PreventSystemSleep", "PreventUserIdleSystemSleep", "PreventUserIdleDisplaySleep"):
            require(re.search(rf"pid\s+{inhibitor_pid}\(caffeinate\):[^\n]*\b{kind}\b", row["assertions"]), "missing owned inhibition")
    return observations


def flatten(summary):
    result = {}
    for key,value in summary.items():
        if isinstance(value,dict):
            result.update({f"{key}_{name}":item for name,item in value.items()})
        else:
            result[key] = value
    return result


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(root=ROOT):
    attempt = root / "service-comparison-v2"
    completed = json.loads((attempt/"completion.json").read_bytes())
    require(not list(attempt.rglob("failure.json")), "completed attempt contains failure")
    intent = json.loads((attempt/"intent.json").read_bytes())
    auth_path = root/"service-recovery-authorization-v2.json"
    authorization = json.loads(auth_path.read_bytes())
    require(digest(auth_path) == intent["authorization_sha256"] == completed["recovery_authorization_sha256"], "authorization binding changed")
    same(authorization,intent["recovery_authorization"])
    for name, expected in authorization["files"].items():
        require(digest(root/name) == expected, "bound evidence changed")
    require(digest(root.parent/"run_followup_service_recovery_20261002.py") == authorization["launcher_sha256"], "launcher changed")
    require(digest(root.parent/"test_followup_service_recovery_20261002.py") == authorization["test_sha256"], "launcher tests changed")
    require(digest(root/"service-recovery-tests-v2.txt") == authorization["verification_receipt_sha256"], "test receipt changed")
    preservation = json.loads((root/"service-v1-preservation.json").read_bytes())
    old = root/"service-comparison-v1"
    require({str(path.relative_to(old)):digest(path) for path in old.rglob("*") if path.is_file()} == preservation["source_sha256"], "interrupted evidence changed")
    require(completed["interrupted_schedule_pooled"] is False and completed["original_hypothesis_decisions_changed"] is False, "scope changed")
    manifest = json.loads((root/"execution-manifest-v1.json").read_bytes())
    require(digest(root/"execution-manifest-v1.json") == intent["execution_manifest_sha256"], "execution manifest changed")
    same(manifest,intent["manifest"])
    context = root.parents[1]
    repository = context/"gwu_working/study-followup-development-20261001"
    require(subprocess.check_output(["git","rev-parse","HEAD"],cwd=repository,text=True).strip() == manifest["code_revision"], "frozen code changed")
    require(not subprocess.check_output(["git","status","--porcelain"],cwd=repository,text=True), "frozen source dirty")
    require(digest(context/"gwu_working/study-only-v1-reviewed-inputs/primary/logistic-l1.json") == manifest["baseline_file_sha256"], "baseline changed")
    request_manifest = json.loads((attempt/"synthetic-manifest.json").read_bytes())
    manifest_hash = digest(attempt/"synthetic-manifest.json")
    require(manifest_hash == intent["synthetic_manifest_sha256"] == digest(old/"synthetic-manifest.json"), "synthetic inputs changed")
    require(request_manifest["class_prevalence"] is None and len(request_manifest["requests"]) == 10000, "wrong synthetic population")
    for index,row in enumerate(request_manifest["requests"]):
        require(row == {"record_id":f"synthetic-{index:05d}","raw_url":f"{'https' if index%2 else 'http'}://host{index%257}.example/path/{index}?token={index%97}"}, "synthetic manifest mismatch")
    schedule = [{"workload":workload,"concurrency":concurrency,"pair":pair,"client":client}
                for workload in ("no_model","structural_detector") for concurrency in (1,64)
                for pair in range(1,11) for client in (("shared","worker") if pair%2 else ("worker","shared"))]
    same(schedule,intent["schedule"])
    require(completed["arm_count"] == 80, "not all 80 arms completed")
    arms, pairs, groups = [], [], defaultdict(list)
    primary_rows, primary_latency, primary_errors, primary_exact, primary_prediction = [], [], 0, 0, 0
    previous_end = intent["started_at"]
    warmup_errors = 0
    for offset in range(0,80,2):
        runs, summaries = {}, {}
        for arm in schedule[offset:offset+2]:
            name = f'{arm["workload"]}-c{arm["concurrency"]}-pair{arm["pair"]:02d}-{arm["client"]}'
            directory = attempt/name
            record = json.loads((directory/"run.json").read_bytes())
            done = json.loads((directory/"completion.json").read_bytes())
            launch = json.loads((directory/"launch.json").read_bytes())
            exited = json.loads((directory/"exit.json").read_bytes())
            require(launch["started_at"] > previous_end and done["completed_at"] > exited["ended_at"] > launch["started_at"], "arm chronology invalid")
            require(exited["pid"] == launch["pid"] and exited["exit_code"] == 0 and exited["forced"] is False, "unclean arm exit")
            previous_end = done["completed_at"]
            require(record["protocol"] == "bounded-followup-service-v1" and record["synthetic_class_prevalence"] is None, "wrong record protocol")
            same(arm,record["arm"])
            same(arm,done["arm"])
            require(done["run_sha256"] == digest(directory/"run.json"), "arm source changed")
            run = record["legacy_transport_record"]["run"]
            require(run["manifest_sha256"] == manifest_hash and run["concurrency"] == arm["concurrency"] and run["run_index"] == (arm["pair"]-1)%5+1, "arm run identity differs")
            require(run["workload"] == "fixed_cascade" and run["prevalence_basis_points"] == 100, "legacy codec identity changed")
            summary = summarize(run,manifest_hash)
            same(summary,done["summary"])
            same({"summary":summary,**arm},completed["arms"][len(arms)])
            warmup_errors += sum(row["error"] is not None for row in run["warmup"])
            arms.append({**arm,**flatten(summary)})
            runs[arm["client"]] = run["measured"]
            summaries[arm["client"]] = summary
            groups[(arm["workload"],arm["concurrency"],arm["client"])].append((run["measured"],summary))
        identity = {key:schedule[offset][key] for key in ("workload","concurrency","pair")}
        match = agreement(runs["shared"],runs["worker"])
        same({**identity,"response_agreement":match},completed["pairs"][len(pairs)])
        shared, worker = [summaries[client]["success_latency"]["p95_ms"] for client in ("shared","worker")]
        ratio = worker/shared if shared is not None and shared > 0 and worker is not None else None
        pair_row = {**identity,"shared_p95_ms":shared,"worker_p95_ms":worker,"ratio":ratio,**match}
        pairs.append(pair_row)
        if identity["workload"] == "structural_detector" and identity["concurrency"] == 64:
            primary_rows.append(pair_row)
            primary_latency.extend(row["elapsed_ms"] for row in runs["worker"] if row["error"] is None)
            primary_errors += summaries["worker"]["request_errors"]
            primary_exact += match["exact_except_request_id"]
            primary_prediction += match["prediction_agreement"]
    result = primary([row["ratio"] for row in primary_rows],primary_latency,errors=primary_errors,exact=primary_exact)
    same(result,completed["primary"])
    group_rows = []
    for (workload,concurrency,client), selected in groups.items():
        rows = [row for outcomes,_ in selected for row in outcomes]
        elapsed = sum(summary["measured_elapsed_ms"] for _,summary in selected)
        errors = sum(row["error"] is not None for row in rows)
        group = {"workload":workload,"concurrency":concurrency,"client":client,"arms":len(selected),"request_count":len(rows),"request_errors":errors,"request_error_rate":errors/len(rows),"measured_elapsed_ms":elapsed,"client_attempts_per_second":len(rows)*1000/elapsed,"successful_responses_per_second":(len(rows)-errors)*1000/elapsed}
        for label, chosen in (("all",rows),("success",[row for row in rows if row["error"] is None]),("failure",[row for row in rows if row["error"] is not None])):
            group.update({f"{label}_{key}":value for key,value in quantiles([row["elapsed_ms"] for row in chosen]).items()})
        group.update({key:sum(summary[key] for _,summary in selected) for key in COUNTERS})
        group_rows.append(group)
    inhibitor = json.loads((attempt/"sleep-inhibitor-exit.json").read_bytes())
    require(inhibitor["exit_code"] == -15, "unexpected inhibitor shutdown")
    observations = check_conditions(attempt/"conditions.jsonl",inhibitor["pid"])
    preflight = root/"service-recovery-preflight-v2"
    stable = check_conditions(preflight/"conditions.jsonl",inhibitor["pid"])
    transition = check_conditions(preflight/"transition.jsonl",inhibitor["pid"])
    require(digest(preflight/"completion.json") == intent["preflight_completion_sha256"] and digest(preflight/"transition.jsonl") == intent["preflight_transition_sha256"], "preflight receipts changed")
    preflight_receipt = json.loads((preflight/"completion.json").read_bytes())
    require(digest(preflight/"conditions.jsonl") == preflight_receipt["conditions_sha256"], "stable samples changed")
    stable_seconds = (datetime.fromisoformat(stable[-1]["observed_at"])-datetime.fromisoformat(stable[0]["observed_at"])).total_seconds()
    require(stable_seconds >= 180 and len(stable) >= 2, "insufficient stable preflight")
    require(transition[-1]["observed_at"] < intent["started_at"] < observations[0]["observed_at"], "preflight not before schedule")
    report = {"status":"verified","checked_at":datetime.now(timezone.utc).isoformat(),
              "scope":"Independent arithmetic over retained JSON; not independent external replication. No fitting or predictions.",
              "arms":80,"pairs":40,"groups":8,"measured_requests":800000,"warmup_requests":80000,
              "measured_errors":sum(row["request_errors"] for row in arms),"warmup_errors":warmup_errors,
              "primary":result,"primary_prediction_agreement":primary_prediction,
              "all_pairs_prediction_agreement":sum(row["prediction_agreement"] for row in pairs),
              "conditions":{"schedule_samples":len(observations),"preflight_samples":len(stable),"stable_observed_seconds":stable_seconds,"violations":0,"sampled_not_continuous_proof":True},
              "started_at":intent["started_at"],"completed_at":completed["completed_at"],
              "interrupted_v1_files_unchanged":len(preservation["source_sha256"]),"interrupted_v1_pooled":False,
              "original_hypothesis_decisions_changed":False,
              "verifier_sha256":digest(Path(__file__)),
              "source_sha256":{str(path.relative_to(root)):digest(path) for directory in (attempt,preflight) for path in sorted(directory.rglob("*")) if path.is_file()}}
    output = root/"verified-service-v2"
    output.mkdir()
    write_csv(output/"arm-metrics.csv",arms)
    write_csv(output/"group-metrics.csv",group_rows)
    write_csv(output/"pair-metrics.csv",pairs)
    write_csv(output/"primary-pairs.csv",primary_rows)
    write_csv(output/"requirements.csv",[{"requirement":key,"passed":value} for key,value in result["requirements"].items()])
    report["aggregate_sha256"] = {path.name:digest(path) for path in sorted(output.glob("*.csv"))}
    with (output/"verification.json").open("x") as stream:
        json.dump(report,stream,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({key:value for key,value in report.items() if key != "source_sha256"},indent=2))
    return report


if __name__ == "__main__":
    verify()
