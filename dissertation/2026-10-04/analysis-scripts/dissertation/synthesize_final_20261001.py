"""Assemble the results-complete manuscript from verified, retained aggregates."""

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "final-evidence-20261001"
ARCHIVE = HERE / "Tallam_Krti_Praxis_v3_Body_PreSynthesis_2026-10-01.md"


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def records(name):
    with (EVIDENCE / name).open() as source:
        return list(csv.DictReader(source))


def table(headers, rows):
    widths = [min(36, max(10, len(str(header)), max(len(str(row[index])) for row in rows))) for index, header in enumerate(headers)]
    return "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in [headers, ["-" * width for width in widths], *rows])


def percent(value):
    return "undefined" if value is None else f"{100 * float(value):.4f}%"


def decimal(value):
    return "undefined" if value is None else f"{float(value):.6f}"


def label(name):
    return {"length_only": "Length-only", "logistic_l1": "Logistic-L1", "cascade": "Fixed cascade", "transformer": "Transformer", "policy": "GMM policy", "tabular.formatting": "Formatting", "tabular.random_forest": "Random Forest"}.get(name, name.replace("tabular.", "").replace("_", " "))


def replace_paragraph(text, prefix, replacement):
    paragraphs = text.split("\n\n")
    matches = [index for index, paragraph in enumerate(paragraphs) if paragraph.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one paragraph beginning {prefix!r}, found {len(matches)}")
    paragraphs[matches[0]] = replacement
    return "\n\n".join(paragraphs)


def main():
    if read("verification.json")["status"] != "verified" or read("secondary-verification.json")["status"] != "verified_and_exported":
        raise ValueError("Verification prerequisites are incomplete")
    secondary = read("complete-secondary-results.json")
    primary = read("primary-results.json")
    internal = secondary["internal"]["metrics"]
    external = secondary["external"]["populations"]["gold_plus_certified"]["detectors"]
    original = ARCHIVE.read_text()
    methods = original.split("# Chapter 4—", 1)[0]
    references = original.split("## References\n\n", 1)[1]
    references += "\n\nEfron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552\n\nHolm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65–70. https://www.jstor.org/stable/4615733\n\nMcNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), 153–157. https://doi.org/10.1007/BF02295996"
    references = "\n\n".join(sorted(paragraph.strip() for paragraph in references.split("\n\n") if paragraph.strip()))
    methods = methods.replace("Primary recall differences are estimated with", "Bootstrap resampling follows the general framework of Efron (1979), with the domain-clustered estimand specified below. Primary recall differences are estimated with")
    methods = methods.replace("The four McNemar contrasts", "The four paired contrasts use the exact conditional form of McNemar's (1947) test with Holm's (1979) familywise adjustment. The four McNemar contrasts")
    methods = methods.replace("structural URL features and selective character-model escalation provide", "structural URL features and character-level representations provide")
    methods = methods.replace("upper confidence bounds will accompany", "upper confidence bounds accompany")
    methods = methods.replace("The value of this study will depend on the locked joint evaluation and its recorded results", "The value of this study lies in the locked joint evaluation and its recorded results")
    methods = methods.replace("Future paired predictions will use this common convention", "The accepted paired predictions use this common convention")
    methods = methods.replace("the current outstanding work is verification and completion of study evidence under the disclosed recovery boundary", "Chapter 4 reports the completed, verified evidence under the disclosed recovery boundary")
    methods = methods.replace("The analysis will not attribute", "The analysis does not attribute")
    methods = methods.replace("Section 4.7", "Section 4.6").replace("Section 4.8 reports the separate accepted development run", "Section 4.5 reports the accepted secondary evidence").replace("Section 4.9", "Section 4.6").replace("Section 4.10", "Section 4.5")
    methods = methods.replace("H1 and H3 await final adjudication.", "H1 and H3 are not supported by the complete results in Chapter 4.")
    methods = methods.replace("## 1.4 Research Questions and Hypotheses", """### 1.3.1 Thesis Statement

CyberSentinel will test low-FPR structural and character-level detection, GMM-guided escalation, and whether selective execution stays within two percentage points of transformer-only recall while invoking the transformer no more than 30%, keeping p95 at or below 200 ms, and holding errors below 0.1%.

This is the working thesis stated on slide 5 of the August 20 advisor deck, not a claim that the targets were achieved. The three questions and their unchanged conjunctive hypotheses operationalize the proposition. Chapter 5 distinguishes the measured component benefits from the joint claim that the complete system is viable inline.

### 1.3.2 Research Objectives

The first objective is to quantify the incremental recall and false-positive behavior of structural and character representations on registrable-domain-disjoint internal data and external source/tier strata. The second is to measure distribution sensitivity, reference false alerts and the consequences of future-only escalation without treating an alert as proof of harmful drift. The third is to measure actual HTTP latency, throughput, transformer attempts and request errors against the frozen operational constraints. Together these objectives provide an end-to-end answer about the tested system, rather than separate favorable accuracy and speed claims.

## 1.4 Research Questions and Hypotheses""")
    methods = methods.replace("| Closest work |", "Table 2.1. Closest prior work and the boundary of the present contribution.\n\n| Closest work |", 1)
    methods = methods.replace("Final synthesis must use the authenticated evidence selected", "Final synthesis uses the authenticated evidence selected")
    methods = methods.replace("observed development result is reported in Section 4.4", "observed development result is reported in Section 4.3")
    methods = replace_paragraph(methods, "The study evaluates representation value", "The study evaluates representation value, shift-aware routing, and inline viability through an auditable protocol and its disclosed amendments. PhiUSIIL supplies fitting, validation and registrable-domain-disjoint internal evaluation; PhishVN v3.1.0, Mendeley Data repository Version 4, supplies external source/domain evaluation. The September 28 joint preparation stopped before prediction because the original raw-field rules left required populations absent. An operator-authorized, preparation- and label-count-informed amendment adopted the publisher's exact external url_norm while preserving raw cells and all models, thresholds and population rules. Four amended executions were interrupted. Following a disclosed post-interruption checkpoint amendment, both source outputs and all 72 complete cells from the fourth attempt qualified for retention; 53 complete remaining cells were measured in original order. The final evidence contains all 125 operational cells, 25 groups, 22 primary gates and the declared secondary analyses. All three hypotheses are not supported under their unchanged conjunctive rules. Completion of the investigation and support for a hypothesis are separate questions; Chapters 4 and 5 report the measured answers and their limits.")
    methods = replace_paragraph(methods, "The study began with a prospective staged-freeze design", "The study used a staged-freeze design for three linked evaluations. RQ1 compares representations under fixed validation-selected thresholds; RQ2 tests distribution monitoring and future-only routing; RQ3 measures the real HTTP path. Development, evaluation and execution amendments have different dates and evidentiary roles. The accepted study combines unchanged scientific source results and cells 1–72 from the fourth amended attempt with cells 73–125 from the October 1 continuation. It is therefore a disclosed two-session evaluation, not a single uninterrupted experiment or a wholly pre-data specification. No accepted adverse cell was replaced. The source and code lineage, the original preparation hold, all interrupted attempts and the later acceptance decisions remain preserved in the accompanying history supplement.")
    methods = replace_paragraph(methods, "The fixed schedule contains 125 cells", "The fixed schedule contains 125 cells in 25 five-repeat groups: ninety fixed-cascade cells across three prevalences and six concurrencies, thirty transformer-only cells on the 1% manifest, and five serialized live-shift repeats. Each fixed/transformer repeat measures 10,000 requests; each live-shift repeat measures the actual 8,701-row external stream after its separate 1,000-request warmup and phase reset. The original single-session policy was amended after four interruptions. Both source results and cells 1–72 from attempt four passed the all-or-none historical review; the October 1 segment completed cells 73–125. The stopped cell 73 was measured afresh in full, without splicing its partial requests. All 125 accepted cells retain their original ordinal, request/deadline rules and manifests. Primary H3 reference cell 1 and latency/error cells 21–25 remain in the historical session; descriptive group 71–75 spans sessions. Session effects cannot be separated from fixed workload order.")
    methods = replace_paragraph(methods, "The September 30 recovery instruction authorizes", "The September 30 completion directive authorized bounded checkpoint implementation, retained-evidence verification and conditional continuation, not waiver of scientific or physical gates. The exact continuation code was frozen at 77d128377ce5b401437d7179f5cd78fb4294b72c with a hash-bound study-only profile and execution envelope. The sealed history selected only attempt four; attempts one through three supplied no substitute observations. Both scientific source worker exits, all accepted cells, preparation ancestry and stop history were checked before continuation. The final verification reauthenticated that evidence and recomputed saved-outcome counts, primary intervals and pooled latency reductions without fitting or executing a model. It observed all 250 owned service/client exits as successful. In the October 1 segment, 1,945 retained condition samples recorded AC, sleep inhibition, no thermal/performance warning and no detected known competing workload. Sampled known-command checks cannot exclude every possible workload or establish uninterrupted conditions between samples. Root exit was zero with no recorded session violation. This procedural verification is not independent external replication or advisor approval.")
    methods = methods.replace("Its use in a checkpointed continuation requires historical custody, scientific and physical eligibility verification", "Its use in the checkpointed continuation required historical custody, scientific and physical eligibility verification")
    methods = methods.replace("The prospective contribution is a joint systems evaluation", "The contribution is a joint systems evaluation")
    methods = methods.replace("The narrower unresolved question is whether", "The question evaluated here is whether")
    methods = methods.replace("The novelty claim is bounded to that prospective joint systems evaluation. A null finding would still be informative because it would identify which constraint fails under the common protocol.", "The contribution claim is bounded to that joint systems evaluation. The study identifies which constraints fail under the common protocol; it does not establish priority for a component method.")

    detection_rows = []
    for population, metrics in [("Internal", internal), ("External", external)]:
        for name in ["length_only", "logistic_l1", "transformer", "cascade"] + (["policy"] if population == "External" else []):
            counts = metrics[name]["counts"]
            detection_rows.append([population + ": " + label(name), f"{counts['true_positives']}/{counts['recall']['denominator']}", str(counts['false_negatives']), f"{counts['false_positives']}/{counts['fpr']['denominator']}", str(counts['true_negatives']), percent(counts['recall']['estimate']), percent(counts['fpr']['estimate']), percent(counts['fpr']['upper_95'])])
    detection = "Table 4.2a. Confusion counts; P and N are the declared stratum denominators.\n\n" + table(["Population / model", "TP/P", "FN", "FP/N", "TN"], [row[:5] for row in detection_rows])
    detection += "\n\nTable 4.2b. Corresponding rates and one-sided FPR upper bounds.\n\n" + table(["Population / model", "Recall", "FPR", "FPR upper95"], [[row[0], *row[5:]] for row in detection_rows])
    contrasts = records("paired-contrasts.csv")
    contrast_table = table(["Positive stratum / contrast", "Candidate / reference TP", "Rows / domains", "Difference (pp)", "95% interval (pp)"], [[row["contrast"].replace("_", " "), f"{row['candidate_true_positives']} / {row['reference_true_positives']}", f"{row['positive_count']} / {row['domain_count']}", f"{100 * float(row['estimate']):.4f}", f"[{100 * float(row['lower']):.4f}, {100 * float(row['upper']):.4f}]"] for row in contrasts])
    gate_rows = []
    for row in records("primary-gates.csv"):
        is_latency = row["name"] == "http_pooled_p95_ms"
        operand = f"{float(row['estimate']):.4f} ms" if is_latency else percent(row["estimate"])
        threshold = f"{float(row['threshold']):g} ms" if is_latency else percent(row["threshold"])
        if "_minus_" in row["name"]:
            operand = f"{100 * float(row['estimate']):.4f} pp"
            threshold = f"{100 * float(row['threshold']):.4f} pp"
        gate_rows.append([row["hypothesis"], row["name"].replace("_", " "), operand, row["operator"] + " " + threshold, row["status"]])
    gates = table(["H", "Gate", "Observed operand", "Required", "Result"], gate_rows)
    groups = records("operational-groups.csv")
    runs = records("operational-runs.csv")
    operational_rows = []
    throughput_rows = []
    for group in groups:
        matching = [run for run in runs if all(run[field] == group[field] for field in ("workload", "concurrency", "prevalence_basis_points"))]
        if len(matching) != 5:
            raise ValueError("Operational group lacks all five runs")
        caption = {"fixed_cascade": "Fixed", "transformer_only": "Transformer", "shift_period": "Live shift"}.get(group["workload"], group["workload"])
        prevalence = "external" if group["workload"] == "shift_period" else f"{float(group['prevalence_basis_points']) / 100:g}%"
        title = f"{caption} {prevalence} / c{group['concurrency']}"
        operational_rows.append([title, f"{float(group['p50_ms']):.3f}", f"{float(group['p95_ms']):.3f}", f"{float(group['p99_ms']):.3f}", f"{group['request_errors']}/{group['request_count']}", percent(group["physical_invocation_fraction"])])
        attempts = [float(run["client_attempts_per_second"]) for run in matching]
        successes = [float(run["successful_responses_per_second"]) for run in matching]
        throughput_rows.append([title, f"{min(attempts):.2f}–{max(attempts):.2f}", f"{min(successes):.2f}–{max(successes):.2f}", f"{min(float(run['measured_drain_ms']) for run in matching):.3f}–{max(float(run['measured_drain_ms']) for run in matching):.3f}"])
    operations = table(["Workload / concurrency", "p50 ms", "p95 ms", "p99 ms", "Errors / requests", "Physical attempts"], operational_rows)
    throughput = table(["Workload / concurrency", "Attempts/s range", "Successes/s range", "Drain ms range"], throughput_rows)
    monitor_rows = []
    historical = secondary["historical"]
    gmm = historical["rq2-gmm-development-v1-summary.json"]["data"]
    development = historical["secondary-development-correction-v2-summary.json"]["data"]["completion"]
    drift = next(member for member in development["retained_audit"]["result"]["members"] if member["member"] == "drift")["summary"]["result"]
    for monitor in secondary["monitors"]:
        count = sum(window["alert"] for window in monitor["windows"])
        audit_count = gmm["audit_alert_count"] if monitor["name"] == "gmm" else drift["methods"][monitor["name"]]["audit_alert_count"]
        monitor_rows.append([monitor["name"].upper(), str(monitor["threshold"]), f"{audit_count}/252 ({percent(audit_count / 252)})", f"{count}/132 ({percent(count / 132)})"])
    monitor_table = table(["Monitor", "Fixed boundary", "Original audit alerts", "External alerts"], monitor_rows)
    bic_table = table(["Components", "Training BIC", "Iterations", "Converged"], [[row["components"], f"{row['bic']:.6f}", row["n_iter"], row["converged"]] for row in gmm["candidates"]])
    metric_rows = []
    for population, metrics in [("Internal", internal), ("Gold + certified", external)]:
        for name in ["length_only", "logistic_l1", "transformer", "cascade", "tabular.formatting", "tabular.random_forest"] + (["policy"] if population != "Internal" else []):
            metric = metrics[name]
            metric_rows.append([population + ": " + label(name)] + [decimal(metric[field]["value"]) for field in ("average_precision", "roc_auc", "brier", "calibration_error")])
    metric_table = table(["Population / detector", "AP", "ROC AUC", "Brier", "10-bin ECE"], metric_rows)
    permutation_table = table(["Permutation seed", "Internal AP", "Internal AUC", "External AP", "External AUC"], [[seed] + [decimal(metrics[f"tabular.permutation_{seed}"][field]["value"]) for metrics in (internal, external) for field in ("average_precision", "roc_auc")] for seed in range(42, 47)])
    seed_table = table(["Seed", "Internal T recall", "Internal T FPR", "Gold T recall", "Certified T FPR", "Fixed band selections"], [[seed, percent(internal[f"seed_{seed}.transformer"]["counts"]["recall"]["estimate"]), percent(internal[f"seed_{seed}.transformer"]["counts"]["fpr"]["estimate"]), percent(external[f"seed_{seed}.transformer"]["counts"]["recall"]["estimate"]), percent(external[f"seed_{seed}.transformer"]["counts"]["fpr"]["estimate"]), "0 internal; 0 external"] for seed in range(42, 47)])
    seeds = records("seed-logical-invocations.csv")
    if any(int(row["logical_band_count"]) != 0 for row in seeds):
        raise ValueError("Seed routing text needs revision")
    correction = historical["secondary-seed-probe-correction-v1-summary.json"]["data"]["completion"]
    streams = correction["probes"]["result"]["result"]["streams"]
    probe_rows, probe_decisions, probe_monitors = [], [], []
    for stream in streams:
        counts = stream["mapping_counts"]
        probe_rows.append([stream["name"].replace("_", " "), stream["row_count"]] + [counts[key] for key in ("eligible", "changed", "eligible_noop", "ineligible")])
        for detector, paired in stream["paired_with_original"]["detectors"].items():
            probe_decisions.append({"stream": stream["name"], "detector": detector, "rows": stream["row_count"], "positive_decisions": stream["detectors"][detector]["positive_decision_count"], **paired["decision_transitions"], **paired["score_differences"]})
        for monitor, paired in stream["paired_with_original"]["monitors"].items():
            probe_monitors.append({"stream": stream["name"], "monitor": monitor, "threshold": stream["monitors"][monitor]["threshold"], "alerts": stream["monitors"][monitor]["alert_count"], "windows": stream["monitors"][monitor]["window_count"], **paired["alert_transitions"], **paired["score_differences"]})
    probe_table = table(["Stream", "Rows", "Eligible", "Changed", "Eligible no-op", "Ineligible"], probe_rows)
    for name, rows in [("probe-decisions-and-scores.csv", probe_decisions), ("probe-monitors-and-scores.csv", probe_monitors)]:
        with (EVIDENCE / name).open("w") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    mcnemar_rows = []
    for cell in primary["ablation_family"]["cells"]:
        test_id = cell["test_id"]
        source = "internal" if test_id.startswith("internal") else "external"
        metric = secondary[source]["mcnemar"][test_id]
        mcnemar_rows.append([test_id.replace("_", " "), metric["both_correct"], metric["candidate_only_correct"], metric["reference_only_correct"], metric["both_incorrect"], "underflow" if cell["raw_pvalue"]["value"] == 0 else f"{cell['raw_pvalue']['value']:.6g}", "underflow" if cell["adjusted_pvalue"]["value"] == 0 else f"{cell['adjusted_pvalue']['value']:.6g}"])
    mcnemar_table = table(["Contrast", "Both correct", "Candidate only", "Reference only", "Both incorrect", "Exact p", "Holm p"], mcnemar_rows)
    tier_table = table(["Detector", "Silver TP / 417", "Bronze TP / 4,555", "Tranco alerts / 1,163"], [[label(name)] + [f"{secondary['external']['populations'][population]['detectors'][name][field]['numerator']} ({percent(secondary['external']['populations'][population]['detectors'][name][field]['estimate'])})" for population, field in [("ncsc_silver", "recall"), ("chongluadao_openphish_bronze", "recall"), ("tranco", "control_alert_rate")]] for name in ("length_only", "logistic_l1", "transformer", "cascade", "policy")])
    source_table = table(["Source / tier", "Role", "Rows", "Domains", "Reference outcome"], [[row["source_group"].replace("_", " ") + " / " + row["confidence_tier"], row["role"], row["row_count"], row["domain_count"], "unlabeled control" if row["is_phishing"] is None else ("phishing (1)" if row["is_phishing"] == 1 else "legitimate (0)")] for row in secondary["external"]["source_contingency"]])
    chapters = f'''# Chapter 4—Results

## 4.1 Completed Evaluation and Populations

The study completed its frozen primary measurement matrix and reports the full declared secondary program, including its identifiability and audit limits. The final accepted matrix contains 125 operational cells in 25 five-repeat groups, both source evaluations and all 22 primary hypothesis checks. The retained fourth-attempt prefix contributed 72 eligible complete cells; the October 1 continuation contributed the remaining 53 in original order. No accepted cell was selected or discarded according to its result. H1, H2 and H3 are each complete and not supported under their original conjunctive decision rules. This is a completed primary evaluation with adverse findings, not a partial study with missing gates.

Final saved-evidence verification checked 7,340 file-hash comparisons and 238,159 assertions, authenticated both historical scientific worker exits and 250 actual service/client exits, and independently recomputed primary counts, six clustered contrasts and pooled terminal latency reductions. The continuation completed at 2026-10-01T22:05:09Z with root exit 0 and no recorded session violation. “Independent recomputation” means a separate read-only calculation from saved observations, not an independent investigator or replication. No model was refitted, no new predictions were made for synthesis and no original dataset was reopened.

PhiUSIIL preparation retained 233,536 of 235,795 rows across 197,105 registrable domains. The 2,259 exclusions comprise 1,380 invalid or unsupported URLs, 877 same-label canonical duplicates and two rows from one conflicting canonical group. The train, validation and group-test partitions contain 166,248, 32,695 and 34,593 rows, respectively, with 137,973, 29,566 and 29,566 domains. The group-test has 14,326 positives and 20,267 negatives. Its positive stratum contains 9,757 domains. These are publisher reference outcomes, not newly adjudicated labels.

The amended external preparation retained 8,701 of 8,941 published test records, with 240 test exclusions and 4,150 distinct retained domains. The primary outcome strata comprise 69 NCSC gold-positive records in 69 domains and 2,497 certified-registry negative records in 234 domains. Secondary strata comprise 417 NCSC silver positives in 412 domains, 4,555 Chongluadao/OpenPhish bronze positives in 2,294 domains, and 1,163 label-free Tranco controls in 1,163 domains. Stratum domain counts need not add to the overall distinct-domain count. The entire retained external stream participates in routing before these outcome strata are applied. The source contingency is shown below; it must not be interpreted as deployment prevalence.

Table 4.1. Retained external source, tier and outcome composition.

{source_table}

The complete preparation package covers 53,116 publisher rows; its overall 38 invalid-URL and 1,423 PhiUSIIL-overlap exclusions are not test-only counts. Test exclusions, split membership, canonical-overlap and registrable-domain-overlap checks remain in the authenticated preparation records. The original raw-field hold and the exact publisher-url_norm amendment are disclosed in Section 4.6. No local repair or label reinterpretation was used to improve these results.

## 4.2 RQ1: Representation Value and Generalization

Table 4.2 reports fixed operating points. Internal recall uses P=14,326 and FPR uses N=20,267. External recall uses the 69 gold positives; external FPR uses the 2,497 certified negatives. TN and FN are supplied explicitly. Upper95 is the exact one-sided 95% Clopper–Pearson FPR bound. The prespecified test gates use observed FPR, not that upper bound. The bounds describe the binomial calculation and do not account for residual dependence among URLs sharing a domain.

Table 4.2. Primary detection counts and rates at unchanged validation-selected thresholds.

{detection}

Structural features yield substantial incremental recall over length alone: 64.0723 percentage points internally and 76.8116 points on gold positives. Their clustered intervals exclude zero. Internally, the structural model retains FPR below 1%. Externally, however, its certified FPR is 90.5487%, compared with 9.4513% for length alone. Thus structural discrimination learned from PhiUSIIL does not transport as an acceptable low-FPR operating point to the certified external population. The different source composition and representation prevent a causal attribution of this degradation to any single feature or data source.

Table 4.3. All six primary paired recall contrasts; 2,000 domain-clustered PCG64 replicates per contrast. Differences and intervals are percentage points.

{contrast_table}

The fixed band selects zero of 34,593 internal and zero of 8,701 external rows. Consequently, cascade and Logistic-L1 decisions coincide in both populations; both incremental cascade intervals equal [0,0] and fail the strict-improvement rule. This is a degenerate selective policy on the evaluated populations, not evidence that character representations are unnecessary generally. Transformer-only recall is 99.3229% internally, versus 98.6807% for Logistic-L1, but its internal FPR is 1.0115% and its external certified FPR is 99.2791%. That descriptive comparator is not a substitute H1 contrast.

H1 has five passing and five failing components. Its two positive structural-recall contrasts cannot compensate for three failed external FPR checks and two failed incremental-cascade contrasts. H1 is not supported. The answer to RQ1 is therefore conditional: the structural representation adds recall, especially internally, but selective character escalation adds no measured recall in this frozen cascade and the required external low-FPR generalization does not hold.

## 4.3 RQ2: Monitoring and Future-Only Routing

All six GMM candidates converged under the frozen training procedure. Minimum training BIC selected six components, without using external outcomes. Calibration used 16,325 validation rows and audit used 16,370; each stream has 14,783 disjoint domains and 252 complete windows. The GMM boundary remains -67.45792380813624. The original audit alerts on 28/252 windows (11.1111%), exceeding the 5% criterion. That failure was recorded before external evaluation and remains decisive for H2.

Table 4.4. Training-only GMM model-selection record.

{bic_table}

The external stream yields 132 complete 256-row windows at stride 64. GMM alerts on 115 (87.1212%), passing the 80% detection gate. This measures departure of the external P(X) representation from the fitted reference, not the sensitivity of a detector for independently labeled harmful-drift events. An alert ending at t affects only t+1 through t+256; activations are unioned and truncated at the end of the stream. Incomplete terminal windows do not enter the alert denominator, although their requests can remain subject to earlier alerts.

The saved external policy routes 8,253/8,701 requests (94.8512%) to the transformer, compared with zero fixed-band selections. Gold recall remains 69/69 for both the policy and fixed cascade: the paired recall difference, equivalently the reduction in false-negative rate, is zero with interval [0,0]. Certified false positives increase from 2,261 to 2,479, a difference of 218 records or 8.7305 percentage points. These are realized paired policy outcomes, not a randomized causal effect. High alert frequency supplies no evidence that escalation remedies the relevant detection problem.

Table 4.5. Frozen GMM and descriptive comparator monitors. All rates use complete overlapping windows, without independent-binomial intervals.

{monitor_table}

MMD and PSI each flag every external window. Their original audit fractions are lower than the GMM's, but they are secondary comparators with their own training-fixed references, bins and calibration boundaries. Neither replaces the primary monitor nor rescues H2. All 396 external monitor-window records and 3,432 PSI feature scores are retained in the data supplement. The external gold policy-minus-cascade contrast, certified FPR and original audit gates fail; only the external-window alert gate passes. H2 is not supported. RQ2 is answered directly: this monitor detects the specified external departure, but its audit specificity and routed low-FPR utility do not meet the promised constraints.

## 4.4 RQ3: Real-HTTP Cost and Inline Viability

The measured host is an Apple M4 Max MacBook Pro with 16 CPU cores and 128 GB memory, running macOS 15.6.1; the frozen runtime uses singleton transformer inference and the recorded numerical-library limits. Hardware and environment bindings are retained with each accepted segment. The HTTP harness includes serialization, queueing, model execution, body reading and response validation. Warmup, loading and control work are excluded from measured client-phase throughput and latency; phase drain is reported separately. Closed-loop latency excludes unsent local backlog, so these results do not establish open-loop production capacity.

The primary invocation cell records zero physical transformer attempts among 10,000 measured requests, satisfying the <=30% gate. This agrees with zero fixed-band selection and cannot be sold as accuracy-preserving selective work savings when the external accuracy safeguards fail. For cells 21–25, all 50,000 terminal latencies pool to p95=364.3004101 ms, exceeding 200 ms; request errors are 0/50,000, satisfying the strict <0.1% gate. The primary cells were retained unchanged from the historical session, not replaced by later or better runs.

Table 4.6. All 25 operational groups. Each row pools five complete repeats. Fixed and transformer rows contain 50,000 requests; live shift contains 43,505. Physical attempts are transformer forward attempts divided by all measured client requests, not a latency reduction.

{operations}

Table 4.7. Five-run throughput and post-phase drain ranges. Full individual run values and counters are in operational-runs.csv; ranges are descriptions, not uncertainty intervals or best-run selection.

{throughput}

The throughput pattern is nonmonotonic. In the 1% fixed workload, concurrency 8 yields approximately 1,244–1,272 attempted requests/s, whereas concurrency 64 yields approximately 399–503. At concurrency 128, fixed-workload request errors occur at all three prevalences (120 at 1%, 36 at 0.1%, and 59 at 5%, each out of 50,000). Transformer-only has 74 errors at concurrency 64 and 1,612 at concurrency 128, with pooled p95 approximately 476.34 and 970.86 ms. Physical attempt fractions below one in those transformer-only groups reflect request/admission/failure behavior, not selective computation. All request failures remain in their original denominators. Admitted/completed/failed work and successful transformer scores are separately retained.

The five serialized live-shift runs reproduce the offline routing trace exactly, with 41,265 physical attempts among 43,505 measured requests (94.8512%), zero request errors and pooled p95=11.5816 ms. Their concurrency is one, not 64; their workload, denominator and original order differ from H3's fixed reference. This descriptive result establishes execution of the policy, not compliance with the primary latency gate or production safety.

Externally, both cascade and transformer fail certified FPR and Tranco safeguards. Each alerts on all 1,163 Tranco controls. These are label-free control alert rates, not labeled false-positive rates. The gold cascade-minus-transformer interval [0,0] satisfies the -0.02 noninferiority margin on 69 domains; it does not show equality in the target population or acceptable specificity. H3 has three passing and five failing gates and is not supported. RQ3 is answered by the joint constraints: low transformer use and a clean primary error count do not establish inline viability when external specificity and the designated latency budget fail.

## 4.5 Secondary Analyses and Interpretation

The supplement retains every declared model, not just favorable comparators: 21 internal and 22 external detector columns. Across the internal population and six external population definitions, there are 153 population–detector records. Full mixed-class metrics apply to 43 records (21 internal and 22 gold-plus-certified), including 430 calibration bins, 43 low-FPR score curves and 129 prevalence projections. Gold and certified remain separate confusion-count strata; silver and bronze receive source/tier-specific positive recall; Tranco receives only label-free alerts. Undefined single-class metrics are not filled in with invented AP, AUC or calibration. The source contract makes a PhiUSIIL source-only classifier unidentifiable without per-record provenance; no such classifier is claimed.

Table 4.8. Selected full-metric summaries. External AP and calibration describe the observed 69-positive/2,497-negative mixture, not production prevalence. All columns, precision, F2, MCC and balanced accuracy remain in secondary-metrics.csv.

{metric_table}

Internal Logistic-L1 AP is 0.995978, but external AP is 0.314882 and ECE is 0.879966. External transformer and policy ECE are approximately 0.9659. Even an external ranking statistic above chance cannot make the frozen operating threshold acceptable. The descriptive low-FPR score curves enumerate tied score cutoffs without interpolation; they are not newly selected deployment thresholds. For example, the structural external curve attains 20/69 positives with 24/2,497 false positives at a score cutoff near 0.999992. That retrospective curve must not replace the frozen operating result of 2,261 false positives.

Prevalence projections transport the measured class-conditional TPR and FPR to 0.1%, 1% and 5% hypothetical phishing prevalence. They are not new HTTP measurements. At 1%, the external fixed-cascade rates imply approximately 9,064 alerts per 10,000, including 8,964 false alerts and zero misses under that assumption. This arithmetic illustrates the cost of poor specificity; it neither validates the transport assumption nor establishes actual deployment prevalence. All originating rates and projection values are supplied.

Table 4.9. Additional positive tiers and label-free controls at frozen primary thresholds.

{tier_table}

The five-indicator formatting comparator attains internal recall 60.4216% with zero observed false positives, but external gold recall is zero and certified FPR is 1.0012%. This shows source-dependent predictive structure in a restricted representation; it is not a causal shortcut diagnosis. The fixed 100-tree Random Forest achieves internal recall 99.2042%, FPR 0.7006%, AP 0.996749 and AUC 0.996394. Externally, recall is 100% but FPR is 91.3496%, AP 0.036487 and AUC 0.634068. Its correction-v2 artifact and validation cutoff 0.2 are retained; it is a secondary benchmark, not a replacement primary model.

Table 4.10. Every accepted permutation comparator; no seed selection.

{permutation_table}

The permutation comparators do not establish successful negative controls. Their retained training-label reconstructions preserve class counts, but digests of the label vectors actually consumed by the original fits were not retained. Near-0.5 score means do not explain their variable ranking metrics. These observations establish neither leakage nor its absence and provide no permutation-test p-value. The limitation remains unresolved; fabricating a clean diagnostic would be stronger than the evidence. No new fit or test-informed comparator selection is used to repair it.

Table 4.11. Transformer seed/runtime sensitivity at each accepted secondary operating point. T denotes transformer. All five secondary cascades coincide with Logistic-L1 on these evaluated populations.

{seed_table}

Seeds 42–46 remain a single accepted family. Seed 42 used its historical training runtime; seeds 43–46 used the later pinned runtime, so their differences are not an isolated causal seed effect. Each had three logical band selections among 32,695 validation rows and zero selections in the internal and external evaluation streams. The full accepted cutoffs, bands, checkpoint records and development values are retained in complete-secondary-results.json. No best seed is promoted.

Table 4.12. Positive-only paired McNemar tables and four-slot Holm adjustment. “Underflow” means the saved floating-point computation returned 0.0, not an exact mathematical p-value of zero.

{mcnemar_table}

These nominal paired tests use aligned positive outcomes and do not model registrable-domain or routing dependence. They supplement, rather than decide, the primary clustered contrasts. A small p-value for structural recall does not cancel an external FPR failure.

Table 4.13. Accepted label-free development probe accounting.

{probe_table}

All four probe streams retain 16,370 row positions and 252 complete overlapping monitor windows. ASCII scheme/host uppercasing produces no binary detector transitions, but GMM alerts increase from 28 to 29 and MMD from 8 to 15, while PSI decreases from 11 to 10. Percent-escape uppercasing changes only 22 eligible records and produces no binary or monitor-alert transitions. It is therefore weak evidence for any broad invariance claim. First-literal path encoding changes 1,563 records: GMM-policy decisions change 0→1 for 105 and 1→0 for 63; length-only has 48 new alerts; transformer has 14; fixed cascade and Logistic-L1 have none. GMM alerts increase to 213, MMD to 14 and PSI decrease to 6. Full paired score deltas, all four decision-table cells, monitor transitions, fixed boundaries and eligibility counts are supplied in probe-decisions-and-scores.csv and probe-monitors-and-scores.csv. These streams have no inherited outcome labels and establish no accuracy, semantic-equivalence, adversarial-success or production-robustness rate.

## 4.6 Failed Attempts, Amendments and Evidence Limits

No failed research or execution result was deleted. The original baseline nonconvergence, provenance-incomplete diagnostic, platform scoring stop, transformer reconstruction mismatch, two secondary-development failures and seed/probe metadata stop remain in the preserved history supplement. Separately authorized corrections restored strict scoring and artifact checks rather than relaxing a scientific threshold. The accepted RF correction refitted only its declared RF; the seed/probe correction accepted all five retained stages without refitting and performed one authorized probe execution. An accepted correction does not relabel its failed predecessor as successful.

The September 28 raw-field preparation retained only 294 of 8,941 external test rows: 266 certified and 28 bronze, with no eligible gold or Tranco population. It recorded 8,553 invalid-URL and 94 overlap exclusions and triggered the whole-study hold before prediction. Aggregate diagnosis established that all 69 publisher gold and 1,241 publisher Tranco raw values lacked schemes. The operator separately authorized exact publisher url_norm for both parsing and model input, retaining original raw cells and all overlap, model, threshold and gate rules. This decision followed preparation and label-count exposure but preceded amended prediction; it is not fully pre-access prespecification or advisor approval.

The amended attempts then stopped after seven cells on AC loss, after fifteen on AC loss, during the third source evaluation on a detected competing test workload, and after 72 cells on AC loss. All four roots, partial cells and unattempted accounting remain unchanged. The separately reviewed checkpoint amendment followed source predictions and partial measurement. It selected only attempt four, required both source outputs and the entire accepted 72-cell prefix to qualify, and admitted 53 complete remaining cells in original order. The original stopped cell 73 was preserved, not resumed at the request level. No evidence from attempts one through three was substituted, and no accepted adverse measurement was repeated for a better outcome.

The final matrix spans two physical sessions and does not satisfy the original single-session intention. Descriptive group 71–75 spans that boundary. Different thermal state, caches, background activity and elapsed time may align with fixed workload order; the session labels do not permit their statistical separation. Sampled AC and workload checks establish recorded observations, not continuous absence of every disturbance. The designated primary cells remain the originally prescribed observations. Repeated source scoring and the documented September 3 and September 9 broad-search exposures also prevent a claim of a lifetime-unseen, single-pass test set. The unchanged frozen model-selection boundary is preserved without erasing those exposures.

## 4.7 Complete Hypothesis Adjudication

Table 4.14 contains every primary gate. For a contrast row, “Observed operand” is the lower endpoint of its clustered interval, not its point estimate; Table 4.3 gives both. Rate numerators, denominators and confidence bounds are in Table 4.2 and the accompanying primary-gates.csv. All decisions use full-precision values; rounded displays never determine passage. The H3 latency and error denominators are the designated five primary runs, not all 125 cells.

Table 4.14. All 22 primary hypothesis checks under unchanged decision rules.

{gates}

H1: 10/10 components measured, five pass, five fail, not supported. H2: 4/4 measured, one passes, three fail, not supported. H3: 8/8 measured, three pass, five fail, not supported. Thus nine gates pass and thirteen fail. “Not supported” is the prespecified conjunction outcome, not a claim that every component has failed or a formal rejection of every possible related model. Every promised primary question has measured evidence; explicit secondary identifiability and audit limitations remain limitations, not fabricated completed tests.

# Chapter 5—Discussion and Conclusions

## 5.1 Direct Answers to the Research Questions

RQ1: What incremental value do structural URL features and character-level representations provide under registrable-domain-disjoint and external evaluation?

The 25-feature structural model improves recall over length alone by 64.07 percentage points internally and 76.81 points on external gold positives, with clustered intervals excluding zero. Its internal FPR is 0.7747%. That success does not transfer to the certified external negatives, where FPR rises to 90.5487%. The fixed cascade selects no evaluation rows for character inference and adds no recall over Logistic-L1. Transformer-only offers a descriptive internal recall gain but has 99.2791% certified external FPR. Accordingly, H1 is not supported: structural recall value is real within the evaluated contrasts, while the joint low-FPR and incremental-cascade claim fails.

RQ2: Can GMM-based monitoring detect an external source/domain shift and guide escalation without exceeding the low-FPR operating constraint?

The monitor detects the specified external departure in 115/132 windows, but the original audit false-alert fraction is 28/252 rather than <=5%. Future-only routing invokes the transformer for 94.8512% of the external stream, adds no gold recall and increases certified false positives by 218. H2 is not supported. An unsupervised P(X) alarm therefore cannot be treated as an instruction to trust the more expensive detector: on this study's frozen data and policy, distribution sensitivity and useful low-FPR remediation are distinct requirements, and only the first external-window gate passes.

RQ3: What detection, escalation, throughput, and latency tradeoffs determine whether the fixed cascade is viable inline?

The reference cascade performs zero transformer calls among 10,000 requests and the designated concurrency-64 runs return zero errors among 50,000. Yet pooled p95 is 364.30 ms, not <=200 ms, and both systems violate certified-negative and Tranco safeguards. H3 is not supported. The specified implementation is not demonstrated viable inline under the promised joint gates. The complete matrix identifies service cost and reliability limits without treating low model-call count, a favorable low-concurrency run or gold recall alone as deployment readiness.

## 5.2 End-to-End Argument and Contribution

The evidence supports one connected argument. A detector can learn strong structural discrimination within a domain-separated development corpus while failing to maintain a usable operating point on a different source. A validation-minimized cascade can then become effectively identical to its first stage, producing impressive call savings without adding the intended detection value. A monitor can recognize the external distribution but route nearly all traffic into a model whose external specificity is worse. Finally, the full HTTP path can exceed a latency budget even when transformer inference is absent. These observations connect data representation, model selection, policy behavior and service execution rather than leaving four unrelated metric tables.

The contribution is a bounded empirical systems evaluation: common, unchanged decision rules applied to internal and external strata, explicit future-only routing, physical rather than simulated inference counters, and all five repeats across 25 workload groups. It supplies a reproducible case in which the components' apparent individual advantages do not compose into a passing system. The operational and statistical artifacts allow a reviewer to trace each conclusion back to denominators, saved decisions and observed service work. This contribution does not require claiming that negative findings are novel discoveries in themselves.

The literature already documents dataset bias, cross-source degradation, calibrated cascades and security drift monitoring (Rashid et al., 2024; Tsai et al., 2024; Li et al., 2021; Yang et al., 2021). The results are consistent with those concerns rather than a refutation of them. Ahamed et al. (2026), Alajaji (2026) and Hussain et al. (2027) address related representation, deferral or fusion settings, but their tasks and procedures do not supply a passing result for the present frozen policy. The present contribution is the specific linked evaluation and measured constraints, not invention of its components or an unsupported “first-ever” claim.

The study does not claim to test the superseded June/August 6 distillation or feature-fusion proposals. The active questions are those in the August 20 and September 3 decks and carried forward on September 17. No 300M-to-30M distillation, 92% performance retention, five-point AUC improvement or 40% latency reduction is asserted from these measurements.

## 5.3 Implications for Practice

Validation feasibility is not external safety. Before using a URL detector for inline alerting, an operator needs independent negative-source evidence at the actual intended operating point. Gold-positive recall alone can be perfect while nearly every trusted external URL is alerted. Tranco controls provide an additional alarm about generality, but popularity must not be relabeled as verified benign truth.

Selective inference should be evaluated jointly with the detection benefit it preserves. A zero-invocation cascade needs examination rather than celebration: here the band excluded every evaluation record, so the character model could not supply incremental recall. Similarly, routing expansion should be justified by observed error improvement under its specificity constraint, not merely by a high shift-alert rate. The policy measured here fails that requirement.

Service budgets require end-to-end measurement. Queueing, validation, transport and serialization remain when the expensive model is not invoked. Closed-loop throughput, terminal latency, service-side attempts and timeout/drain behavior answer different questions and should be reported separately. The observed matrix bounds this implementation on this host; it is not a production capacity guarantee or evidence for automatic blocking. The current frozen system should not be presented as deployment-ready under the tested gates.

## 5.4 Limitations and Threats to Interpretation

First, the source labels are reference classifications, not independent contemporaneous forensic judgments. Only 69 external gold-positive domains support the primary external recall contrasts, while 2,497 certified negatives are concentrated in 234 domains. Degenerate [0,0] recall-difference intervals on identical predictions do not establish certainty over unseen domains. Exact binomial rate bounds do not eliminate clustering. The gold-plus-certified mixture is deliberately descriptive, not a deployment prevalence estimate.

Second, domain-disjoint allocation prevents domain sharing across fitting and internal evaluation, but not collection-source shortcuts. The formatting comparator and severe external degradation are consistent with source dependence, without identifying its cause. A source-only classifier is unidentifiable from PhiUSIIL's available per-row provenance. Permutation comparisons retain incomplete consumed-label audit evidence and unexplained ranking variation. They cannot certify absence of leakage. Seed 42's historical runtime differs from the later seed family, so seed/runtime variation is not causally separable.

Third, the publisher-url_norm amendment followed preparation and label-count exposure, and checkpointed recovery followed predictions and partial execution. Both are disclosed departures from the original timing/session intentions. Repeated authorized source scoring and earlier search exposures prevent an unseen-test or one-lifetime-pass claim. Frozen weights and thresholds restrict further adaptation but do not erase that history. The excluded attempts, original hold and stopped requests remain preserved alongside accepted observations.

Fourth, P(X) monitoring cannot establish harmful concept drift or causal improvement. Windows overlap; shared policy activations introduce dependence beyond individual domains. Bootstrap contrasts are conditional on the realized stream, and no routing is rerun within a bootstrap. The single published order is not a distribution of production stream orders. Label-free probes neither inherit correctness labels nor prove semantic preservation, adversarial resistance or real-world robustness.

Fifth, the operational experiment uses one host and a fixed workload order spanning two accepted sessions. Thermal/cache/time effects may align with workload and concurrency. Sampled condition checks cannot prove continuous exclusivity. Closed-loop client timing omits unsent backlog and does not simulate all production arrivals. Successful process exits establish execution integrity, not passing scientific hypotheses. The saved-evidence recomputations are procedural checks, not external replication.

## 5.5 Future Research, Separate from This Study

The completed results motivate a new, separately designed study rather than retrospective repair of these hypotheses. Priorities are independently adjudicated external negatives and positives with provenance and timestamps; representation-sensitive source diagnostics; calibration transfer under a declared deployment population; and a routing rule whose escalation is justified by demonstrated conditional error benefit. Any new threshold, band, model or source must be selected without using a future untouched evaluation set.

Operational follow-up should separate session effects from workload order, examine open-loop arrivals and measure the implementation's queueing and serialization costs. These are prospective research opportunities, not missing measurements to be silently filled into the present protocol. The complete current matrix and adverse outcomes remain the baseline against which a genuinely new system could be tested.

## 5.6 Conclusion

All three research questions have direct measured answers, and all 22 primary checks are complete. H1, H2 and H3 are not supported. Structural recall improvement within the internal study does not establish an externally acceptable detector; external shift detection does not establish beneficial escalation; and low inference counts do not establish end-to-end inline viability. The dissertation's result is an auditable account of those joint limits, supported by the full operational matrix, source-specific analysis and preserved methodological history. The completed evaluation provides a reproducible basis for testing future systems against detection, routing and operational constraints together.

## References

{references}'''
    methods = methods.replace("## 3.2 Outcome-Label Contract", """### 3.1.1 Architecture and Information Flow

The implementation can be understood as four connected planes: data preparation, inference, monitoring and evaluation. Preparation determines which records may enter the study and preserves their source identities. Inference maps a permitted URL to a score and an allow/alert decision. Monitoring observes the feature distribution and can change future routing, but cannot rewrite an outcome label or a previous decision. Evaluation joins saved decisions with the declared outcome strata and separately measures the HTTP service. This separation makes it possible to ask whether a failure arises in the tested operating behavior without confusing a model score, a routing choice and an externally assigned label.

Figure 3.1 shows the frozen dataflow. Each URL produces 25 structural features and a Logistic-L1 score. The fixed uncertainty rule either retains that decision or requests the character transformer. The GMM observes the structural features plus the first-stage score in complete windows. An alert affects only the next 256 requests; it does not rerun the window that caused the alarm. Internal and external source evaluation saves paired outputs, while HTTP replay measures actual service execution. The diagram describes the implemented paths, not a newly proposed system or a claim that any component passes its operating gate.

![Frozen inference and evaluation dataflow](gwu-system-dataflow-20261001.png)

Figure 3.1. Frozen inference, future-only routing and separate evaluation boundaries.

Table 3.1. Reading map from technical component to experiment and evidence.

| Plane | Technical operation | What its evidence establishes |
|---|---|---|
| Preparation | Preserve source fields; apply declared parsing, quarantine, domain allocation and overlap checks | Which records and domains enter each denominator; not independently adjudicated labels |
| Representation | Length-only, 25-feature Logistic-L1, character transformer and fixed cascade | Paired recall differences and specificity at unchanged thresholds; RQ1, Tables 4.2–4.3 |
| Monitoring and policy | Training-fixed GMM; validation-calibrated windows; next-256 escalation | Distribution sensitivity, reference false alerts and the policy's separate error consequences; RQ2, Tables 4.4–4.5 |
| Service | Real HTTP client/service processes; physical inference attempts; terminal outcomes and latencies | End-to-end cost, failures and throughput for every scheduled repeat; RQ3, Tables 4.6–4.7 |
| Verification | Bound artifacts, retained-cell eligibility, process exits and saved-evidence recomputation | Traceability from accepted execution to all 22 decisions; not independent replication or institutional approval |

The decisive information-flow restriction is that evaluation strata do not select routing. After mechanical validity and conflict exclusions, every retained external record participates in the same ordered stream; gold, certified and other outcome groups are applied to its saved outputs afterward. Thus a reported certified-negative error rate answers how the common policy treated that stratum, rather than how a separate source-aware policy performed. The distinction is especially important here because source composition and operational behavior differ sharply from the internal evaluation.

## 3.2 Outcome-Label Contract""")
    chapters = chapters.replace("## 5.3 Implications for Practice", """### 5.2.1 Technical Value of the Completed Investigation

The central value is a connected explanation of why a plausible cascade did not meet its intended operating requirements, backed by measurements that distinguish its component behaviors. This is more informative than an isolated adverse accuracy result. The study shows where an apparent benefit is real, where it stops applying, and which additional requirement invalidates a deployment claim. The structural model's recall improvement is measured with paired domain-clustered uncertainty; its lack of external specificity is measured on a distinct negative source; and the service's cost is measured through actual HTTP requests rather than inferred from model complexity. These are three different evidentiary contributions, even though they lead to a common decision against the tested joint claim.

Table 5.1. Technical contributions, supporting observations and reusable value.

| Contribution | Evidence in this study | Value to a systems researcher |
|---|---|---|
| Joint feasibility evaluation | All 22 unchanged checks completed; 9 pass and 13 fail | Exposes precisely which conditions prevent a favorable component result from establishing a passing system |
| Representation-to-operation comparison | Large structural recall gain; no fixed-band selections; severe external FPR | Separates useful discrimination from a viable operating threshold and from incremental cascade value |
| Monitoring-to-action evaluation | 115/132 external alerts; 28/252 audit alerts; 218 added certified false positives after routing | Tests whether detecting distribution change leads to useful action instead of treating alarm sensitivity as remediation |
| Physical service evidence | 125 cells, 25 five-repeat groups, actual attempts and pooled request latencies | Distinguishes model-call economy, end-to-end response cost, errors and throughput under common workload rules |
| Auditable completion and recovery | 72 eligible retained cells plus 53 new cells, with interruptions and amendments preserved | Makes acceptance and aggregation inspectable without selecting favorable repeats or erasing failed predecessors |

For representation research, the important finding is not simply that the external dataset is difficult. Internal Logistic-L1 recall reaches 98.6807% at 0.7747% observed FPR, yet the unchanged classifier alerts on 2,261 of 2,497 certified external negatives. Gold recall alone would conceal this problem: all 69 gold positives are detected. Reporting both strata at the same locked threshold demonstrates that a favorable positive-class result can coexist with an unusable negative-class operating point. The secondary AP, calibration and low-FPR score analyses make the distinction between ranking and threshold behavior visible without retrospectively replacing the operating threshold. The study does not establish which collection artifact causes the gap; it establishes that this internal success is insufficient evidence for the external operating claim.

For cascade research, the zero-selection result reveals a specific failure of the intended mechanism. The character stage exists and can be invoked, but the validation-selected fixed band selects none of the 34,593 internal or 8,701 external evaluation records. The cascade therefore reproduces the first stage instead of adding a conditional correction. Its zero physical transformer calls in the designated reference cell are genuine, but they cannot be interpreted as retaining a demonstrated incremental character-model benefit. Measuring routing membership alongside paired decisions makes this degeneracy observable. A study that reported only call reduction or noninferior gold recall could miss it.

For adaptive systems research, the study separates three propositions often compressed into one: the input distribution changed, the monitor reliably distinguishes that change from its reference, and escalation improves the decisions. The external GMM window gate passes, the original reference false-alert gate fails, and the downstream policy adds false positives without gold recall gain. Expanding transformer use to 94.8512% of the external stream is therefore not evidence of successful adaptation. The preserved future-only routing and the five matching live/offline traces establish what the policy executed; they do not turn its adverse decision effect into a software malfunction or a favorable scientific result. The useful contribution is a concrete evaluation of the action that follows an alarm.

For service engineering, the work replaces a proxy cost argument with a measured one. Even when the fixed cascade makes no transformer calls, the primary pooled p95 is 364.3004101 ms at concurrency 64. Model-call avoidance is consequently insufficient to establish the 200-ms service target. The matrix also records transformer-only errors at higher concurrency and separates terminal latency, client-phase throughput, physical attempts and drain time. These observations do not identify a causal bottleneck: the study does not isolate queueing, runtime scheduling, validation or transport in a factorial experiment. They do establish that an end-to-end budget cannot be certified from inference counts alone.

### 5.2.2 Engineering Work and Reusable Research Artifacts

The implementation effort has value because it supports inspectable scientific statements, not because complexity or elapsed effort substitutes for evidence. The project includes deterministic source preparation, domain-disjoint allocation, training-only fitting, validation-only operating-point selection, two-stage inference, windowed monitors, ordered future routing, a real-HTTP measurement harness and reducers that retain every required numerator and denominator. These components are joined by explicit artifact and execution identities. The reader can move from a research question to a table, from the table to an aggregate file, and from that file to its accepted source and run lineage. Appendix A provides that navigation map.

The full five-repeat matrix is also an artifact rather than a collection of best-case demonstrations. All 125 cells contribute in their prescribed positions, including adverse runs. They contain 1,243,505 measured client requests: 1,200,000 across the 120 fixed/transformer cells and 43,505 across five live-shift repeats, excluding warmup. All 1,901 request errors remain in the full matrix; that descriptive total does not replace the designated 50,000-request primary denominator. Request-level evidence supports pooled quantiles; actual process exits distinguish installed summaries from completed workers; and physical attempt counters distinguish route selection from performed inference. Recovery accepted a complete eligible prefix and ran the remaining complete cells without splicing partial requests. These choices make the completed two-session investigation auditable while leaving the session limitation explicit. A reviewer need not accept a success flag or a narrative assurance in place of the underlying accounting.

What can be reused is the evaluation structure, implementation and documented evidence relationships. A subsequent study can state a new operating population and fit different detectors while preserving the separation between model selection, routing, labeled evaluation and service measurement. It can also test whether alternative calibration or routing resolves the specific failures observed here. The present result is not that all URL cascades must fail, or that the component algorithms are novel. It is that this particular fixed combination has now been fully evaluated against its promised requirements, and its behavior identifies concrete targets for a genuinely new investigation.

## 5.3 Implications for Practice""")
    appendix = """

# Appendix A—Evidence and Reproduction Map

This appendix is a reader's guide to the accompanying aggregate supplement. File names identify the package's aggregate-data directory unless a different directory is stated. The package exposes computed results and verification scope without redistributing row-level URLs, predictions, model weights or private execution capabilities. Aggregate files support inspection and reanalysis of the reported summaries; repeating model fitting or HTTP measurement also requires the original controlled sources and environment.

Table A.1. Research questions, complete evidence products and reading order.

| Question or concern | Read in the manuscript | Aggregate or provenance artifact |
|---|---|---|
| Exact commitments and hypotheses | Sections 1.3–1.4 | provenance/advisor-deck-scope-crosswalk-20261001.md; primary-gates.csv |
| RQ1 representation comparison | Sections 3.6–3.8, 4.2, 5.1 | primary-results.json; paired-contrasts.csv; secondary-metrics.csv |
| RQ2 monitor and routing effects | Sections 3.9–3.11, 4.3, 5.1 | external-monitor-windows.csv; external-psi-features.csv; complete-secondary-results.json |
| RQ3 service tradeoffs | Sections 3.12, 4.4, 5.1 | operational-runs.csv (125 rows); operational-groups.csv (25 rows) |
| Every primary decision | Section 4.7 | primary-gates.csv (22 rows); primary-results.json |
| Ranking, calibration and prevalence | Section 4.5 | low-fpr-score-curves.csv; calibration-bins.csv; prevalence-projections.csv |
| Controls, sources, seeds and probes | Sections 4.5, 5.4 | source-contingency.csv; seed-logical-invocations.csv; probe-decisions-and-scores.csv; probe-monitors-and-scores.csv |
| Integrity and methodological history | Sections 3.13, 4.6 | verification.json; secondary-verification.json; provenance/Historical_Evidence_Supplement_2026-10-01.md |

The measured academic code revision is 77d128377ce5b401437d7179f5cd78fb4294b72c in the automated-phishing-detection-public repository. The package's provenance directory includes the frozen contracts and environment identities; its analysis-scripts directory includes the actual verifier, secondary exporter and synthesis sources. These scripts retain their controlled-workspace dependencies and are not represented as a standalone substitute for the research inputs. The README and data dictionary identify units, population restrictions and undefined fields. SHA256SUMS.txt verifies package byte identity; it is not independent replication or proof that every scientific assumption holds.
"""
    final = methods + chapters + appendix
    (HERE / "Tallam_Krti_Praxis_v3_Body_Working.md").write_text(final)
    (HERE / "Tallam_Krti_Praxis_v3_Body_Results_2026-10-01.md").write_text(final)
    supplement = "# Preserved development and interrupted-execution history\n\nThis is the pre-synthesis October 1 record, retained verbatim as historical evidence. Its pending-status language describes that earlier revision and is superseded only by the separately verified final results. Failed roots and prior artifacts remain unchanged.\n\n" + original.split("# Chapter 4—", 1)[1].split("# Chapter 5—", 1)[0]
    (HERE / "Historical_Evidence_Supplement_2026-10-01.md").write_text(supplement)
    print(json.dumps({"manuscript_words": len(final.split()), "operational_groups": len(groups), "gates": len(gate_rows), "contrasts": len(contrasts), "probe_detector_rows": len(probe_decisions), "probe_monitor_rows": len(probe_monitors)}, indent=2))


if __name__ == "__main__":
    main()
