"""Integrate verified detection findings into a separate, explicitly working copy."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "followup-20261001"
ORIGINAL = HERE / "Tallam_Krti_Praxis_v3_Body_Results_2026-10-01.md"
OUTPUT = ROOT / "manuscript-work/Tallam_Krti_Praxis_Engineering_Working_2026-10-01.md"


def extract(text, heading):
    marker = f"## {heading}\n"
    if text.count(marker) != 1:
        raise ValueError(f"Ambiguous section: {heading}")
    return marker + text.split(marker, 1)[1].split("\n## ", 1)[0].rstrip() + "\n\n"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Ambiguous replacement: {old[:100]}")
    return text.replace(old, new, 1)


def compose():
    verification = json.loads((ROOT / "verified-detection-v1/verification.json").read_bytes())
    if verification["status"] != "verified" or verification["records"] != 8622 or verification["D_requirement_met"] is not False:
        raise ValueError("Unexpected or unverified detection evidence")
    for model, expected in [("baseline", (4619, 3940, 31, 32)), ("candidate", (4435, 3689, 282, 216))]:
        counts = verification["metrics"][model]["counts"]
        if tuple(counts[key] for key in ("tp", "fp", "tn", "fn")) != expected:
            raise ValueError("Narrative counts do not match verified evidence")
    body = ORIGINAL.read_text()
    additions = (ROOT / "manuscript-additions-v1.md").read_text()
    additions = additions.replace("2021a", "2021TEMP").replace("2021b", "2021a").replace("2021TEMP", "2021b")
    additions = replace_once(additions, "### 3.15.3 Transport-Neutral Structural Representation", """Figure 3.2 separates the roles of development, diagnosis and additional evaluation. Previously exposed evaluation outcomes inform the mechanism-directed design; the unchanged training and validation partitions supply coefficient estimation and threshold selection. Retained full-source domain manifests are used only for overlap screening of the additional benchmark. They are not new fitting observations. The admitted benchmark supplies the final paired comparison and does not select the candidate or its operating point.

![Follow-up source partitions](followup-source-partitions.png)

Figure 3.2. Source and partition roles in the bounded detection comparison. Arrows show permitted information flow; domain exclusion does not establish temporal or source-mechanism independence.

### 3.15.3 Transport-Neutral Structural Representation""")
    for before, section in [
        ("# Chapter 3—Methodology", "2.8 Evaluation-Guided Engineering Iteration"),
        ("# Chapter 4—Results", "3.15 Bounded Follow-up Design and Evaluation"),
        ("## 5.3 Implications for Practice", "5.2.3 What the Engineering Iteration Adds"),
    ]:
        addition = extract(additions, section)
        if section.startswith("5.2.3"):
            addition = addition.replace("## 5.2.3", "### 5.2.3", 1)
        body = replace_once(body, before, addition + before)
    results = extract(additions, "4.8 Diagnostic Investigation and Detection Comparison")
    results = replace_once(results, "### 4.8.3 Representation Invariance and Decision", """![Follow-up paired recall](followup-paired-recall.png)

Figure 4.1. Candidate-minus-comparator recall difference at frozen thresholds; 97.5% domain-cluster interval.

![Follow-up calibration](followup-calibration.png)

Figure 4.2. Fixed-bin calibration and bin populations on the admitted benchmark. Small populated bins can have extreme fractions; empty bins have no reliability point. The class mixture is not deployment prevalence.

### 4.8.3 Representation Invariance and Decision""")
    service_status = """## 4.9 Controlled Service Comparison: Working-Copy Status

Service comparison S has not yet been measured. Its implementation and fixed 80-arm schedule are specified in Section 3.15.5, but controlled timing awaits verified AC power and completion of competing computational work. No diagnostic timing is substituted for that comparison. There is no measured S decision, and this working copy is not the final extension package. The completed initial study and its 125-cell results remain available in the preserved October 1 package.

"""
    body = replace_once(body, "# Chapter 5—Discussion and Conclusions", results + service_status + "# Chapter 5—Discussion and Conclusions")
    body = replace_once(body, "### 1.3.1 Thesis Statement", """A bounded follow-up extends the investigation from evaluation to mechanism-directed modification. It tests a scheme-neutral structural representation on an additional admitted benchmark and specifies a separate client-topology comparison. This extension adds a measured representation property and an explicit comparison of its detection tradeoffs; it does not rename the original hypotheses. The technical account follows design, evaluation, diagnosis, modification and measured comparison. The controlled service comparison remains pending in this working copy.

### 1.3.1 Thesis Statement""")
    original_contribution = extract(body, "1.5 Contribution Boundary")
    new_contribution = """## 1.5 Contribution Boundary

The contribution is an implemented raw-URL inference-gateway artifact and a connected empirical investigation of its representation, routing and service behavior. The initial evaluation joins structural and character inference, future-only GMM-guided routing, domain-separated outcome strata and real HTTP measurements under common operating requirements. This makes the relationships among detection quality, escalation cost, tail latency and request reliability inspectable in one system rather than in unrelated accuracy and timing demonstrations.

The engineering iteration adds mechanism-level evidence. A consistent transport-neutral feature transformation removes scheme-spelling sensitivity, and a separately specified benchmark comparison measures the resulting detection tradeoff. A no-model service control and paired client-topology design isolate an additional transport hypothesis, with final measurements still pending in this working copy. Representation conformance, external risk and service cost are evaluated as separate requirements; passing one is not substituted for another.

Prior work establishes structural and character URL models, domain-aware evaluation, selective cascades, source-shift analysis and security monitoring (Ahamed et al., 2026; Hussain et al., 2027; Li et al., 2021; Rashid et al., 2024; Tsai et al., 2024; Yang et al., 2021). This study does not claim invention of those components or of scheme normalization and connection reuse. Its technical value is the specific integrated artifact, controlled comparisons and traceable evidence that connect a diagnosed behavior to an implemented change and a measured engineering decision.

"""
    body = replace_once(body, original_contribution, new_contribution)
    answer_marker = "RQ2: Can GMM-based monitoring detect an external source/domain shift and guide escalation without exceeding the low-FPR operating constraint?"
    discussion_start = body.index("# Chapter 5—Discussion and Conclusions")
    before, discussion = body[:discussion_start], body[discussion_start:]
    discussion = replace_once(discussion, answer_marker, """The bounded follow-up sharpens the representation answer. Scheme-neutral inputs achieve exact feature and score invariance for all 8,622 benchmark pairs, eliminating 481 comparator decision flips under the prescribed companion transformation. At the frozen thresholds, however, candidate recall falls from 99.31% to 95.36% and FPR remains 92.90%, compared with 99.22% for the comparator. D is not supported. The contribution is a verified representation correction and measured tradeoff, not evidence that scheme removal alone achieves low-FPR source transfer.

""" + answer_marker)
    old_argument = discussion.split("## 5.2 End-to-End Argument and Contribution\n\n", 1)[1].split("\n\n", 1)[0]
    new_argument = """The investigation connects requirements to an artifact, measures its behavior, diagnoses specific mechanisms and evaluates a bounded modification. Structural features first demonstrate substantial recall value over length alone under domain-disjoint evaluation. Explicit routing traces then show what the character stage contributes, while separate monitor and policy outcomes establish whether an alarm leads to useful action. Real HTTP measurements expose costs that inference-call counts do not describe. The representation follow-up closes a further loop: the code removes the intended scheme dependency exactly, and the additional benchmark establishes both the achieved stability and the remaining low-FPR gap. This is one engineering argument about what each layer can and cannot establish, supported by complete comparators and denominators."""
    discussion = replace_once(discussion, old_argument, new_argument)
    discussion = discussion.replace("a specific failure of the intended mechanism", "the distinction between the optimized objective and the intended mechanism", 1)
    discussion = replace_once(discussion, "## 5.4 Limitations and Threats to Interpretation", """The follow-up representation contract can be reused wherever publishers supply different HTTP(S) spellings, but invariance is not a substitute for labeled specificity evidence. On this benchmark, fewer false alerts coexist with fewer detected positives, and the 1% target remains unmet. Deployment decisions must therefore require both a stable input contract and a validated operating point in the intended population.

## 5.4 Limitations and Threats to Interpretation""")
    discussion = replace_once(discussion, "## 5.5 Future Research, Separate from This Study", """Sixth, the additional benchmark is retrospective and publisher-labeled, shares upstream feed mechanisms, and has incomplete exact collection-month agreement between metadata and the inspected preprint. Excluding observed domains does not establish temporal or source-mechanism independence. The candidate is trained and calibrated on the same development partitions used earlier; it is a separately evaluated modification, not untouched model development. Metamorphic scheme companions test representation behavior, not semantic preservation for live websites. D's domain-cluster intervals are conditional on the admitted benchmark. S remains unmeasured in this working copy and supports no performance or agreement claim.

## 5.5 Future Research, Separate from This Study""")
    discussion = discussion.replace("representation-sensitive source diagnostics; calibration transfer", "diagnostics for the source differences remaining after scheme-neutral representation; calibration transfer", 1)
    old_conclusion = extract(discussion, "5.6 Conclusion")
    conclusion = """## 5.6 Conclusion

The completed initial investigation answers all three promised research questions through 125 operational cells, 25 workload groups and 22 primary checks. Nine component checks pass, including structural recall contrasts, internal specificity, external-window monitoring sensitivity, call economy and primary request reliability. The original H1–H3 conjunctions are not supported. Those decisions identify the unmet joint requirements; they do not erase the implemented system or the demonstrated component benefits.

The engineering iteration advances from those findings to a concrete modification and comparison. Consistent scheme neutralization achieves exact representation and prediction invariance across 8,622 paired benchmark inputs. Its frozen operating point yields 251 fewer false positives and 184 fewer detected positives than the unchanged comparator, leaving D's low-FPR and recall-gain requirements unmet. This separates a successful representation contract from a successful external detector and provides a reusable, quantitatively evaluated design result. The controlled service comparison remains pending; this working copy makes no final S claim.

The contribution is the connected artifact and evidence: a reader can follow the operating requirement, the model or policy that implements it, the measured outcome, the diagnosed mechanism and the effect of a specified change. This supports concrete engineering judgment without claiming algorithmic priority or universal deployment readiness. The final extension package must add the controlled service evidence and complete its cross-document verification before it is labeled complete.

"""
    discussion = replace_once(discussion, old_conclusion, conclusion)
    body = before + discussion
    existing_references = body.split("## References\n\n", 1)[1].split("# Appendix A", 1)[0].strip()
    new_references = extract(additions, "New References").split("\n", 1)[1].strip()
    combined = sorted((existing_references + "\n\n" + new_references).split("\n\n"), key=str.casefold)
    body = replace_once(body, existing_references, "\n\n".join(combined))
    body += """

The bounded follow-up has a distinct source identity, ef8ba5f0b357cf3dd60c4d663e6297d13334460c. Its prospective comparison specification, execution manifest, model-freeze record, population admission and saved-prediction recomputation are retained under followup-20261001. The aggregate detection supplement includes two model rows, twenty calibration-bin rows, two invariance rows and a domain-size distribution. The separate verifier uses a second arithmetic implementation for counts, rank-based AUC, average precision, Brier score and domain-weighted bootstrap summaries; this is not external replication. Candidate fitting used Python 3.10.19, NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.7.2. The source freeze recorded 215 focused passing tests and an in-progress broader regression suite; no independent reviewer approval is claimed.
"""
    body += """

The subsequently completed broad regression run records 13,016 passed, three failed and two skipped tests. All three failures occur before the intended SIGINT injection because a local test double omits the async aclose method required by the HTTP client lifecycle. A separately retained fixture-corrected copy adds only that method; all six original interruption assertions pass, including the three shift cases that already passed. The frozen measurement source and original test remain unchanged. The original failure log is preserved and is not relabeled as a green full-suite run. The supplemental verification record also retains an initial configuration-selection error before test collection. These are software-verification outcomes, not new detection or service measurements.
"""
    return body


def main():
    original_hash = hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(compose())
    if hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() != original_hash:
        raise ValueError("Preserved manuscript changed")
    report = {"status": "working_copy_detection_verified_service_pending", "original_sha256": original_hash,
              "new_body_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
              "detection_verification_sha256": hashlib.sha256((ROOT / "verified-detection-v1/verification.json").read_bytes()).hexdigest()}
    (OUTPUT.parent / "integration-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
