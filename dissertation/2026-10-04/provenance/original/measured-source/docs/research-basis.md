# Research Basis

This note records the public source basis, the ownership of the study's
decision gates, and the boundary between established work and the proposed
evaluation. It separates development evidence from confirmatory hypothesis
results and does not treat a planned gate as having been met.

The September 3 advisor report, SHA-256
`b72da89a4cc8a5b06f6ca88d79fe78dd54e3199a96b7450209ea53b4a4c04215`,
directed the study to complete and freeze the source-provenance release, then
conduct systematic hypothesis testing with all gates, thresholds, features,
and train/validation procedures locked before test results, with particular
attention to H1 and the GMM. The public `phiusiil-development-v1` release
completed that requested source freeze after the meeting.

Protocol v1.7 froze `rq1-baselines-v2` before its later
`completed_development_validation` execution. Those operating points remain
development validation only. Protocol v1.8's byte-preserved
`rq1-transformer-cascade-v1` is `superseded_unrun`. Protocol v1.9 freezes
`rq1-transformer-cascade-v2`, SHA-256
`686c0d86b33b8a6c2e09cd6e174003db0bd2f7c30b087faf5470e6a270524213`, at
contract status `frozen_not_run`. Version 2 changes publication semantics but no
scientific or artifact-content rule. The implementation is unit-tested at
historical pre-execution status `frozen_implemented_not_run`; its first official
MPS execution stopped at a stage-one integrity check. The controlled retry then
completed development validation. Its accepted
[summary](../reports/rq1-transformer-cascade-v2-summary.json), SHA-256
`41499aa388babe60442de7231b4087f67a53f96f340568a7cc58a3268606a2fd`,
records transformer recall `0.9910299535479737` and cascade recall
`0.9842223290084895`. The cascade's logical mask selected escalation for
3 of 32,695 validation rows, and its recall was identical to `Logistic-L1`.
Calibration computed transformer scores for every validation row. This does not
decide H1 and does not establish measured HTTP savings. Reviewed verifier
commit `ef4e4567df979fef3afc91f8ad8097691f94d1ad` validated the named bundle
without fitting and without reading or scoring research rows.

The [current execution record](advisor-approval/approval-status.md) preserves
both attempts and the full artifact audit. H1 and H3 remain undecided, the group
test remains analyst-exposed but model-unscored, and no PhishVN record has been
accessed. Protocol v1.10 incorporates unchanged transformer v2 and freezes
[`rq2-gmm-development-v1`](../data/rq2-gmm-development-contract-v1.json),
SHA-256 `22d32088b05e74432704f9671ab76ba28b4f573ead418846b23bc366315cb393`,
before any GMM execution.
Its calibration/audit allocation hashes normalized ASCII domains under namespace
`rq2-gmm-validation-v1` and seed `20260816`, sorts by digest bytes then domain
bytes, and assigns the first `floor(D/2)` domains to calibration and the rest to
audit while preserving input row order. It uses neither labels nor class counts
and cannot be rerolled. Training-only scaling precedes all six diagonal-GMM
candidates; training BIC selects the component count with exact ties favoring
the smaller count. The calibration boundary uses NumPy's linear 0.95 quantile;
the independent gate is `20 * alert_windows <= complete_windows` for strict
`score > boundary` alerts. No overlapping-window binomial interval is reported,
and the audit cannot retune the boundary. Even a failed audit gate remains
development validation only. The recorded 28 alerts in 252 audit windows exceed
the mandatory 5% limit, so H2 is not supported. Remaining external detection
and routing work can characterize RQ2 but cannot rescue that conjunctive support
rule. Synthetic-only preflight, not research observations, established the
supported NumPy 2.2.6 `scipy-openblas` 0.3.29 runtime after Accelerate raised a
fatal warning. The GMM rejects unsupported backends before input reads and adds
no warning exemption.

## Decisions to Defend

| Choice | Purpose and limit |
|---|---|
| Raw-URL structural features | The 25 fixed features describe length, syntax, and character composition without fetching a page or using publisher fields. They provide a reproducible structural baseline, not a claim that this is an optimal feature set. Several features are dependent, so individual coefficients are not causal feature importance. |
| Fixed Logistic-L1 first stage | Reuse the accepted fitted model and operating threshold. Reconstruction must preserve the authoritative scoring procedure; a mathematically equivalent formula can still change floating-point ties. No new fit is needed to diagnose that implementation discrepancy. |
| Validation-selected fixed cascade | The development mask selected escalation for 3 of 32,695 validation rows and produced recall identical to `Logistic-L1`. Calibration scored every validation row with the transformer. That logical mask is not confirmatory H1 evidence or a measurement of transformer work saved over HTTP. |
| One through six GMM components | Fit every prespecified candidate on training data and choose minimum training BIC. This selects among those six candidates; convergence and BIC do not establish useful drift detection or acceptable audit alerts. |
| Separate calibration and audit domains | Use calibration to choose the boundary, then evaluate it on disjoint domains without labels, balancing, or rerolls in the allocation. Both streams come from the same development source; this is not an external or temporal evaluation. |
| Calibration percentile and audit gate | The 95th calibration percentile defines a boundary. The separate <=5% audit gate asks whether it transfers. A percentile selected on calibration does not guarantee the audit fraction. Overlapping windows do not supply independent trials. |
| Low-FPR and service limits | The limits below are study-defined operating requirements, not achieved results or universal literature standards. A failed required component remains failed even if another metric is favorable. |

The internal group-test partition is analyst-exposed but model-unscored, as
recorded in the execution history. Later results must disclose that limitation
rather than call it unseen. The private working manuscript also retains
protected, outdated front matter; updating its research body does not make the
whole document submission-ready.

## Source basis and limitations

The [UCI record for dataset 967](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+website+dataset)
identifies the PhiUSIIL Phishing URL (Website) dataset, its accompanying file,
and its CC BY 4.0 license. UCI links the dataset to Prasad and Chandra's
[publisher article](https://doi.org/10.1016/j.cose.2023.103545), which reports
134,850 legitimate URLs and 100,945 phishing URLs. UCI states the native label
semantics directly: `0` denotes phishing and `1` denotes legitimate. This
study preserves those cells and applies one versioned mechanical mapping:
native `0` -> `is_phishing=1`; native `1` -> `is_phishing=0`. It does not
relabel a URL by personal judgment.

The publisher article reports that the legitimate class was drawn from Open
PageRank and the phishing class from PhishTank, OpenPhish, and MalwareWorld.
It reports a phishing-source retrieval window from 2022-10-01 through
2023-05-21. The public materials do not establish a collection window for the
legitimate class, so the phishing dates must not be applied to it.

These labels are publisher-provided, source-derived reference
classifications. The public UCI record, variable table, and released CSV do
not supply a per-row source name, retrieval timestamp, source-snapshot
identifier, or independent-adjudication trail. Consequently, this study can
test performance against the published reference classifications, but it
cannot represent each row as independently verified historical ground truth.
That limitation is a statement about the available provenance, not evidence
that a particular published label is wrong.

Manual review is permitted only as separately reported post hoc descriptive
error analysis and cannot assign or override labels, change quarantine or
inclusion, thresholds, features, model or procedure choices, gates, or
hypothesis decisions.

`data/sources.json` records these facts together with the exact archive and
CSV hashes. Its `phiusiil-development-v1` contract identifier remains stable
because the mapping and preparation algorithms remain version 1; source
schema version 2 adds provenance metadata without changing the input bytes or
row-processing rules.

The reserved external release is Mendeley Data repository Version 4, named
PhishVN v3.1.0. The publisher's
[release comparison](https://data.mendeley.com/datasets/compare/b97hxbxtpd)
describes a documentation-only update: the data files are byte-identical to
Version 3, while the datasheet adds the completed label-audit results. This is
not a corrected-data release or independent certification of every label. No
PhishVN record has been accessed.

## Study-defined decision gates

Every gate below is study-defined. Each is an operating constraint for this
study, not literature-prescribed or assumed to have been met. Its recorded
status must come from the frozen protocol before any hypothesis can be
supported.

| Gate | Prespecified use | Ownership and comparison limit |
|---|---|---|
| False-positive rate `<= 1%` | Maximum observed FPR at the stated internal and external decision points | Study-defined risk ceiling. Published FPR values use different data sources, class mixtures, splits, and thresholds. |
| Real-HTTP p95 latency `<= 200 ms` | H3 latency gate at concurrency 64 | Study-defined service target. Model-only or mean latency from prior work is not a real-HTTP p95 measurement under this harness. |
| Request errors `< 0.1%` | Errors among the 50,000 measured concurrency-64 requests | Study-defined reliability budget. Prior model evaluations generally do not use this request denominator or timeout treatment. |
| Shift detection `>= 80%` | Fraction of prespecified external shift windows alerted | Study-defined sensitivity target. Detection rates change with the shift, representation, window, and labeling rule. |
| False alerts `<= 5%` | Alert rate on the independent PhiUSIIL validation audit stream | Study-defined alert budget. A validation-calibrated percentile does not itself establish the held-out audit rate. |
| Recall noninferiority margin `-0.02` | Lower confidence bound for `recall(cascade) - recall(transformer)` | Study-defined tolerated loss, not a margin supplied by prior phishing studies. |
| Transformer invocation `<= 30%` | Maximum share of requests sent to the transformer in normal fixed-cascade replay | Study-defined resource budget. Cascade papers demonstrate selectable coverage-cost tradeoffs but do not prescribe 30%. |
| Client timeout `2000 ms` | Frozen request-harness timeout and error definition | Study-defined measurement rule, not a universal HTTP or phishing-detection standard. |

Values reported elsewhere are therefore not directly comparable to these
gates unless source composition, domain isolation, threshold selection,
hardware, batching, HTTP overhead, concurrency, warm-up, and timeout handling
also match. Prior measurements motivate what to test; they do not establish
compliance in advance.

For H2, every complete 256-request window of the retained external stream is a
prespecified external-shift window. The detection-rate numerator is windows
with score strictly greater than the boundary; the denominator is all such
complete windows. Overlapping windows count separately. An incomplete terminal
window is excluded from this rate, but its requests remain routable from a
prior alert. The independent validation-audit false-alert fraction uses the
same complete-window numerator and denominator rule. These definitions are
prospective. The September 17 GMM development run subsequently recorded
28 alerts in the audit's 252 windows (11.11%), above the 5% gate. The
calibration boundary was not retuned. Because false alerts are a mandatory
conjunctive component, H2 is not supported. This failed development gate is not
evidence that the monitor detects useful external shifts or improves routing.
The [aggregate record](../reports/rq2-gmm-development-v1-summary.json) preserves
the result; external detection and routing outcomes remain unmeasured. Those
later measurements can characterize RQ2 but cannot rescue H2 under the frozen
decision rule.
The [post hoc description](research-evidence-outline.md#post-hoc-audit-description)
compares saved window-score distributions and accounts for overlapping alerts.
It does not identify a causal explanation, select another threshold, or replace
the original result.

## Closest prior work and contribution boundary

The following primary sources constrain the contribution claim.

RQ1 asks: What incremental value do structural URL features and selective
character-model escalation provide under registrable-domain-disjoint and
external evaluation? H1 retains two primary contrasts:
`recall(Logistic-L1) - recall(length-only)` and
`recall(cascade) - recall(Logistic-L1)`. The latter is a selective system
contribution, not a pure causal isolation of representation. Transformer-only
remains a comparator and operational reference, not a third primary H1 gate.

- [Ahamed et al. (2026)](https://doi.org/10.3389/fcomp.2026.1834407) jointly evaluate structural and character-based URL models, adversarial robustness, domain-disjoint behavior, explanation stability, and external data. This establishes that broad integrated URL-evaluation claims are already occupied.
- [ExpertFusion (2027)](https://www.sciencedirect.com/science/article/pii/S0957417426029957) combines calibrated semantic, structural, sequential, and lexical URL experts through confidence- and uncertainty-aware routing and evaluates cross-dataset distribution shift with registered-domain-stratified splits. It is direct prior work on calibrated URL expert integration under shift.
- [Alajaji (2026)](https://doi.org/10.3390/electronics15143051) evaluates a validation-selected classical-first selective cascade with transformer deferral for phishing email detection. Selective cascading and a classical-first ordering are therefore not inventions of this study.
- The [PhishVN Data in Brief article (2026)](https://doi.org/10.1016/j.dib.2026.113195) reports time-stamped source records, confidence tiers, and registrable-domain-grouped splitting for a phishing URL corpus. Domain grouping and source-aware external evaluation are not independently novel.
- [Rashid et al. (2024)](https://doi.org/10.1016/j.comnet.2024.110398) show cross-dataset degradation in phishing URL detection and evaluate unsupervised domain adaptation. [Tsai et al. (2024)](https://doi.org/10.1609/aaai.v38i19.30161) identify dataset bias in malicious-URL models and evaluate adversarial training for more invariant representations. Together they make source-shift generalization an established problem rather than a new premise.
- [CascadeBERT (Li et al., 2021)](https://aclanthology.org/2021.findings-emnlp.43/) uses calibrated cascades of complete language models to trade inference cost against prediction quality. [CADE (Yang et al., 2021)](https://www.usenix.org/conference/usenixsecurity21/presentation/yang-limin) detects and explains drifting samples in security applications. Model cascading and security drift detection are established components.

**Study inference.** The defensible contribution is the prospective joint
systems evaluation of a frozen structural-to-character-transformer URL
cascade whose future routing changes after a validation-calibrated GMM alert
under external source/domain shift, evaluated jointly against FPR,
transformer-invocation, real-HTTP p95, and request-error gates. The monitor's
alert is evidence of distributional departure; it does not by itself prove
harmful drift, concept drift, or causal performance deterioration. The study
does not claim invention of, or priority for, any individual model, cascade,
calibration method, domain split, GMM, drift detector, or metric.
