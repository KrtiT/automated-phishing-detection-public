# Manuscript sentence changes

Every changed prose sentence and table label/cell is recorded below. Sentence splits and joins are grouped so the complete before and after wording remains visible. “Added” means no corresponding text existed in the prior edition. Existing text outside these entries is preserved.

## 1. Abstract — paragraph 23

**Before 1:**

> This Praxis establishes an empirical design basis for cascaded uniform resource locator (URL) phishing detection through CyberSentinel, a system combining structural classification, a character transformer and distribution monitoring.
> Three research questions connect representation, monitoring and selective execution to detection quality and service cost.
> Paired cross-dataset comparisons, controlled input interventions and service measurements explain how these design choices affect system behavior.
> Structural features increase internal recall by 64.07 percentage points.
> Validation controls reveal a strong association between URL format and class labels, while fixed-weight interventions establish sensitivity to scheme spelling.
> Together, these findings identify representation dependence without assigning an unmeasured share of the internal gain to format.
> Paired model outputs establish a 90.5487% false-positive floor for routing between unchanged decisions on the evaluated external cohort, making the expert’s available corrections a measurable design constraint.
> Monitoring detects 87.12% of external windows, while escalation adds 218 false positives without additional gold-positive detections, demonstrating the importance of evaluating alerts through their resulting decisions.
> Controlled service experiments identify an effective implementation change: worker-owned clients reduce median paired 95th-percentile latency by 79.68% in single-host synthetic structural-only measurements, with a no-model control supporting the client-path explanation.
> The resulting contribution is concrete engineering guidance: evaluate representation stability alongside detection, select experts for useful corrections, measure monitoring through response outcomes, and target execution costs identified by controlled measurement.

**After 1:**

> This Praxis builds and tests CyberSentinel, a uniform resource locator (URL) phishing gateway combining structural classification, a character transformer, distribution monitoring and a Hypertext Transfer Protocol (HTTP) service.
> Internal structural recall reaches 98.6807% at a 0.7747% false-positive rate, a 64.07-percentage-point gain over length alone.
> Controlled experiments explain the system's behavior and test improvements.
> Validation labels are strongly associated with URL format, and scheme spelling changes frozen-model decisions; these tests do not quantify the format's share of the internal gain.
> On the evaluated external cohort, shared errors impose an exact 90.5487% false-positive floor on routing between unchanged decisions.
> Monitoring recognizes 87.12% of external windows, exceeding its sensitivity target, while audit false alerts exceed their limit.
> Escalation adds 218 false positives without another gold detection, establishing the need to evaluate alerts and subsequent decisions separately.
> In single-host synthetic structural-only service tests, worker-owned clients reduce median paired 95th-percentile latency by 79.68%, preserving every comparable prediction.
> A no-model control supports the client-path explanation.
> The complete gateway did not meet its prespecified joint hypotheses; the experiments identify the governing constraints and test concrete remedies.
> CyberSentinel contributes a working architecture, a measured service improvement, a benchmark-format check and an exact bound that guides expert selection.

## 2. 1.1 Background — paragraph 227

**Before 1:**

> The central engineering task is to determine which learned signals transfer and which interventions improve decisions.
> Domain separation tests performance beyond the training domains, while evaluation across sources examines the influence of shared collection conventions.
> Paired expert decisions identify where routing can help.
> CyberSentinel brings these questions into one evaluation of representation, routing and monitoring.

**After 1:**

> CyberSentinel makes three design choices testable: what information to extract from a URL, when to invoke an expert and how to respond to changed inputs.
> Domain-separated testing evaluates unseen domains; evaluation across sources probes collection conventions that domain separation can preserve.
> Comparing both models on identical inputs establishes which decisions the expert can improve.

## 3. 1.2 Problem Statement — paragraph 229

**Before 1:**

> The design challenge is to make those requirements work together: expert calls consume resources, and recall gains must justify their false-alert cost.
> This study evaluates a low-cost structural detector, selective character-model execution and distribution monitoring against joint detection and service constraints.
> That joint evaluation gives each component's contribution an operational meaning.

**After 1:**

> A cheap first model saves computation; an expert may recognize patterns it misses.
> A distribution monitor flags inputs that differ from the training reference.
> CyberSentinel tests these components together under joint detection and service constraints.
> Each must earn its place through better decisions or lower cost.

## 4. 1.2 Problem Statement — paragraph 230

**Before 1:**

> Selecting an effective intervention requires locating the behavior it should change.
> Collection-specific cues, threshold transfer and shared expert errors suggest different detection interventions; request handling presents a separate opportunity to reduce delay.
> Paired decisions and controlled comparisons distinguish these explanations in CyberSentinel.
> Chapter 3 defines the evaluated populations and the limitations of their labels.

**After 1:**

> The engineering question is which change addresses the observed behavior.
> Collection-specific cues, threshold transfer and shared model errors require different tests.
> Delay can also arise outside the classifiers.
> Paired decisions and controlled comparisons examine these possibilities separately.
> Chapter 3 defines the populations and reference-label limitations.

## 5. 1.3 Purpose of the Study — paragraph 234

**Before 1:**

> Each research question informs a design decision.
> Paired comparisons measure representation value; scheme interventions test format dependence.
> Monitoring replay and routing comparisons measure useful expert corrections.
> Shuffled-label controls explain validation ranking, while client and computation measurements locate execution costs.

**After 1:**

> Each research question leads to a design test.
> Paired model comparisons measure the feature gain; scheme interventions test format dependence.
> Monitoring replay follows alerts into decisions, and routing comparisons count expert corrections.
> Shuffled-label controls examine validation ranking.
> Client and computation tests locate execution costs.

## 6. 1.3.1 Thesis Statement — paragraph 236

**Before 1:**

> CyberSentinel connects URL representation, monitoring and selective execution to measured detection and service outcomes.
> Paired comparisons identify format dependence, quantify expert corrections and evaluate escalation.
> Controlled service measurements locate a client-path improvement.
> Together, these results provide an empirical basis for choosing and evaluating gateway designs.

**After 1:**

> CyberSentinel demonstrates how to evaluate a URL gateway as a connected system: measure the information its features supply, the corrections its expert can make and the cost of execution.
> The experiments reveal a benchmark-format dependency, derive an exact routing constraint and verify a client-path improvement.
> These findings guide representation, expert and service design.

## 7. 1.3.1 Thesis Statement — paragraph 237

**Before 1:**

> Chapter 4 evaluates those commitments and develops the diagnostic evidence needed to interpret their outcomes.
> Appendix C preserves the working thesis and methodological record.

**After 1:**

> Chapter 4 reports their outcomes and uses controlled comparisons to explain them.
> Appendix C retains the working thesis and methodological record.

## 8. 1.3.2 Research Objectives — paragraph 239

**Before 1:**

> The objectives are to quantify representation value across sources, evaluate monitoring through its subsequent decisions, and measure the detection and execution consequences of selective expert use.
> Both classes are assessed at fixed operating points.
> Chapter 5 connects the results to gateway design priorities.

**After 1:**

> The objectives are to measure what structural and character representations add across sources, whether monitoring improves decisions, and how selective expert use affects detection and execution.
> Fixed operating points support comparisons in both classes.
> Chapter 5 develops the resulting design priorities.

## 9. 1.5 Engineering Contributions — paragraph 257

**Before 1:**

> For RQ1, the study connects internal discrimination to URL-format dependence.
> Structural features add 64.07 percentage points of internal recall; validation controls reveal format–class association, and a fixed-weight scheme intervention measures external sensitivity.
> A separate refit achieves exact invariance across 8,622 pairs with a measured detection trade-off.
> These experiments develop the robustness questions of Ahamed et al. (2026), Rashid et al. (2024) and Tsai et al. (2024) without assigning an unmeasured share of the gain to HTTPS (Sections 4.8 and 4.10.1).

**After 1:**

> For RQ1, structural features add 64.07 percentage points of internal recall.
> Validation labels also follow a URL-format convention, and changing the scheme changes frozen-model external decisions.
> A separate refit removes sensitivity across 8,622 pairs and measures the detection trade-off.
> These tests assess representation stability alongside accuracy, extending the robustness questions of Ahamed et al. (2026), Rashid et al. (2024) and Tsai et al. (2024).
> They do not assign a fraction of the gain to HTTPS (Sections 4.8 and 4.10.1).

## 10. 1.5 Engineering Contributions — paragraph 258

**Before 1:**

> For RQ2, the study follows distribution alerts into detection decisions.

**After 1:**

> For RQ2, the study follows monitoring alerts into decisions.

**Before 2:**

> The comparison establishes why response utility must accompany sensitivity.
> Paired routing analysis identifies the expert corrections available to that response, applying the downstream-error reasoning of Jitkrittum et al. (2023) to this gateway (Sections 4.8.1 and 4.10.2).

**After 2:**

> Recognizing unfamiliar inputs and improving their classification require separate tests.
> Paired routing analysis counts available expert corrections, applying the downstream-error reasoning of Jitkrittum et al. (2023) to this gateway (Sections 4.8.1 and 4.10.2).

## 11. 1.5 Engineering Contributions — paragraph 259

**Before 1:**

> For RQ3, paired decisions quantify expert value and separate measurements locate execution cost.
> The models' common false positives establish a 90.5487% floor for routing between unchanged decisions.
> Supporting service experiments demonstrate faster worker-owned clients on synthetic structural-only HTTP requests, corroborated by a no-model control.
> Comparable predictions agree despite changed admission order; expert scoring cost is measured separately.
> Building on established cascades (Li et al., 2021), the study connects selective execution to detection and service behavior.

**After 1:**

> For RQ3, shared false positives set an exact 90.5487% floor on routing between unchanged decisions.
> Separately, worker-owned clients improve synthetic structural-only HTTP latency; a no-model control supports the client-path explanation.
> Comparable predictions agree while admission order changes.
> Expert scoring cost is measured on its own workload.
> Building on established cascades (Li et al., 2021), these experiments evaluate both selective execution and service design.

## 12. Chapter 2—Review of the Literature — paragraph 266

**Before 1:**

> This focused, nonsystematic review develops the questions behind the evaluation.
> Which URL signals are available, and which transfer across sources?
> When do expert calls or monitoring decisions help, and what do they cost?
> Collection conventions, correlated errors and execution overhead connect these questions.

**After 1:**

> This focused, nonsystematic review explains CyberSentinel's design choices.
> URL detection determines what evidence the gateway uses; transfer research asks whether it survives a change of source.
> Cascades and monitoring inform when to spend computation and how to judge the resulting decisions.
> Service measurement examines the caller's experience.

## 13. 2.7 Gap Addressed by the Study — paragraph 295

**Before 1:**

> The study connects URL-detector design to detection and execution through three questions: representation value across sources, monitoring-guided escalation, and selective expert execution.
> Paired outcomes and controlled interventions explain the results and identify design targets.
> Table 2.1 relates this contribution to robustness, adaptation and deferral research.

**After 1:**

> CyberSentinel connects three engineering choices: URL representation, monitoring-guided expert selection and service implementation.
> Paired outcomes and controlled interventions measure how each choice affects the gateway.
> Table 2.1 places this contribution within robustness, adaptation and deferral research.

## 14. 2.7 Gap Addressed by the Study — paragraph 297

**Before 1:**

> Structural-versus-length comparisons establish the internal baseline; validation controls and scheme interventions examine its format dependence.
> Ordered replay follows monitoring alerts into decisions.
> Paired outputs measure expert corrections and introduced errors, while separate scoring and HTTP measurements locate execution cost.
> Each comparison informs a gateway design choice.

**After 1:**

> Structural-versus-length tests measure the feature gain; validation controls and scheme interventions examine its dependencies.
> Ordered replay follows alerts into decisions, while paired outputs count expert corrections and new errors.
> Separate scoring and HTTP tests locate execution costs.
> Each comparison informs what to change and how to judge it.

## 15. 3.1 Research Design — paragraph 307

**Before 1:**

> The design pairs observations at the unit relevant to each research question.
> RQ1 compares decisions on identical uniform resource locators (URLs), with contrast-specific uncertainty in Section 3.8.

**After 1:**

> Each research question is tested at the unit where its effect can be observed.
> RQ1 compares decisions on identical uniform resource locators (URLs), using the contrast-specific uncertainty in Section 3.8.

**Before 2:**

> Models, thresholds and decision rules remain fixed across its two sessions; Section 4.6 and Appendix C describe the resulting limitations.

**After 2:**

> Models, thresholds and decision rules stay fixed across its two sessions; Section 4.6 and Appendix C describe the resulting limitations.

## 16. 3.1.1 Architecture and Information Flow — paragraph 309

**Before 1:**

> Preparation first determines which URLs are eligible and establishes their identities.
> Each eligible URL then supplies 25 structural features and a Logistic-L1 score.
> The uncertainty band determines whether this first-stage decision stands or the character transformer replaces it.
> The Gaussian mixture model (GMM) observes the features and first-stage score, but its alerts affect only future requests.
> Saved outcomes are joined to publisher-label strata afterward for evaluation.
> HTTP execution is measured separately, as shown in Figure 3.1 and Table 3.1.

**After 1:**

> Preparation determines which URLs are eligible and assigns their identities.
> Each URL supplies 25 structural features and a Logistic-L1 score.
> The uncertainty band decides whether to retain that decision or replace it with the character transformer's.
> The Gaussian mixture model (GMM) observes the features and first-stage score; its alerts affect only future requests.
> Publisher-label strata are joined to saved outcomes afterward for evaluation.
> HTTP execution is measured separately (Figure 3.1 and Table 3.1).

## 17. 3.1.1 Architecture and Information Flow — paragraph 310

**Before 1:**

> The router does not receive the external outcome strata.
> Every record retained after mechanical quarantine enters the same ordered stream; gold, certified and secondary designations are applied only in the later analysis.
> The records also distinguish assignment to the transformer from an actual transformer execution.
> This distinction allows logical routing and physical computation to be measured separately.

**After 1:**

> The router processes every mechanically eligible record in the same ordered stream.
> It never receives the external outcome strata: gold, certified and secondary designations enter only the later analysis.
> Saved records distinguish a request assigned to the transformer from an actual transformer execution, allowing routing decisions and computation to be measured separately.

## 18. 3.8 RQ1 Analysis — paragraph 337

**Before 1:**

> Systems are compared on aligned records under frozen models and thresholds.
> Section 4.6 and Appendix C document the recovery and information exposure qualifying those comparisons.

**After 1:**

> Every system is compared on the same aligned records with models and thresholds held fixed.
> Section 4.6 and Appendix C document the recovery procedures and prior information exposure that affect interpretation.

## 19. 3.11 RQ2 Analysis — paragraph 353

**Before 1:**

> H2 requires more than recognition of external windows.
> Section 1.4.2 also specifies the validation-audit false-alert limit, certified-negative policy FPR and positive paired recall improvement.
> The original 2,000-replicate domain bootstrap describes saved outcomes conditional on stream order.
> Section 3.8 corrects the uncertainty claims it cannot support.
> The comparison evaluates the decisions following an alert, without attributing a causal effect solely to the monitor signal.

**After 1:**

> RQ2 evaluates the response as well as the alert.
> Alongside external-window recognition, Section 1.4.2 requires a validation-audit false-alert limit, certified-negative policy FPR and positive paired recall improvement.
> The original 2,000-replicate domain bootstrap describes saved outcomes conditional on stream order; Section 3.8 corrects the uncertainty claims it cannot support.
> Following each alert into subsequent decisions measures the response without assigning its causal effect solely to the monitor signal.

## 20. 3.13 Reproducibility and Integrity Controls — paragraph 360

**Before 1:**

> The evidence supports examination from input to outcome.
> Appendix A links each result to its population, fitted state and evaluation procedure, and independent calculations check saved counts, paired decisions and latency.
> Every declared diagnostic fit and all outcomes are retained.
> Appendix C documents information exposure and execution history so the comparisons can be interpreted and reused accurately.

**After 1:**

> The record lets a reader trace a result from input to outcome.
> Appendix A links each finding to its population, fitted model and evaluation procedure.
> Separate calculations check saved counts, paired decisions and latency.
> Every declared diagnostic fit and its outcomes are retained.
> Appendix C records information exposure and execution history so others can interpret and reuse the comparisons.

## 21. Chapter 4—Results — paragraph 405

**Before 1:**

> Sections 4.1–4.7 answer RQ1–RQ3 through representation, monitoring and service measurements and adjudicate H1–H3.
> Sections 4.8–4.10 explain the outcomes using scheme interventions, expert correction analysis, client controls and shuffled-label diagnostics.
> Each comparison connects an implemented design choice to its measured consequences.

**After 1:**

> The evaluation measures CyberSentinel's component contributions and investigates their mechanisms.
> Sections 4.1–4.7 answer RQ1–RQ3 through representation, monitoring and service tests and adjudicate H1–H3.
> Sections 4.8–4.10 examine scheme sensitivity, expert corrections, client costs and shuffled-label ranking, connecting the findings to design decisions.

## 22. 4.1 Completed Evaluation and Populations — paragraph 408

**Before 1:**

> Saved confusion counts, paired outcomes and request records support independent arithmetic reconstruction.
> Section 4.6 describes the two-session operational matrix; Appendix C preserves its execution identities and verification record.

**After 1:**

> Saved confusion counts, paired outcomes and request records allow the arithmetic to be checked independently.
> Section 4.6 describes the two-session operational matrix; Appendix C retains execution identities and the verification record.

## 23. 4.2 RQ1: Representation Value and Generalization — paragraph 419

**Before 1:**

> Structural features deliver a substantial recall gain: 64.0723 percentage points internally and 76.8116 on gold positives relative to length-only (Table 4.3).

**After 1:**

> Structural features produce a substantial recall gain over length alone: 64.0723 percentage points internally and 76.8116 on gold positives (Table 4.3).

**Before 2:**

> External evaluation identifies the associated specificity cost: certified FPR is 90.5487%, compared with 9.4513% for length-only.
> The paired comparison quantifies the external specificity cost of the recall gain.
> Because source composition and representation both change, this comparison alone cannot separate their effects.

**After 2:**

> On certified external negatives, FPR is 90.5487%, compared with 9.4513% for length-only.
> Measuring both classes shows the specificity cost alongside the additional detections.
> This comparison spans changes in source composition and representation, so it cannot separate their effects.

## 24. 4.2 RQ1: Representation Value and Generalization — paragraph 422

**Before 1:**

> RQ1 establishes the measured representation gain within the benchmark and identifies the properties that govern its transfer.
> Format–class association and the fixed-weight scheme test connect discrimination to supplied URL spelling; the different partitions do not quantify the share of recall gain caused by HTTPS.
> The zero-call cascade reproduces structural-only recall.
> Paired errors and scheme interventions therefore direct design toward transferable discrimination and useful expert corrections.
> The joint H1 decision incorporates external specificity and incremental cascade benefit.

**After 1:**

> RQ1 measures what the richer representation adds and opens a second question: which signals support that gain?
> Validation format–class association and the fixed-weight scheme test show dependence on supplied URL spelling.
> The different partitions do not quantify the HTTPS-attributable share of the recall gain.
> The zero-call cascade reproduces structural-only recall, making expert corrections a separate design question.
> H1 evaluates both external specificity and incremental cascade benefit.

## 25. 4.3 RQ2: Monitoring and Future-Only Routing — paragraph 427

**Before 1:**

> Following monitoring alerts through subsequent decisions measures the value of the resulting action.
> The policy routes 8,253/8,701 external requests to the transformer, or 94.8512%, compared with none for the fixed band.
> Both detect 69/69 gold positives; certified false positives rise from 2,261 to 2,479, adding 218 errors or 8.7305 percentage points.
> These paired outcomes identify expert selection as a distinct design requirement from detecting input change.
> They describe the ordered replay; its stateful, nonrandomized design and degenerate historical bootstrap do not establish a population or randomized causal effect.

**After 1:**

> Following an alert into the next decisions reveals what escalation contributes.
> The policy sends 8,253/8,701 external requests to the transformer, or 94.8512%, compared with none for the fixed band.
> Both detect 69/69 gold positives.
> Certified false positives rise from 2,261 to 2,479: an additional 218 errors, or 8.7305 percentage points.
> Recognizing input change therefore needs a separate test of the expert selected to respond.
> These results describe ordered replay; its stateful, nonrandomized design and degenerate historical bootstrap do not establish a population or randomized causal effect.

## 26. 4.3 RQ2: Monitoring and Future-Only Routing — paragraph 430

**Before 1:**

> The monitor comparison separates sensitivity to input change from the quality of the resulting action.
> External-window sensitivity, audit false alerts, policy FPR and recall improvement give distinct evidence about the monitor and the action it triggers.

**After 1:**

> The comparison makes monitoring and response two measurable parts of the design.
> External-window sensitivity and audit false alerts describe the monitor; policy FPR and recall improvement describe the action it triggers.

**Before 2:**

> Action utility still requires paired decision evidence.

**After 2:**

> Whether an alert leads to a better decision still requires paired outcomes.

## 27. 4.4 RQ3: Real-HTTP Cost and Inline Viability — paragraph 432

**Before 1:**

> The service evaluation measures what the caller experiences across the complete HTTP request path, including costs outside isolated model scoring.

**After 1:**

> The HTTP tests measure the caller's full wait across the request path, exposing costs beyond isolated model scoring.

## 28. 4.4 RQ3: Real-HTTP Cost and Inline Viability — paragraph 434

**Before 1:**

> The operational groups appear in Table 4.6, with throughput and post-phase drain separated in Table 4.7.
> The designated invocation cell records zero physical transformer attempts across 10,000 measured requests, meeting the ≤ 30% limit and agreeing with the zero fixed-band selection count.
> This quantifies the expert computation saved by the selected gate; external detection is evaluated alongside that saving.
> For primary cells 21–25, pooled p95 across 50,000 terminal latencies is 364.3004101 ms, above the 200 ms requirement.
> The error count is 0/50,000, satisfying < 0.1%.
> Together, these observations distinguish invocation, latency and request reliability.

**After 1:**

> The fixed gate meets the expert-invocation and primary request-error limits.
> Its designated invocation cell records zero physical transformer attempts across 10,000 measured requests, within the ≤ 30% limit and consistent with zero fixed-band selections.
> Primary cells 21–25 record 0/50,000 errors, satisfying < 0.1%, while pooled p95 across 50,000 terminal latencies is 364.3004101 ms against the 200 ms requirement.
> These measurements separate computation saved from time spent serving a request.
> Table 4.6 reports the operational groups; Table 4.7 separates throughput from post-phase drain.
> External detection is assessed alongside these service results.

## 29. 4.4 RQ3: Real-HTTP Cost and Inline Viability — paragraph 439

**Before 1:**

> Joint detection and service measurements show what the low invocation rate delivers.
> Both the cascade and transformer exceed the certified FPR limit, flag all 1,163 label-free Tranco controls and detect the same 69 gold positives.

**After 1:**

> The joint measurements show why invocation, detection and latency need their own tests.
> The cascade and transformer exceed the certified FPR limit, flag all 1,163 label-free Tranco controls and detect the same 69 gold positives.

**Before 2:**

> It is too wide to establish the −2-point margin, but does not demonstrate inferiority.
> RQ3 establishes low transformer use and primary request reliability while locating the specificity, recall-uncertainty and latency constraints.
> H3 satisfies two of its eight gates; the remaining six specify the conditions for qualifying the complete gateway.

**After 2:**

> It leaves the −2-point margin unresolved and does not demonstrate inferiority.
> RQ3 measures low transformer use and primary request reliability, while specifying the constraints on specificity, recall uncertainty and latency.
> H3 satisfies two of its eight gates; the remaining six define what the complete gateway must additionally demonstrate.

## 30. 4.8.1 From Aggregate Behavior to Testable Mechanisms — paragraph 469

**Before 1:**

> The diagnostic comparisons make the results actionable.
> Changing scheme spelling under fixed weights tests representation sensitivity.
> Error overlap determines the best outcome available from selecting between fixed model decisions.
> A no-model control locates request latency that persists without classifier work.
> Each comparison connects an observed behavior to a distinct intervention and a measurable consequence.

**After 1:**

> Each diagnostic tests an explanation for observed behavior.
> Changing scheme spelling with weights fixed tests input sensitivity.
> Paired model errors establish available expert corrections.
> Removing model work from the HTTP control probes delay elsewhere in the request path.
> These comparisons identify changes to implement and evaluate.

## 31. 4.8.1 From Aggregate Behavior to Testable Mechanisms — paragraph 470

**Before 1:**

(Added)

**After 1:**

> The supplied URL format provides a concrete clue.

**Before 2:**

> These subset summaries do not describe every training row or live transport; external inputs use publisher url_norm.
> Section 4.10.4 measures validation format–class association.
> This connects internal separability to the frozen-model test in Section 4.10.1.
> Because a prefix affects is_https and string-composition predictors, neutralization must cover several pathways.
> Collection-source causality remains unidentified.
> The no-model control separately motivates testing client organization with the scorer fixed.

**After 2:**

> These summaries concern the stated subsets and publisher url_norm, not every training row or live transport.
> Validation format–class association in Section 4.10.4 complements the fixed-weight test in Section 4.10.1.
> A prefix changes several predictors, including is_https and string composition, so removing scheme dependence requires more than dropping one indicator.
> The evidence does not identify how collection practices produced the association.
> Separately, the no-model control motivates changing client organization while keeping the scorer fixed.

## 32. 4.8.1 From Aggregate Behavior to Testable Mechanisms — paragraph 471

**Before 1:**

> Paired error analysis yields an exact routing bound for the evaluated models.
> Both falsely flag the same 2,261 of 2,497 certified negatives.

**After 1:**

> The paired decisions give an exact limit on what routing can do.
> Both models falsely flag the same 2,261 of 2,497 certified negatives.

**Before 2:**

> Even a router selecting the correct decision whenever either model supplies one must retain every shared false positive: its FPR is at least 2,261/2,497, or 90.5487%.
> This cohort-specific bound directs redesign toward changed thresholds, new evidence or retraining before gate selection alone can improve specificity.
> Each would create different available decisions and require evaluation.

**After 2:**

> Even a router that chooses correctly whenever either model does must retain every shared false positive.
> Its FPR floor is 2,261/2,497, or 90.5487%.
> For these fixed decisions, improving the gate cannot lower that floor.
> Changed thresholds, additional evidence or retraining would create different decisions and require a new evaluation.

## 33. 4.8.1 From Aggregate Behavior to Testable Mechanisms — paragraph 473

**Before 1:**

> The fixed cascade's inactivity follows directly from calibration: the selected width is zero and evaluation contains no ties (Section 4.2).

**After 1:**

> The fixed cascade's inactivity follows from calibration: its width is zero, and evaluation contains no ties (Section 4.2).

**Before 2:**

> This establishes a basis for examining wider gates while retaining the original rule and its recorded decisions.

**After 2:**

> With that behavior verified, wider gates can be examined against the original rule and its recorded decisions.

## 34. 4.8.2 Population Flow and Operating-Point Comparison — paragraph 479

**Before 1:**

> The labeled comparison quantifies the detection trade-off of scheme neutralization.
> At the two frozen operating points in Table 4.16, the refitted candidate produces 251 fewer false positives and 184 fewer true positives.
> These paired counts distinguish beneficial false-alert reductions from suppressed detections, giving the invariance result its operational meaning.

**After 1:**

> The refit removes an input dependency and quantifies its detection trade-off.
> At the two frozen operating points in Table 4.16, it produces 251 fewer false positives and 184 fewer true positives.
> Paired labels make both consequences of the invariance repair explicit.

## 35. 4.8.3 Representation Invariance and Decision — paragraph 487

**Before 1:**

> The representation meets its invariance requirement across the tested population.
> All 8,622 candidate feature vectors and scores exactly match their opposite-scheme companions, with no decision changes (Table 4.17).

**After 1:**

> The refitted representation removes scheme sensitivity exactly across the tested population.
> All 8,622 candidate feature vectors and scores match their opposite-scheme companions, with no decision changes (Table 4.17).

**Before 2:**

> A flip establishes sensitivity, not necessarily an error.
> The companions do not add independent outcome labels.

**After 2:**

> A flip shows sensitivity; whether it is an error requires a label.
> The companions add no independent outcome labels.

## 36. 4.8.3 Representation Invariance and Decision — paragraph 489

**Before 1:**

> D establishes exact scheme invariance and measures the accompanying detection trade-off.
> Normalization, refitting and recalibration reduce both false positives and true-positive detections, with FPR remaining above the required ceiling.
> The engineering finding is that representation stability and class separation are distinct properties, each requiring labeled evaluation when the detector is redesigned.
> The joint requirement evaluates both properties together.

**After 1:**

> D verifies scheme invariance.
> The combined normalization, refitting and recalibration also reduce both false positives and true-positive detections, with FPR above the ceiling.
> Stability and class separation are distinct properties; the joint requirement tests both.

## 37. 4.9.2 Paired Tail-Latency Effect — paragraph 496

**Before 1:**

> Changing client ownership reduces p95 in every primary pair (Table 4.18).

**After 1:**

> Giving each worker its own client reduces p95 in every primary pair (Table 4.18).

**Before 2:**

> Its two-sided 97.5% interval, from 10,000 fixed run-pair resamples, is [0.187903, 0.213508].
> This estimates the effect in the tested synthetic structural-only HTTP setup; every pair and resample has a defined result.

**After 2:**

> Across 10,000 fixed run-pair resamples, the two-sided 97.5% interval is [0.187903, 0.213508].
> Every pair and resample has a defined result.
> The estimate describes this synthetic structural-only HTTP setup.

## 38. 4.9.2 Paired Tail-Latency Effect — paragraph 497

**Before 1:**

> The control locates the improvement outside changes to the classifier.
> Pooled successful-request p95 falls from 358.3663 to 71.9990 ms for structural scoring and from 345.95 to 67.21 ms with no model.
> The similar reduction supports a client/transport explanation, although it does not identify a particular queue or scheduling cost.

**After 1:**

> The no-model control helps explain where the time was saved.
> Pooled successful-request p95 drops from 358.3663 to 71.9990 ms with structural scoring and from 345.95 to 67.21 ms with no model.
> Both improvements point to the client/transport path, without isolating a particular queue or scheduling cost.

**Before 2:**

> These pooled request quantiles and the median paired run ratio are different summaries.
> Section 4.9.3 reports all control and response comparisons.

**After 2:**

> Pooled request quantiles and the median paired run ratio summarize different quantities; Section 4.9.3 reports the full control and response comparisons.

## 39. 4.9.4 Requirement S and Engineering Interpretation — paragraph 508

**Before 1:**

> S achieves four of its five requirements (Table 4.20): the latency, reliability and comparable-prediction results establish a useful client-path improvement.
> The fifth requirement concerns admission-sequence equality, which is not preserved.
> The thresholds and required response fields retain their prespecified definitions.

**After 1:**

> S meets four of its five requirements (Table 4.20), demonstrating lower latency and reliable worker requests.
> Comparable predictions also agree.
> Admission-sequence equality, the fifth requirement, is not preserved.
> Each threshold and required response field keeps its prespecified definition.

## 40. 4.9.4 Requirement S and Engineering Interpretation — paragraph 510

**Before 1:**

> S demonstrates a substantial client-path optimization: every comparable prediction is preserved, and worker latency and reliability meet their requirements.
> The response comparison also makes the design choice explicit.
> Applications requiring identical admission order need an additional ordering mechanism and evaluation to satisfy the full response contract.
> The measured benefit applies to the single-host, synthetic, structural-only closed-loop workload.
> It is separate from H3 and from transformer-service acceleration.

**After 1:**

> Worker-owned clients improve latency and reliability in this experiment, preserving every comparable prediction.
> Applications requiring identical admission order need a separately evaluated ordering mechanism.
> The single-host, synthetic, structural-only closed-loop result is separate from H3 and does not measure transformer-service acceleration.

## 41. 4.10 Mechanism Tests: Representation, Routing and Cost — paragraph 512

**Before 1:**

> The mechanism tests turn representation, routing and cost measurements into design evidence.

**After 1:**

> These tests examine decision changes and computation.

**Before 2:**

> Domain-balanced routing identifies selected repairs and harms, and invocation counters quantify the work performed by the expert.
> Audited shuffled-label controls explain how substantial ranking can arise after training labels are randomized.

**After 2:**

> The domain-balanced cohort measures revised gates' repairs and new errors, while invocation counters record actual work.
> Audited shuffled-label fits explain strong validation ranking after training-label randomization.

## 42. 4.10.1 Frozen-model Scheme Intervention — paragraph 514

**Before 1:**

> A prefix change has a substantial effect even with the structural model held fixed (Table 4.21).
> HTTP detects all 69 gold positives but flags every certified negative.
> With Hypertext Transfer Protocol Secure (HTTPS), certified false positives fall from 2,261 to 50, while gold detections fall from 69 to 22.
> The change removes 2,211 false positives at the cost of 47 true-positive detections.
> FPR remains 2.0024%, above 1%, and recall declines to 31.8841%.
> The intervention quantifies the detection cost of prefix replacement and establishes the need to evaluate normalization against both classes.

**After 1:**

> Changing the prefix changes decisions even when the structural model is held fixed (Table 4.21).
> HTTP detects all 69 gold positives and flags every certified negative.
> With Hypertext Transfer Protocol Secure (HTTPS), certified false positives drop from 2,261 to 50 and gold detections from 69 to 22.
> The change removes 2,211 false positives and 47 true-positive detections.
> FPR is 2.0024%, above 1%, and recall is 31.8841%.
> The paired intervention demonstrates scheme dependence and measures its cost in both classes.
> It also shows why normalization needs a labeled detection test.

## 43. 4.10.1 Frozen-model Scheme Intervention — paragraph 516

**Before 1:**

> HTTPS also reduces secondary-positive detections from 4,972 to 687 and label-free Tranco alerts from 1,163 to 63.
> F establishes scheme dependence under fixed weights and threshold; validation format–class association supplies related evidence about internal separability.
> D instead refits and recalibrates on the original development partitions, then evaluates another cohort.
> Its invariance and detection trade-off cannot be pooled with F or attributed to normalization alone.

**After 1:**

> HTTPS reduces secondary-positive detections from 4,972 to 687 and label-free Tranco alerts from 1,163 to 63.
> F demonstrates scheme dependence with weights and threshold fixed; the validation association gives related evidence about internal separability.
> D asks a different question by refitting and recalibrating on the original development partitions before evaluating another cohort.
> Its invariance and detection effects cannot be pooled with F or attributed to normalization alone.

## 44. 4.10.2 Expanded Domain-disjoint Routing Evaluation — paragraph 520

**Before 1:**

> The recall comparison is precise but operates at a ceiling.
> Original and symmetric cascades each share 1,392 detections with transformer-only and have no positive discordances.
> Their conservative two-sided 95% envelope is [−0.3143, 0.3143] percentage points, above the −2-point lower-margin requirement under independent labeled-pair assumptions.

**After 1:**

> The recall comparison supports preservation of positive detections at these operating points.
> Original and symmetric cascades share 1,392 detections with transformer-only, with no positive discordances.
> Under independent labeled-pair assumptions, their conservative two-sided 95% envelope is [−0.3143, 0.3143] percentage points, above the −2-point lower-margin requirement.

**Before 2:**

> This supports recall-only noninferiority at these operating points; it does not demonstrate expert utility when specificity remains poor.

**After 2:**

> This is recall-only noninferiority at a detection ceiling; the false-positive counts and selected corrections determine expert utility.

**Before 3:**

> Resampling all-zero differences gives descriptive [0,0] and cannot establish campaign independence or replace the nondegenerate interval.

**After 3:**

> Resampling all-zero differences gives descriptive [0,0], which cannot establish campaign independence or replace the nondegenerate interval.

## 45. 4.10.2 Expanded Domain-disjoint Routing Evaluation — paragraph 521

**Before 1:**

> The selected calls reveal the distinction between limiting expert use and selecting useful corrections.

**After 1:**

> Routing quality depends on the corrections selected as well as how often the expert is called.

**Before 2:**

> Across all available transformer decisions, one would repair a structural false positive and 194 would introduce one.

**After 2:**

> Among all transformer decisions, one would repair a structural false positive and 194 would introduce one.

**Before 3:**

> Useful selection is assessed through paired decisions; the cohort mixture is not a deployment-prevalence estimate.

**After 3:**

> The paired decisions judge selection, while the cohort mixture supplies no estimate of deployment prevalence.

## 46. 4.10.4 Shuffled-label Controls and Ranking Mechanism — paragraph 535

**Before 1:**

> The controls explain how small random coefficients rank structured validation labels despite near-constant probabilities.
> Component removal, randomization and convergence checks support the mechanism; correlated features also contribute.
> Format–class association and the external scheme intervention converge on a representation dependency.
> The 32,695-row validation and 34,593-row group-test analyses do not identify the HTTPS-attributable fraction of the 64.07-point gain.
> Collection-source causality, historical-fit authentication and certification against every leakage path remain outside these checks; threshold-selected performance requires separate evaluation.

**After 1:**

> The controls explain why small random coefficients can rank validation labels well while producing probabilities close to a constant.
> Removing a score term, repeating randomization and tightening convergence give complementary tests of that explanation; correlated features also contribute.
> The internal format association and external scheme intervention both expose representation dependence.
> The 32,695-row validation and 34,593-row group-test analyses do not quantify the HTTPS-attributable fraction of the 64.07-point gain.
> These checks also do not identify collection-source causality, authenticate historical fits or certify the absence of every leakage path.
> Threshold-selected performance requires separate evaluation.

## 47. 5.1 Direct Answers to the Research Questions — paragraph 538

**Before 1:**

> The study answers Research Questions 1–3 (RQ1–RQ3) with an empirical account of representation value, expert selection and execution cost.
> Paired interventions identify which system property changes, what benefit it delivers and what trade-off accompanies it.
> Table 4.14 reports the hypothesis decisions; the findings below explain their implications for design.

**After 1:**

> CyberSentinel answers Research Questions 1–3 (RQ1–RQ3) through an implemented gateway and controlled experiments.
> The structural model supplies a substantial internal recall gain; subsequent comparisons investigate its dependencies, available expert corrections and service improvements.
> Table 4.14 gives the hypothesis decisions.
> The answers below develop their implications for design.

## 48. 5.1 Direct Answers to the Research Questions — paragraph 539

**Before 1:**

> Format–class association qualifies transfer without quantifying the HTTPS-attributable share.
> External certified FPR is 90.5487%, versus 9.4513% for length-only; the cascade adds no recall or expert calls.
> These paired measurements distinguish internal representation value from transferable specificity and incremental expert benefit (Tables 4.2–4.3).

**After 1:**

> Validation labels also follow a format convention whose share of the gain is unquantified.
> External certified FPR is 90.5487%, versus 9.4513% for length-only; the cascade adds no recall or expert calls (Tables 4.2–4.3).
> This finding motivates checking the signals behind a feature gain and measuring specificity on the intended source.

## 49. 5.1 Direct Answers to the Research Questions — paragraph 540

**Before 1:**

> Paired outcomes quantify the scope for expert correction: common false positives persist under selection between unchanged decisions.
> Representation tests establish a second design target.
> Supplied scheme changes frozen-model decisions; a refit achieves exact invariance while reducing both true and false positives.
> These distinct interventions require their own labeled evaluation; the refit's detection effect cannot be assigned to normalization alone.

**After 1:**

> The interventions make that test concrete.
> Scheme spelling changes frozen-model decisions.
> A separate refit removes scheme sensitivity while reducing true and false positives; its combined fitting and calibration changes require their own labeled evaluation.
> Paired outputs establish which errors an expert can correct.
> Shared errors survive every selection between those unchanged decisions.

## 50. 5.1 Direct Answers to the Research Questions — paragraph 541

**Before 1:**

> The GMM flags 115/132 external windows (87.12%) against an 80% target and 28/252 audit windows (11.11%) against 5%.
> Ordered, stateful replay selects the expert for 8,253/8,701 requests (94.8512%); the resulting policy adds 218 certified false positives and retains 69/69 gold detections.
> Following alerts into paired decisions establishes a concrete evaluation method for response utility alongside input-departure recognition (Table 4.5).

**After 1:**

> The GMM recognizes 115/132 external windows (87.12%), exceeding its 80% target.
> Its audit alerts are 28/252 windows (11.11%) against 5%, so sensitivity and false alerts must be assessed together.
> Ordered, stateful replay sends 8,253/8,701 requests (94.8512%) to the expert.
> The policy retains 69/69 gold detections and adds 218 certified false positives (Table 4.5).
> Recognizing a change in inputs does not tell a gateway which response will help.
> Following alerts into paired decisions supplies that test.

## 51. 5.1 Direct Answers to the Research Questions — paragraph 542

**Before 1:**

> The joint evaluation separates the costs of expert execution and request handling from detection quality.
> The fixed cascade meets its invocation and primary request-error limits; pooled p95 is 364.30 ms against a 200 ms budget, and external false-alert safeguards are exceeded.
> For the original 69-pair recall sample, the corrected lower bound is −6.1533 percentage points under independent labeled-pair assumptions.
> Its position below the −2-point margin leaves noninferiority unresolved, rather than establishing inferiority.
> These measurements identify the constraints governing inline operation (Tables 4.6–4.7 and 4.14) and motivate the controlled comparisons of expert work and request handling.

**After 1:**

> The fixed cascade meets the invocation and primary request-error limits.
> Pooled p95 is 364.30 ms against a 200 ms budget, and external false-alert safeguards are exceeded.
> The original 69-pair recall sample gives a corrected lower bound of −6.1533 percentage points under independent labeled-pair assumptions.
> That bound leaves the −2-point noninferiority margin unresolved; it does not establish inferiority (Tables 4.6–4.7 and 4.14).
> Measuring these quantities separately reveals where to intervene: detector computation and request handling need distinct experiments, alongside detection quality.

## 52. 5.1 Direct Answers to the Research Questions — paragraph 543

**Before 1:**

> The results identify three design priorities: evaluate representation stability with labeled detection, choose escalation for useful corrections, and distinguish expert computation from request handling.
> Client ownership improves synthetic structural-only HTTP latency; transformer calls increase scoring cost on another workload.
> Each result supports a specific intervention.

**After 1:**

> Worker-owned clients improve the synthetic structural-only service; separate scoring tests quantify transformer cost.
> These experiments connect a proposed change to a measurable consequence.
> Representation stability is tested with labeled detection, expert selection with corrections, and monitoring with the decisions its alerts change.

## 53. 5.2 Engineering Findings and Contribution — paragraph 545

**Before 1:**

> The representation evidence identifies a concrete dependency.
> All 20,209 validation negatives use HTTPS, versus 46.6362% of positives; the negative HTTPS indicator has AUC 0.7668.

**After 1:**

> The benchmark investigation gives URL researchers a concrete check.
> All 20,209 validation negatives use HTTPS, versus 46.6362% of positives; the negative HTTPS indicator reaches AUC 0.7668.

**Before 2:**

> This qualifies the internal gain without quantifying its HTTPS-attributable fraction.
> Paired error analysis establishes the decisions available to expert selection (Sections 4.8.1 and 4.10.2).

**After 2:**

> These findings justify checking format conventions before interpreting a gain as transferable detection, without quantifying the HTTPS-attributable fraction.
> Paired errors establish available expert corrections (Sections 4.8.1 and 4.10.2).

## 54. 5.2 Engineering Findings and Contribution — paragraph 546

**Before 1:**

> Table 4.21 isolates scheme sensitivity under fixed weights.
> Tables 4.16–4.17 verify exact invariance after refitting and quantify the changes in true and false positives.
> Different cohorts and fitting procedures prevent attribution of the refit's effect to normalization alone.
> Together with internal format association, these results justify testing scheme dependence in URL benchmarks without identifying collection-source causality.

**After 1:**

> Table 4.21 tests spelling changes with weights fixed.
> Tables 4.16–4.17 verify exact scheme invariance after refitting and quantify detection changes.
> Different cohorts and fitting procedures prevent attributing the refit's effect to normalization alone.
> Together, the internal association and external intervention support a practical benchmark check; collection-source causality remains unresolved.

## 55. 5.2 Engineering Findings and Contribution — paragraph 547

**Before 1:**

> Monitoring escalation adds 218 false positives without another gold detection.
> Paired outputs explain the constraint: 2,261 common false positives among 2,497 negatives impose a 90.5487% floor on unchanged-decision selection.
> On the separate cohort, one repair is available, but wider gates select none and add 18 false positives.
> Supporting service measurements identify a client-path improvement (Tables 4.18–4.19); Table 4.23 measures expert cost.
> Comparable predictions agree despite changed admission order.

**After 1:**

> Routing provides an exact design constraint: 2,261 shared false positives among 2,497 certified negatives impose a 90.5487% floor on unchanged-decision selection.
> Escalation adds 218 false positives without another gold detection.
> On the separate cohort, one expert repair is available, but wider gates select none and add 18 false positives.
> Count available corrections before tuning a gate.
> Service tests identify a client-path improvement with matching comparable predictions despite changed admission order (Tables 4.18–4.19).
> Table 4.23 measures expert cost separately.

## 56. 5.2 Engineering Findings and Contribution — paragraph 548

**Before 1:**

> These contributions develop established research questions through concrete gateway experiments.
> Jitkrittum et al. (2023) motivate examining downstream errors; this study measures available corrections, selected repairs and introduced errors.
> Ahamed et al. (2026) examine external specificity and class-specific transformations; the present experiments connect internal format association, a fixed-weight scheme effect and a separately verified invariance property.
> Monitoring replay then measures the outcome of changing routing.
> ExpertFusion's documented calibrated fusion and target-prior adaptation provide architectural context (aI-area, 2026; Hussain et al., 2027), within the documented comparison rather than a claim of full-paper coverage or superiority.

**After 1:**

> These experiments make established research questions concrete for a URL gateway.
> Jitkrittum et al. (2023) motivate examining downstream errors; CyberSentinel counts available corrections, selected repairs and introduced errors.
> Ahamed et al. (2026) examine external specificity and class-specific transformations; this study connects an internal format association to a fixed-weight scheme effect and a verified invariance property.
> Monitoring replay follows a change in routing into its outcomes.
> ExpertFusion's documented calibrated fusion and target-prior adaptation provide architectural context (aI-area, 2026; Hussain et al., 2027).
> The comparison covers that documented architecture, without claiming full-paper coverage or superiority.

## 57. 5.2 Engineering Findings and Contribution — paragraph 549

**Before 1:**

> The paired analysis establishes recall-only noninferiority conditional on independent labeled pairs.
> Pairing that interval with specificity and selected corrections makes its engineering meaning explicit: every model detects every positive while flagging most negatives, the original gate calls no expert, and the wider gate adds false positives.
> The interval precisely addresses recall preservation, while the paired decision counts establish the observed incremental utility.
> Shared observations provide neither independent replication nor a guarantee of campaign independence.

**After 1:**

> The recall-only analysis supports noninferiority conditional on independent labeled pairs.
> Every model detects every positive while flagging most negatives; the original gate calls no expert, and the wider gate adds false positives.
> Specificity and selected corrections therefore determine the practical value of the preserved recall.
> Reused observations provide neither independent replication nor guaranteed campaign independence.

## 58. 5.2.1 Findings and Their Engineering Use — paragraph 551

**Before 1:**

> Table 5.1 connects the findings to redesign decisions and the evidence supporting each one.

**After 1:**

> Table 5.1 connects each measured finding to a decision a gateway designer can make.

## 59. 5.2.2 Validity and Reuse of the Evidence — paragraph 554

**Before 1:**

> The reusable evidence connects inputs, fitted states, outcomes and diagnostics in Appendix A.
> Consumed-input checks, reconstructed scores, permutation results and convergence sensitivity provide complementary support for the ranking explanation.
> Appendix C documents information exposure and execution history.
> The resulting record supports reproducible analysis of these fits; historical-fit authentication and independent replication remain separate questions.

**After 1:**

> Appendix A connects inputs, fitted states, outcomes and diagnostics.
> Input audits, reconstructed scores, permutations and convergence checks support the ranking explanation.
> Appendix C records information exposure and execution history.
> This record makes the analyzed fits reproducible; historical-fit authentication and independent replication remain separate requirements.

## 60. 5.2.2 Validity and Reuse of the Evidence — paragraph 555

**Before 1:**

> The shuffled-label analysis contributes a tested explanation of a misleading control outcome.
> Small residual coefficients can produce substantial ranking on validation data where format is associated with class, even when probabilities are nearly constant.
> Input auditing verifies the fitted data, term removal measures one component's effect, repeated randomization reproduces the spread, and stricter optimization preserves it.
> These complementary checks establish a practical method for examining controls through fitted state and score ordering as well as the score mean.

**After 1:**

> The shuffled-label investigation explains a puzzling control result.
> Small residual coefficients can rank format-associated labels well despite nearly constant probabilities.
> Input audits check the fitted data, term removal measures one contribution, randomization reproduces the spread, and stricter optimization preserves it.
> Inspecting coefficients and score ordering alongside mean probability provides a practical test of such controls.

## 61. 5.2.3 Relationship to the Original Hypotheses — paragraph 557

**Before 1:**

> Table 5.2 connects achieved component findings to the complete gateway decision.
> Internal feature gains, the exact routing bound, verified scheme invariance and the measured client-path improvement each support a specific engineering choice.
> The joint H1–H3 decisions in the table determine system-level qualification; the component findings establish the design evidence developed by this research.
> The contribution is an empirical design basis: the experiments identify which properties to change, quantify their consequences, and specify how to evaluate the resulting gateway.

**After 1:**

> Table 5.2 connects component findings, complete gateway decisions and engineering insight.
> The internal feature gain, routing bound, scheme invariance and client-path improvement each answer a design question.
> Joint H1–H3 decisions determine system-level qualification.
> The contribution is a measured basis for redesign: what to change, why it matters and how to test its consequences.

## 62. 5.3 Implications for Practice — paragraph 560

**Before 1:**

> The evidence supports three priorities for this gateway: measure expert corrections before widening a gate, assess specificity alongside representation invariance, and evaluate the decision changes produced by monitoring alerts.
> Count available repairs, selected repairs and introduced errors on labeled data from the intended source.
> Tranco supplies a useful popularity-based control with a different evidential role from certified labels (Le Pochat et al., 2019).

**After 1:**

> Three priorities follow for this gateway.
> Count available expert repairs, selected repairs and new errors on labeled data from the intended source before widening a gate.
> Test specificity alongside invariance.
> Judge monitoring by the decisions its alerts change.
> Tranco supplies a popularity-based control with a different evidential role from certified labels (Le Pochat et al., 2019).

## 63. 5.3 Implications for Practice — paragraph 561

**Before 1:**

> For service design, optimize the measured request path against an explicit response contract.
> The 99,999 comparable predictions agree while admission order changes and one pair times out.
> Expert counters and p99 expose costs that invocation fractions alone can obscure.
> These measurements identify what to preserve and what to measure again when hardware, arrival processes or workloads change.

**After 1:**

> Service design requires a measured request path and an explicit response contract.
> The 99,999 comparable predictions agree; admission order changes and one pair times out.
> Expert counters and p99 expose costs hidden by invocation fractions.
> These measurements specify what to preserve and retest when hardware, workloads or arrivals change.

## 64. 5.4 Limitations and Threats to Interpretation — paragraph 563

**Before 1:**

> The labels remain publisher assignments without new forensic adjudication.
> The 2,497 original certified negatives also occupy only 234 domains, so the row count overstates the number of distinct groups.
> The expanded cohort takes one URL per domain and consequently changes the estimand.

**After 1:**

> The results use publisher labels without new forensic adjudication.
> The 2,497 original certified negatives occupy 234 domains, so the number of URLs exceeds the number of distinct groups.
> Taking one URL per domain in the expanded cohort changes the estimand.

## 65. 5.4 Limitations and Threats to Interpretation — paragraph 564

**Before 1:**

> Domain separation does not remove shared collection conventions.
> PhiUSIIL lacks independent per-row provenance for a noncircular source-only classifier.
> The control analyses establish a reproducible contribution from the HTTPS term and broad ranking variation under randomized training labels on fixed partitions.
> They do not identify how that convention arose or each feature's contribution.
> Term removal retains correlated predictors, and the 99 permutations describe algorithmic sensitivity rather than independent populations.
> True validation labels select the five controls' thresholds; historical fits still lack consumed-input authentication.

**After 1:**

> Domain separation leaves shared collection conventions possible.
> PhiUSIIL lacks the independent per-row provenance needed to fit a noncircular source-only classifier.
> The controls reproduce a contribution from the HTTPS term and broad ranking variation under randomized training labels on fixed partitions.
> They do not determine how the convention arose or isolate every feature's contribution.
> Correlated predictors remain after term removal.
> The 99 permutations describe algorithmic sensitivity, not independent populations.
> True validation labels select the five controls' thresholds, and historical fits still lack consumed-input authentication.

## 66. 5.4 Limitations and Threats to Interpretation — paragraph 565

**Before 1:**

> The url_norm amendment followed preparation and label-count inspection, and checkpoint recovery followed predictions and partial measurement.
> Earlier exposure and repeated scoring depart from a blind, single-pass evaluation.
> Wider gates were selected on exposed internal group-test outcomes, so those gains remain exploratory; the gates were fixed before the separate domain-balanced comparison.
> Models and thresholds stay fixed within the stated comparisons (Appendix C).
> Overlapping GMM windows and activations across domains also make saved-outcome resampling conditional on stream order.
> Label-free probes establish neither semantic preservation nor real-world adversarial robustness.

**After 1:**

> The analyst's information changed during the study.
> The url_norm amendment followed preparation and label-count inspection; checkpoint recovery followed predictions and partial measurement.
> Earlier exposure and repeated scoring differ from a blind, single-pass evaluation.
> Wider gates were selected using exposed internal group-test outcomes, making those gains exploratory, and fixed before the separate domain-balanced comparison.
> Models and thresholds remain fixed within the stated comparisons (Appendix C).
> Overlapping GMM windows and activations across domains make saved-outcome resampling conditional on stream order.
> Label-free probes demonstrate neither semantic preservation nor real-world adversarial robustness.

## 67. 5.4 Limitations and Threats to Interpretation — paragraph 566

**Before 1:**

> Operational measurements come from one host, fixed workload order and two original sessions.
> Session, thermal, cache and elapsed-time effects cannot be separated, and closed-loop timing excludes unsent backlog.

**After 1:**

> The operational measurements use one host, a fixed workload order and two original sessions.
> Session, thermal, cache and elapsed-time effects cannot be separated; closed-loop timing also excludes unsent backlog.

**Before 2:**

> Domain exclusions do not establish temporal independence.
> Refitting and recalibration are part of its combined intervention, while the companions test representation behavior rather than live-site equivalence.

**After 2:**

> Excluding domains does not establish temporal independence.
> D combines refitting and recalibration with normalization, while its companions test representation behavior rather than live-site equivalence.

## 68. 5.4 Limitations and Threats to Interpretation — paragraph 567

**Before 1:**

> The cost experiments address different workloads.

**After 1:**

> The cost experiments measure different workloads.

**Before 2:**

> C uses real strings at concurrency one, a designed 1% mixture and five cyclic rotations.
> Its positions are not fully balanced, and it has no run-level confidence interval.
> The two experiments cannot be combined to establish a transformer-service speedup.
> Sampled host checks document observed conditions, and saved-evidence calculations verify arithmetic.
> Neither establishes disturbance-free operation or independent replication.

**After 2:**

> C uses real strings at concurrency one, a designed 1% mixture and five cyclic rotations; its positions are not fully balanced, and it has no run-level confidence interval.
> Combining them would not establish a transformer-service speedup.
> Sampled host checks record observed conditions, and saved-evidence calculations verify arithmetic.
> Neither proves disturbance-free operation or independent replication.

## 69. 5.5 Future Research, Separate from This Study — paragraph 569

**Before 1:**

> The next experiments follow directly from the design evidence.
> Independently adjudicated, time-indexed sources with formatting conventions recorded separately from outcome would test transferable discrimination and expert corrections.
> Service studies should balance workload and session order, use open-loop arrivals, and evaluate admission order separately from prediction equality.
> Each extends a measured component of the present work.

**After 1:**

> Independently adjudicated, time-indexed sources recording formatting separately from outcome would test transferable discrimination and expert corrections.
> Balanced workload and session order, open-loop arrivals and separate admission-order tests would extend the service evidence.

## 70. 5.6 Conclusion — paragraph 571

**Before 1:**

> CyberSentinel connects URL representation, monitoring and selective execution to measured outcomes.
> Structural features add 64.07 percentage points of internal recall within the benchmark.
> Validation controls identify format–class association, and a fixed-weight intervention establishes external scheme sensitivity.
> A separate refit achieves exact invariance across 8,622 pairs with a quantified reduction in both true and false positives.
> Together, these experiments provide concrete tests of representation stability and its consequences for labeled detection.

**After 1:**

> CyberSentinel provides a working gateway and controlled design evidence.
> Structural features add 64.07 percentage points of internal recall.
> Validation controls reveal format–class association; a fixed-weight intervention establishes external scheme sensitivity.
> A separate refit removes that sensitivity across 8,622 pairs while reducing true and false positives, distinguishing stability from detection quality.

## 71. 5.6 Conclusion — paragraph 572

**Before 1:**

> Monitoring and routing experiments make expert value measurable.
> External-window sensitivity reaches 87.12%, while escalation adds 218 false positives without another gold detection.
> The models' 2,261 common false positives among 2,497 negatives establish a 90.5487% floor for selection between unchanged decisions.

**After 1:**

> External-window sensitivity reaches 87.12%; escalation adds 218 false positives without another gold detection.
> The models' 2,261 shared false positives among 2,497 certified negatives impose an exact 90.5487% floor on unchanged-decision selection.

**Before 2:**

> Available corrections and selected outcomes therefore guide expert and policy design.

**After 2:**

> Count available corrections before choosing an expert; evaluate the policy by its selections.

## 72. 5.6 Conclusion — paragraph 573

**Before 1:**

> H1–H3 retain their original joint criteria.
> The resulting design guidance is concrete: test format dependence, select experts for useful corrections, follow alerts into decisions, and optimize the execution path identified by measurement.

**After 1:**

> The complete gateway did not meet the prespecified joint criteria of H1–H3; the experiments identify what a redesign must address.
> CyberSentinel provides the next designer with a working architecture, a measured service improvement, a benchmark check and an exact routing bound.

## 73. Table 4.14 — row 4, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 74. Table 4.14 — row 5, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 75. Table 4.14 — row 6, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 76. Table 4.14 — row 8, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 77. Table 4.14 — row 10, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 78. Table 4.14 — row 12, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 79. Table 4.14 — row 13, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 80. Table 4.14 — row 14, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 81. Table 4.14 — row 15, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 82. Table 4.14 — row 16, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 83. Table 4.14 — row 17, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 84. Table 4.14 — row 18, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 85. Table 4.14 — row 19, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 86. Table 4.14 — row 21, column 4

**Before 1:**

> fail

**After 1:**

> not met

## 87. Table 5.2 — row 0, column 3

**Before 1:**

(Added)

**After 1:**

> What it revealed

## 88. Table 5.2 — row 1, column 3

**Before 1:**

(Added)

**After 1:**

> Check format dependence, specificity and available expert corrections.

## 89. Table 5.2 — row 2, column 3

**Before 1:**

(Added)

**After 1:**

> Test escalation outcomes separately from alert sensitivity and false alerts.

## 90. Table 5.2 — row 3, column 3

**Before 1:**

(Added)

**After 1:**

> Evaluate detection, recall uncertainty, expert work and HTTP cost together.

## 91. Table 5.2 — row 4, column 3

**Before 1:**

(Added)

**After 1:**

> Verify scheme stability and labeled detection after refitting.

## 92. Table 5.2 — row 5, column 3

**Before 1:**

(Added)

**After 1:**

> Client ownership improves service; admission order requires a separate contract.

## 93. Table 5.2 — row 6, column 3

**Before 1:**

(Added)

**After 1:**

> Pair ceiling recall with specificity and useful corrections.

## 94. Table 5.2 — row 7, column 3

**Before 1:**

(Added)

**After 1:**

> Low invocation counts need useful selected repairs.

## 95. Table 5.2 — row 8, column 3

**Before 1:**

(Added)

**After 1:**

> Separate input sensitivity, computation cost and ranking mechanisms.
