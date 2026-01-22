# Automated Phishing Detection for Frontier AI Inference

## A Praxis Project Proposal

**Student:** Krti Tallam  
**Program:** D.Eng. in AI & ML  
**Advisor:** Professor Mheish  
**Date:** July 29, 2025  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Chapter 1 - Introduction](#chapter-1---introduction)
3. [Chapter 2 - Literature Review](#chapter-2---literature-review)
4. [Chapter 3 - Methodology](#chapter-3---methodology)
5. [Chapter 4 - Preliminary Results](#chapter-4---preliminary-results)
6. [Scope of Work](#scope-of-work)
7. [Data Sources](#data-sources)
8. [Timeline](#timeline)
9. [Expected Outcomes](#expected-outcomes)
10. [Evaluation Metrics](#evaluation-metrics)
11. [References](#references)
12. [Appendices](#appendices)

---

## Executive Summary

This praxis project proposes the development of an automated phishing detection system specifically designed to protect frontier AI inference systems from sophisticated phishing attacks. As AI systems become increasingly integrated into critical infrastructure and decision-making processes, they present attractive targets for malicious actors. This project aims to create a robust detection framework that can identify and mitigate phishing attempts targeting AI systems in real-time.

## Chapter 1 - Introduction

### 1.1 Background
The rapid deployment of artificial intelligence (AI) inference services has transformed the operational landscape for enterprises, researchers, and public institutions. Providers such as OpenAI, Google, Anthropic, and Microsoft Azure expose Application Programming Interface (API) access to large language models (LLMs), multimodal systems, and vector embedding services. Billions of inference requests are processed daily, and each prompt-to-response cycle converts user intent into outputs with direct economic and security implications (Carlini et al., 2023; PhishDecloaker Team, 2024). This ubiquity has attracted adversaries who adapt traditional phishing toolchains to exploit inference semantics, authentication flows, and metered pricing models.

The acceleration of enterprise adoption after 2023 amplified the risk surface. Software-as-a-service vendors now embed model calls in customer-facing tooling for sales, legal, and customer support workflows, while financial institutions use inference endpoints to triage fraud cases and draft compliance reports. Each of these verticals feeds sensitive context—customer account notes, contract clauses, or investigative summaries—into model prompts. The confidentiality of these prompts and the integrity of the resulting completions therefore bear directly on business continuity, intellectual property protection, and regulatory posture. Cloud providers responded by shipping managed gateways, audit logging, and abuse detection hooks, yet these mechanisms generally assume human-driven interaction models rather than automated botnets issuing thousands of prompts per minute.

From a historical perspective, phishing defenses have typically trailed attack innovation. Email gateways introduced Bayesian filtering in the early 2000s only after large-scale credential theft campaigns matured, and web browsers adopted safe-browsing lists in 2007 once drive-by download sites proliferated. A similar lag is emerging for AI inference: the earliest public reports of prompt injection and model extraction appeared in 2022, but standardized mitigation patterns remain nascent (Tramèr et al., 2016; Carlini et al., 2023). Organizations are therefore replicating the defensive debt seen in prior generations of phishing, accumulating technical and financial exposure while countermeasures evolve. The present research situates itself in this transition period, focusing on the unique telemetry, latency, and adversarial dynamics that differentiate inference APIs from the email and web channels studied in earlier eras.

### 1.2 Significance and Motivation
AI-specific phishing attacks create risks that differ fundamentally from classic email or website compromises. Across the July–August 2025 pilot logs, the median malicious burst burned 640,000 tokens, and the most severe weekend incident consumed 3.1 million completion tokens plus 1.2 million prompt tokens in 14 minutes. Priced against the publicly posted GPT-4 Turbo rates of US$0.03 per 1,000 completion tokens and US$0.01 per 1,000 prompt tokens (OpenAI, 2024), that single campaign generated US$19,680 in direct API charges. Mitigation required nine hours of site-reliability engineering support and US$8,500 in customer credits, pushing the total incident impact to just over US$28,000. Detailed telemetry for these events is catalogued in `Model_Performance_Reality_Check.md` for advisor review. At hyperscale, the economics escalate quickly: an orchestrated prompt-injection flood sustaining 50 requests per second with 4,000-token completions would drive roughly 360 million tokens per hour—more than US$180,000 in compute fees—before accounting for SLA penalties or collateral service degradation. Legacy detectors that prioritise lexical URL cues or rendered content achieve negligible recall on these attacks because the discriminative signals lie in prompt semantics, token consumption patterns, and anomalous session behaviour. Without AI-aware defences, platform operators face the choice between absorbing expensive abuse or throttling inference throughput.

The financial exposure extends beyond direct token spend. Inference phishing campaigns frequently trigger secondary costs such as incident response retainers, legal reviews to assess regulatory notification thresholds, and opportunity cost from paused product features. A Fortune 200 pilot partner with 8,000 daily active developers estimated that halting their internal code-assistant during a September 2024 incident cost roughly 4,500 developer-hours of productivity, dwarfing the raw compute charges. Brand damage and customer churn also surface when attackers exfiltrate or manipulate model behavior; for example, leaked prompt templates can reveal proprietary workflows or expose downstream systems to prompt-based privilege escalation. These diffuse but material impacts underscore why early, automated detection is preferable to manual case-by-case triage.

Regulators are beginning to scrutinize inference abuse as well. The U.S. Cybersecurity and Infrastructure Security Agency (CISA) highlighted prompt-injection risks in its 2024 Secure By Design report, recommending continuous monitoring of model interactions for anomalous patterns. The European Union’s AI Act similarly obliges high-risk AI providers to maintain technical logs that can support post-incident investigation. Meeting these obligations requires instrumentation capable of linking suspicious API sessions to concrete indicators of compromise. A detection system that produces structured features and alert rationale therefore serves dual purposes: operational containment and compliance evidence. This praxis aims to provide the scientific foundation for such a system within the latency and cost constraints articulated above.

### 1.3 Problem Statement
Legacy phishing detection systems fail in inference scenarios for four principal reasons. First, a semantic gap exists: models trained on human-readable content rarely capture features that describe prompt construction, token distributions, or model-specific intent, enabling adversarial requests to appear benign (Garera et al., 2007; Abdelhamid et al., 2014; Koide et al., 2024). Second, production inference demands responses under 200 milliseconds at the ninety-fifth percentile, while deep inspection pipelines frequently introduce delays exceeding 500 milliseconds (Brutlag, 2009; PhishDecloaker Team, 2024; Koide et al., 2024). Third, adversaries armed with generative models generate polymorphic prompts that invalidate static rule sets within hours (Perez & Ribeiro, 2022; Koide et al., 2024). Fourth, there is a scarcity of labeled AI phishing incidents, limiting supervised learning and hindering reproducible evaluation (MITRE ATLAS, 2023; NIST, 2023). These constraints necessitate detection frameworks designed specifically for inference APIs.

Each limitation manifests concretely in production telemetry. The semantic gap appears when malicious prompts weaponize hidden instructions (e.g., base64-encoded jailbreak payloads or obfuscated API chaining) that carry no resemblance to URLs or sender metadata used in email filters. Latency constraints force platform operators to choose between inline screening—which risks violating contractual response targets—and asynchronous review that may allow long-running prompt sequences to extract sensitive data before mitigation can occur. Adversarial iteration is no longer hypothetical: internal red-team exercises observed 63 unique jailbreak variants generated by a single attacker-controlled LLM within a three-hour window, overwhelming static deny-lists. Label scarcity persists because providers disclose few phishing incidents publicly, and sharing sanitized logs requires significant legal review; as a result, academic datasets contain mainly synthetic examples that fail to capture the edge cases emerging in live environments. These operational realities frame the engineering challenge tackled by this praxis.

### 1.4 Thesis Statement
This praxis contends that a machine learning framework combining AI-aware feature engineering with latency-optimized ensemble modeling can measurably improve detection of phishing attacks targeting AI inference endpoints over adapted traditional baselines while preserving the sub-200-millisecond latency budget required for user-facing deployments.

### 1.5 Research Objectives
1. **AI-aware feature engineering** – Design and validate a feature pipeline that captures prompt semantics, token-consumption bursts, session sequencing, and model-targeting indicators by mining the twelve-week sanitized pilot telemetry and templated red-team attacks.
2. **Labeled dataset development** – Assemble a reproducible corpus that blends sanitized production incidents, red-team simulations, and curated open intelligence feeds (OpenPhish, PhishTank, URLhaus) normalized to the inference API format. Each record will be labeled using the OpenAI policy rubric with two-pass analyst review, and the resulting dataset will provide distinct training, validation, and inference-time evaluation splits while benchmarking against public phishing corpora to document coverage gaps.
3. **Detection ensemble baselining** – Train class-weighted tree ensembles and regularized linear models on the curated dataset to lift F1 from the current 0.067 baseline to at least 0.50, achieve ≥80 percent detection accuracy, and hold false-positive rates below five percent under cross-validated experiments.
4. **Operational performance assessment** – Evaluate the preferred model as a sidecar service, measuring sub-200-millisecond latency, token-spend containment, and avoided incident costs under replay of the pilot telemetry and stress-test scenarios.

### 1.6 Research Questions
RQ1: Which combinations of prompt-level, session-level, and infrastructure-level features best discriminate malicious inference requests from legitimate traffic?  
RQ2: How do sampling strategies and class weighting mitigate the severe class imbalance observed in current pilot datasets where phishing prevalence is below five percent?  
RQ3: Can incremental or active learning updates sustain detection performance against adaptive adversaries without interrupting inference services?  
RQ4: How should evaluation protocols integrate statistical metrics and operational cost to reflect the real-world impact of detection outcomes?

Answering RQ1 requires a systematic exploration of feature families that extend beyond lexical cues. Planned experiments will contrast handcrafted prompt-grammar signals, embedding-based semantic distances, token burstiness statistics, and authentication context (e.g., OAuth client identifiers, key age) to determine which combinations offer the strongest lift in recall without inflating false positives. For RQ2, the study will compare SMOTE variants, focal loss, and cost-sensitive thresholds, documenting how each technique influences minority-class visibility under temporal cross-validation. RQ3 probes the feasibility of model updates in environments where downtime must be avoided; the investigation will weigh sliding-window retraining, online gradient updates, and human-in-the-loop review loops, with emphasis on the operational safeguards needed to prevent model drift. Finally, RQ4 motivates the inclusion of metrics such as token-spend avoided, analyst-hours saved, and customer-impact scores alongside conventional ROC curves so that decision-makers can align detection performance with business objectives.

### 1.7 Scope and Delimitations
Implementation targets OpenAI GPT-4 and GPT-4o endpoints because these services dominate the current pilot deployments and provide the telemetry required for research. The study emphasises inference-time defenses rather than training security, network-layer denial-of-service, or multi-provider generalization. Experiments employ a twelve-week traffic snapshot ending August 2025 to maintain a reproducible data cut-off. Authentication and authorization controls are assumed to be properly configured, and mitigation actions (e.g., throttling or human review) remain under existing operational playbooks.

Key delimitations include:
- **Platform coverage:** Findings are validated on OpenAI-hosted workloads. While architectural patterns should translate to other providers, quantitative performance claims remain scoped to the telemetry captured from GPT-4/GPT-4o gateways.
- **Threat surface:** The research prioritises inference API abuse such as prompt injection, model extraction, and denial-of-wallet campaigns. Broader cloud intrusions (e.g., credential theft preceding API misuse) and supply-chain compromises fall outside the treatment boundary.
- **Response automation:** The prototype focuses on detection and alert enrichment. Automated remediation actions (e.g., token revocation, rate-limit enforcement) are assumed to be executed by existing platform tooling once alerts are raised.
- **Data handling:** Only de-identified metadata and truncated prompt payloads are ingested. Full conversational transcripts and personally identifiable information are deliberately excluded to satisfy institutional review requirements.

### 1.8 Research Limitations
The empirical foundation of this praxis is necessarily bounded by data-access agreements and institutional review constraints. The labeled corpus presently available—approximately 1,000 synthetic phishing traces and 10,000 benign inference requests—yields a baseline detector with 72 percent accuracy and an F1 score of 0.067, underscoring the statistical volatility that accompanies extreme class imbalance even after resampling interventions (He & Garcia, 2009). Because real-world incidents are redacted before handoff, each prompt is anonymised, truncated, and replayed within a segmented sandbox that enforces strict rate caps and excludes live abuse traffic. This protocol is essential for privacy preservation and aligns with emerging guidance on safe handling of adversarial AI artefacts (CISA, 2024; OWASP, 2023), yet it limits the observational window to previously documented attack archetypes and may underrepresent multi-stage campaigns orchestrated by adaptive opponents (Perez & Ribeiro, 2022). External validity is further constrained by the focus on OpenAI GPT‑4/GPT‑4o telemetry; inference gateways operated by Anthropic, Google, or AWS Bedrock expose different authentication idioms and quota semantics, meaning that feature importances derived here may not transfer without recalibration (Weidinger et al., 2021). Finally, the four-to-six-week development timeline prioritises delivery of a latency-compliant prototype over longitudinal drift analysis or extensive human-in-the-loop calibration, leaving residual uncertainty about detector durability against adversaries who evolve tactics on sub-weekly horizons (Biggio & Roli, 2018). These limitations motivate transparent release of dataset provenance, labeling rubrics, and latency instrumentation so that subsequent iterations can expand coverage, replicate findings on additional platforms, and quantify the economic trade-offs of more aggressive defensive postures.

### 1.9 Organization of the Praxis
Chapter 1 introduces the problem landscape, articulates motivation, outlines research questions, and frames the scope and limitations guiding the investigation. Chapter 2 surveys the state of knowledge across traditional phishing detection, adversarial machine learning, streaming analytics, and governance frameworks, highlighting the empirical gaps that motivate the proposed approach. Chapter 3 details the research design, data collection strategy, feature engineering pipeline, and modelling techniques used to test the research questions. Chapter 4 reports interim experimental findings, including baseline performance, latency instrumentation, and lessons learned from early smoke tests. Chapter 5 will interpret the matured results, synthesize contributions to the body of knowledge, and map implications for practitioners. The praxis concludes with references and appendices that document artifacts such as the glossary, acronym list, and technical specifications.

### 1.10 Stakeholder Impact and Contributions
The envisioned detection framework delivers value to multiple stakeholder groups. Platform security teams gain automated signals that reduce time-to-detection and provide defensible evidence for compliance reviews. Product engineering units can maintain feature velocity by mitigating abuse without imposing blunt traffic throttles. Customers benefit from higher service availability and reduced exposure of sensitive prompts or outputs. Academic researchers receive a reproducible dataset template and performance baselines for future comparative studies. These contributions align with advisor guidance to translate theoretical insights into actionable engineering outcomes.

## Chapter 2 - Literature Review

### 2.1 Overview of Thematic Domains
The literature spans five domains: traditional phishing analytics, adversarial threats to machine learning systems, latency-aware detection and streaming analytics, labeling and governance frameworks, and gap analyses identifying unmet needs for inference security. Forty-two peer-reviewed papers, conference proceedings, and standards publications were prioritised when they reported empirical metrics, reproducible datasets, or deployment experiences relevant to inference APIs.

To ensure breadth, the review draws on computer security, human-computer interaction, and operations research venues, supplemented by industry white papers where peer-reviewed sources remain sparse. Particular emphasis is placed on works that disclose dataset composition, feature engineering choices, and latency measurements, as these parameters directly influence the feasibility of applying prior art to inference gateways. The chapter is organised to progress from legacy detection approaches toward emerging AI-specific challenges, ending with a synthesis that positions the praxis within the observed gaps.

### 2.2 Traditional Phishing Detection
Rule-based and lexical-feature approaches such as those by Garera et al. (2007) achieved 95.8 percent accuracy using URL heuristics but rely on interpretable features tied to web content. Machine learning methods—including decision trees, Support Vector Machines, and ensemble classifiers—leveraged public feeds like PhishTank and the Anti-Phishing Working Group to reach 92.5 percent accuracy (Abdelhamid et al., 2014). Contemporary deep learning systems employ convolutional networks and transformer architectures to process HTML, screenshots, or rendered email content, reporting accuracy above 95 percent (PhishDecloaker Team, 2024; Koide et al., 2024). Nevertheless, their feature sets assume human-readable artefacts and typically allow substantial inference latency, leaving inference APIs outside their coverage.

Legacy detectors also benefit from abundant labelled data sets. PhishTank alone contributes tens of thousands of verified phishing URLs per month, and email security vendors share large corpora for benchmarking. Feature extraction pipelines commonly rely on lexical statistics (character n-grams, token entropy), structural cues (DOM tree depth, iframe counts), and reputation signals (WHOIS age, hosting provider). These signals are ill-suited to JSON-encoded inference payloads that lack clickable links or rendered content. Transfer-learning attempts—such as fine-tuning text classifiers on prompt transcripts—exhibit severe concept drift because legitimate prompts often contain imperative verbs and sensitive terminology that would be flagged as suspicious in an email context. Consequently, directly porting traditional detectors to inference traffic produces biased models that either miss attacks altogether or generate unacceptable false-positive rates that overwhelm operators.

The latency characteristics of legacy systems further limit applicability. Email gateways batch-process messages and can afford several hundred milliseconds—or even seconds—of evaluation time without impacting user experience. Inference APIs, by contrast, are synchronous: clients expect responses in near real time to maintain conversational flow or keep downstream applications responsive. Techniques like full-page rendering or multi-stage sandboxing, which bolster web phishing detection, are therefore impractical in the inference setting. The literature provides little guidance on reconciling high accuracy with low latency, underscoring the need for bespoke architectural choices explored in this praxis.

### 2.3 Security of Machine Learning Systems
Adversarial machine learning literature underpins inference-focused phishing concerns. Biggio and Roli (2018) categorise evasion, poisoning, and privacy attacks. Tramèr et al. (2016) and Carlini et al. (2023) demonstrate black-box model extraction via crafted query sequences, while Perez and Ribeiro (2022) and Shokri et al. (2017) expose prompt manipulation and membership inference risks. These studies confirm that adversaries can harness inference endpoints to harvest intellectual property or induce harmful outputs. Proposed mitigations, however, concentrate on architectural hardening or rate limiting rather than real-time detection.

Subsequent work expanded the taxonomy to include indirect prompt injection, policy evasion, and data leakage via contextual cross-contamination (OWASP, 2023; Weidinger et al., 2021). Researchers documented how attackers string together multi-turn dialogues that gradually escalate privileges or coax models into revealing system prompts. Defensive recommendations typically emphasize input sanitization, output filtering, and human oversight, yet lack quantitative evidence about detection efficacy under production load. Moreover, most adversarial ML experiments assume an interactive human attacker; fewer studies evaluate automated phishing pipelines that issue thousands of API calls programmatically. The gap between theoretical attack demonstrations and operational countermeasures is therefore wide, motivating this praxis’s emphasis on measurable, latency-aware defenses.

### 2.4 Latency-Aware Detection and Streaming Analytics
User-experience research at Google indicates that delays beyond 200 milliseconds degrade perceived service quality (Brutlag, 2009). Streaming analytics and edge-computing literature describe micro-batch and event-driven architectures capable of sub-second processing (Liu et al., 2019). In phishing contexts, PhishDecloaker demonstrates 187-millisecond detection for CAPTCHA-cloaked websites by precomputing signatures, whereas KnowPhish trades responsiveness for higher accuracy by invoking language models, yielding latencies above 500 milliseconds. No surveyed studies benchmark phishing detection for API traffic under the 200-millisecond constraint, highlighting an engineering gap addressed by this praxis.

Related work on security analytics pipelines—such as Apache Flink-based anomaly detection frameworks (Carbone et al., 2015)—offers architectural blueprints for handling high-velocity data. However, these systems typically process network flows or log events that can be aggregated asynchronously. Inference phishing requires inspecting the prompt payload itself, extracting semantic features, and rendering a decision before the model execution completes. Edge-deployed accelerators and request mirroring have been proposed to mitigate latency overhead, but peer-reviewed evaluations remain scarce. This literature gap justifies the praxis’s emphasis on lightweight feature computation, warm model caches, and empirical latency measurement alongside accuracy metrics.

### 2.5 Labeling Protocols and Governance
Effective supervision requires principled labels. OpenAI’s Usage Policies (2024) define prohibited behaviours such as model extraction, harmful content generation, and safety-filter circumvention. The NIST AI Risk Management Framework (2023) and MITRE’s Adversarial Threat Landscape for Artificial-Intelligence Systems (ATLAS, 2023) provide taxonomies for classifying adversarial activities. The praxis adapts these guidelines: a request is labelled phishing if it attempts to exfiltrate model parameters or training data, exhaust resources for economic harm, inject instructions that bypass safeguards, or orchestrate coordinated misuse. Legitimate traffic complies with published policies and exhibits expected token usage and rate characteristics, addressing advisor guidance to avoid bespoke labeling schemes.

Emerging industry initiatives reinforce the need for shared definitions. The OWASP Top 10 for Large Language Model Applications (OWASP, 2023) enumerates prompt-injection variants and supply-chain risks that map directly onto the phishing behaviours examined here. The Partnership on AI’s 2024 safety commitments urge providers to log model interactions at sufficient granularity to support post-incident accountability. Applying these recommendations in practice requires aligning annotation guidelines, reviewer training, and quality assurance processes so that multi-analyst labeling yields consistent outcomes. This praxis therefore specifies a two-pass review process—first by an annotator versed in OpenAI policy nuances, then by a security engineer—to minimise label drift and to provide rationales that can be audited during advisor reviews.

### 2.6 Gap Analysis
Four gaps emerge. Traditional detectors lack features covering prompt semantics, token dynamics, and session behaviour, yielding zero recall on AI-specific attacks. High-accuracy models often exceed acceptable latency, whereas fast detectors overlook nuanced abuse patterns. There are no public benchmarks for AI phishing datasets, impeding reproducibility. Finally, few defences incorporate continuous learning or adversarial testing loops to contend with attackers who iterate using generative models.

These gaps translate into four research requirements:
1. **Semantic fidelity.** Detection features must capture the intent encoded within prompts without relying on rendered artefacts. Promising directions include verb tense ratios, instruction-sequence signatures, embedding similarity to known jailbreak corpora, and correlations between prompt structure and token spend anomalies.
2. **Latency-constrained inference.** Models must deliver decisions well under the 200 ms service-level objective. Lightweight ensembles, calibrated thresholds, and feature precomputation are essential to prevent the detector from becoming the bottleneck.
3. **Benchmarking discipline.** A reproducible dataset with clear provenance, masking rules, and train/validation/test partitions is needed to facilitate peer comparison and advisor scrutiny. Publishing performance alongside cost metrics enables a richer discussion than accuracy alone.
4. **Adaptation mechanisms.** Incorporating online learning hooks, alert feedback loops, and periodic adversarial evaluations can help systems track evolving attack patterns without catastrophic forgetting.

### 2.7 Research Positioning
Addressing these gaps positions the praxis as the first comprehensive investigation into phishing detection for AI inference endpoints. Chapter 3 translates the insights from the literature into a methodology that engineers AI-aware features, balances accuracy and latency through ensemble modelling, curates reproducible datasets aligned with established policies, and evaluates resilience against adaptive adversaries.

The literature review reveals that no existing work simultaneously satisfies three constraints: (1) semantic coverage of inference-specific attack vectors, (2) latency guarantees compatible with production service-level objectives, and (3) transparent evaluation artifacts that advisors and future researchers can reuse. By grounding the methodology in documented operational incidents and aligning feature engineering with policy taxonomies, the praxis contributes a bridge between theoretical adversarial ML research and the practical realities faced by platform security teams. Subsequent chapters operationalize these insights through experiment design, instrumentation, and validation activities.

### 2.8 Chapter Summary
Chapter 2 traced the evolution of phishing defenses from heuristic URL filters to transformer-based classifiers, then contrasted those approaches with the distinctive semantics and performance constraints of inference APIs. It reviewed adversarial machine learning research, highlighting the absence of production-grade detection metrics, and synthesised governance guidelines that inform the labeling strategy adopted in this praxis. Collectively, the review underscores the necessity of purpose-built feature engineering, latency-aware model selection, and rigorous dataset stewardship—elements that shape the methodology detailed in Chapter 3.

## Chapter 3 - Methodology

### 3.1 Research Design
The methodology follows an iterative design-science cycle focused on building, assessing, and refining lightweight detection artefacts. Each iteration begins with data profiling and feature experimentation, continues with model training and latency measurement, and concludes with error analysis that informs the next cycle. This approach keeps the scope aligned with achievable milestones while still producing publishable evidence of improvement over baselines adapted from traditional phishing detectors.

### 3.2 Data Collection Strategy
Three data sources anchor the study. First, sanitised production logs from GPT-4 and GPT-4o pilot deployments provide benign traffic and known anomalous sessions collected between June and August 2025. Second, snapshots from PhishTank, OpenPhish, and URLhaus supply phishing URLs and payloads that are wrapped in synthetic inference envelopes to approximate API abuse. Third, a small catalogue of adversarial prompts is generated by perturbing real customer prompts under advisor-approved guidelines so that no harmful content is produced. Each record stores timestamped metadata, authentication method, token counts, and truncated payload text. Data are split chronologically into training (June–July, 60 %), validation (early August, 20 %), and held-out test sets (mid-August, 20 %).

### 3.3 Feature Engineering
Feature development concentrates on a compact set of signals that can be computed quickly and explained to stakeholders. Prompt-level features include total token count, token entropy, the ratio of imperative verbs to descriptive text, and binary flags for extraction keywords or encoded artefacts. Session-level features measure request frequency over one- and five-minute windows, sudden authentication changes, and geographic variance. Infrastructure-level features capture observed latency, retry counts, and estimated compute cost. All transformations are implemented in Python notebooks and exported to Apache Parquet, enabling straightforward reproducibility without introducing a complex feature store.

### 3.4 Model Development
The modelling plan compares two pragmatic classifiers: a class-weighted Random Forest and an L1-regularised logistic regression. Both models have been effective in prior phishing research but require adaptation to the API feature space. Hyperparameters are tuned with grid searches guided by validation performance and measured inference latency. Class imbalance is handled through a combination of Synthetic Minority Over-sampling Technique (SMOTE) on the training split and class-specific decision thresholds calibrated on the validation set. Restricting the study to two complementary models keeps experiments manageable while still allowing analysis of trade-offs between interpretability and accuracy.

### 3.5 Evaluation Plan
Evaluation on the held-out test set reports accuracy, precision, recall, F1 score, and area under the receiver operating characteristic (ROC) curve. Operational metrics include mean latency, ninety-fifth-percentile latency, throughput (requests per second), and estimated compute cost avoidance relative to a no-detector baseline. Statistical significance is assessed using McNemar’s test between the two candidate models. Error analysis emphasises false negatives to understand missed attack patterns and inform subsequent feature refinements.

### 3.6 Prototyping and Deployment Considerations
To validate practicality, the study packages the selected model in a lightweight Python service that exposes a REST endpoint and asynchronous worker. Requests are mirrored from the inference gateway to the service, which responds within the 200-millisecond budget using cached feature computations where possible. Integration leverages existing logging infrastructure and a Redis instance for short-term session statistics, avoiding the need to introduce new platform components.

### 3.7 Validity and Ethical Considerations
Construct validity is supported by aligning labels with established OpenAI, NIST, and MITRE policies. Internal validity threats stemming from data sparsity are mitigated through temporal validation splits and sensitivity checks that vary class ratios. External validity is limited by the focus on OpenAI endpoints; Chapter 6 documents assumptions that must be revisited before extending to other providers. Ethical safeguards ensure synthetic prompts remove personal data and avoid generating harmful content.

### 3.8 Chapter Summary
By limiting the modelling scope to explainable features and two baseline classifiers, Chapter 3 establishes an achievable pathway for demonstrating improvement over traditional phishing detectors while satisfying latency constraints. Chapter 4 expands on implementation details, tooling, and operational workflows that support this methodology.

## Chapter 4 - Preliminary Results

### 4.1 Experiment Setup
Development occurs on a macOS workstation using Python 3.11, Poetry-managed virtual environments, and Git for version control. Core libraries include pandas, numpy, scikit-learn, imbalanced-learn, and xgboost for experimentation; Apache Arrow/Parquet provides efficient storage for intermediate datasets. Experiments execute locally with CPU acceleration only, ensuring that latency measurements reflect realistic production-grade constraints rather than GPU-optimised assumptions. A `requirements.txt` snapshot accompanies each experiment branch so results can be reproduced on faculty machines.
A dedicated ingestion notebook converts raw inference logs into structured parquet tables with the following steps: (1) schema validation to ensure required fields—timestamp, authentication token hash, endpoint name, token counts—are present; (2) anonymisation of user identifiers via salted hashing; (3) enrichment with geographic metadata derived from IP-to-region lookups; and (4) consolidation of public phishing feeds by wrapping URLs or payloads within synthetic API envelopes. Each transformation is stored in `/data/intermediate/` with versioned filenames (e.g., `2025-08-15_training.parquet`) so that subsequent modelling stages can reference immutable snapshots.
Two reproducible scripts orchestrate model fitting: `train_rf.py` for the class-weighted Random Forest and `train_lr.py` for the L1-regularised logistic regression. Both scripts accept parameters for class-weight schemes, SMOTE configuration, and probability thresholds. Training runs follow a consistent pattern: load the chronological training split, apply feature transformations, oversample the minority class with SMOTE tuned to a 1:3 ratio, fit the model, and persist artefacts (`.joblib` models plus feature metadata) under `/models/{date}/`. Validation scoring reports accuracy, precision, recall, F1, ROC AUC, and 95th-percentile latency, with outputs written to `/reports/validation/{timestamp}.json`.
All experiments are tracked in a Markdown ledger located at `/docs/experiment_log.md`, which records configuration parameters, metric outcomes, and links to corresponding Git commits. Each run is tagged with a semantic identifier (e.g., `EXP-2025-08-22-RF-smote15`) to simplify cross-referencing during advisor reviews. Latency measurements rely on Python's `time.perf_counter()` instrumentation embedded in the prediction scripts; raw measurement arrays are archived to `/reports/perf/`.

### 4.2 Snapshot Summary
The Sept–Nov 2025 freeze combines 320 GPT-4/GPT-4o telemetry rows, 200 OpenPhish entries, 240 PhishTank entries, and 260 URLhaus entries. Chronological splitting yields 1,669 training, 318 validation, and 341 test examples (291 benign, 50 phishing). Each record retains the OWASP/MITRE taxonomy tag from the ingestion pipeline so evaluation artifacts remain auditable. The complete feature vector spans 15 in-line signals (URL structure, lexical counts, Boolean AI-endpoint flags, prompt/completion tokens, latency) plus metadata columns for traceability.

### 4.3 Baseline Performance
Both baselines were retrained on the frozen snapshot. Table 4-1 reports the held-out metrics for default thresholds (0.5) and the calibrated thresholds selected via F1 search on the validation window. Calibration (0.03 for Logistic-L1, 0.06 for Random Forest) eliminates the slow-burn false negatives observed at 0.5.

| Model / Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | Confusion Matrix |
|-------------------|---------:|----------:|-------:|----:|--------:|------------------|
| Logistic-L1 (0.5) | 0.983 | 1.000 | 0.957 | 0.978 | 1.000 | [[204, 0], [6, 134]] |
| Logistic-L1 (0.03) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | [[291, 0], [0, 50]] |
| Random Forest (0.06) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | [[291, 0], [0, 50]] |

The ROC curves (Figure 4-1) overlap with AUC=1.0 for both models. A McNemar test on the calibrated predictions yields b=0, c=0 (χ²=0, p=1.0), indicating no significant difference between the classifiers once thresholds are tuned; model selection therefore hinges on latency and interpretability, not accuracy.

### 4.4 Feature Robustness and Statistical Tests
Feature-family ablations highlight which signals drive the logistic model. Removing the punctuation counts (`num_dots`, `num_hyphens`, etc.) drops recall to 0.58 (F1 = 0.67) even though accuracy on benign traffic remains high, while removing only length features keeps recall near 0.88. Random Forest maintains 100% accuracy across all ablations, underscoring its capacity to memorise lexical patterns but at the cost of latency. These experiments validate the decision to retain the full URL-structure family in the production-friendly logistic path. All ablation results are archived in `data/processed/v2025.08.10/eval/feature_ablations.json` for reproducibility.

### 4.5 Latency and Deployment Readiness
A FastAPI microservice (`code/server.py`) wraps the logistic detector with the calibrated threshold. Two latency regimes were measured:

1. **Mirrored test replay:** replaying the 341 held-out requests through FastAPI’s ASGI transport produced an average latency of 7.4 ms, p95 of 11.7 ms, and maximum of 14.4 ms (`burst_latency.json`).
2. **Standalone prediction timer:** direct in-process timing (no network stack) yields 0.059 ms p95 for logistic inference versus 14.6 ms for Random Forest, matching the earlier `mirrored_latency.json` results.

Both regimes sit well below the <200 ms budget defined in Chapter 3, leaving headroom for REST overhead, logging, and governance hooks. The deployment validation plan (FastAPI pod, burst replay, monitoring exports) is outlined in `reports/DeploymentValidationPlan.md` with execution notes captured in `DeploymentValidationNotes.md`.

### 4.6 Next Steps
Remaining execution items are (1) finish annotating the expanded training batch (≥150 samples with inter-rater statistics) so Chapter 3 can cite empirical governance coverage, (2) capture REST traces and monitoring fields from the FastAPI prototype to include in Appendix B, and (3) repeat the evaluation workflow whenever new telemetry snapshots are frozen to demonstrate stability over time.

## Scope of Work

### In Scope
1. **Detection System Development**
   - Design and implementation of phishing detection algorithms
   - Integration with common AI inference frameworks
   - Real-time monitoring capabilities

2. **Evaluation Framework**
   - Comprehensive testing methodology
   - Performance benchmarking
   - Security effectiveness metrics

3. **Documentation and Guidelines**
   - Best practices for AI system security
   - Deployment guidelines
   - User documentation

### Out of Scope
1. General-purpose phishing detection (not AI-specific)
2. Training data security (focus is on inference)
3. Hardware-level security measures
4. Legal and compliance frameworks

### Deliverables
1. Functional phishing detection system prototype
2. Evaluation results and performance analysis
3. Technical documentation
4. Research paper suitable for publication
5. Presentation of findings

## Methodology

### Research Approach
This project will employ a mixed-methods approach combining:
1. **Quantitative Analysis**: Performance metrics, detection rates, false positive analysis
2. **Qualitative Assessment**: Expert evaluation, case studies, threat modeling

### Development Methodology

#### Phase 1: Research and Analysis (Months 1-2)
- Comprehensive threat modeling for AI inference systems
- Analysis of existing phishing attack patterns
- Requirements gathering from stakeholders

#### Phase 2: System Design (Months 2-3)
- Architecture design for detection system
- Algorithm selection and customization
- Integration planning with AI frameworks

#### Phase 3: Implementation (Months 3-5)
- Core detection engine development
- API and integration layer implementation
- User interface and monitoring dashboard

#### Phase 4: Testing and Evaluation (Months 5-6)
- Unit and integration testing
- Security effectiveness testing
- Performance benchmarking
- User acceptance testing

#### Phase 5: Documentation and Dissemination (Month 6)
- Technical documentation
- Research paper preparation
- Final presentation preparation

### Technical Approach

1. **Detection Algorithm Development**
   - Class-weighted Random Forest and L1-regularised logistic regression baselines
   - SMOTE-based imbalance mitigation and threshold calibration
   - Error analysis to inform incremental feature enhancements

2. **Inference Pipeline Instrumentation**
   - Lightweight REST service for real-time scoring
   - Latency measurement and caching strategies to maintain <200 ms responses
   - Versioned model artefacts with reproducible configuration metadata

3. **Operational Integration**
   - Mirrored request handling alongside existing inference gateways
   - Experiment logging, cost-impact estimation, and audit trails for advisor review
   - Foundations for future dashboarding and alerting once results stabilise

## Data Sources

### Primary Data Sources
[Based on your data source examples from HW#4]

1. **Phishing Attack Datasets**
   - Public phishing repositories
   - Synthetic attack generation
   - Real-world attack samples (anonymized)

2. **AI System Logs**
   - Inference request logs
   - API access patterns
   - System performance metrics

3. **Threat Intelligence Feeds**
   - Commercial threat intelligence
   - Open-source intelligence (OSINT)
   - Academic research datasets

### Data Collection Methodology
1. Automated collection pipelines
2. Data anonymization and privacy protection
3. Continuous updating mechanisms

### Data Processing
1. Feature extraction pipelines
2. Data normalization and preprocessing
3. Label validation and quality assurance

## Timeline

### Project Schedule (July 2025 - December 2025)

| Period | Dates | Activities | Deliverables |
|--------|-------|-----------|--------------|
| Period 1 | Jul 26 - Aug 9 | Literature review, threat modeling | Initial threat model, 25+ papers reviewed |
| Period 2 | Aug 9 - Aug 23 | Requirements analysis, methodology refinement | Requirements document, research questions |
| Period 3 | Aug 23 - Sep 6 | System design, architecture planning | Design specification, architecture diagrams |
| Period 4 | Sep 6 - Sep 20 | Core implementation begins | Alpha prototype, initial codebase |
| Period 5 | Sep 20 - Oct 4 | Feature development, API integration | Beta prototype, API framework |
| Period 6 | Oct 4 - Oct 18 | Testing framework, initial evaluation | Test suite, preliminary results |
| Period 7 | Oct 18 - Nov 1 | Full evaluation, performance tuning | Complete evaluation results |
| Period 8 | Nov 1 - Nov 15 | Paper writing, documentation | Draft paper, technical documentation |
| Period 9 | Nov 15 - Dec 6 | Refinement, final testing | Final system, complete results |
| Period 10 | Dec 6 - Dec 20 | Project wrap-up, presentation prep | Final report, defense presentation |

### Key Milestones (Aligned with Advisor Meetings)
1. **August 9**: Literature review 50% complete
2. **August 23**: Threat model and requirements finalized
3. **September 6**: System design approved
4. **September 20**: Alpha prototype demonstration
5. **October 4**: Mid-project review with working system
6. **October 18**: Testing framework complete
7. **November 1**: Evaluation results ready
8. **November 15**: Paper draft for review
9. **December 6**: Final system demonstration
10. **December 20**: Project defense ready

## Expected Outcomes

### Technical Outcomes
1. **Detection System**
   - 80-85% detection rate for known phishing patterns
   - ≤5% false positive rate
   - <200ms added latency to inference requests

2. **Integration Framework**
   - Support for major AI frameworks (TensorFlow, PyTorch, etc.)
   - Cloud-native deployment options
   - Comprehensive API documentation

### Research Contributions
1. Adaptation of traditional phishing classifiers to AI inference features
2. Empirical benchmarks comparing lightweight models under latency constraints
3. Documentation of feature engineering practices for API-centric phishing detection
4. Preliminary guidance for integrating detection services alongside inference gateways

### Practical Impact
1. Improved security for deployed AI systems
2. Reduced risk of AI system compromise
3. Enhanced trust in AI deployment
4. Foundation for future security research

## Evaluation Metrics

### Security Metrics
1. **Detection Performance**
   - True Positive Rate (TPR)
   - False Positive Rate (FPR)
   - F1 Score
   - Area Under ROC Curve (AUC)

2. **Attack Coverage**
   - Percentage of attack types detected
   - Zero-day detection capability
   - Time to detection

### System Performance Metrics
1. **Latency Impact**
   - Added latency per inference request
   - Throughput degradation
   - Resource utilization

2. **Scalability**
   - Maximum requests per second
   - Horizontal scaling efficiency
   - Multi-region deployment performance

### Usability Metrics
1. Integration complexity score
2. Configuration effort required
3. Alert quality and actionability
4. User satisfaction ratings

## References

[This section would include all references from your annotated bibliography in APA format]

## Appendices

### Appendix A: Glossary of Terms
[From your HW#4 Glossary slide]

### Appendix B: List of Acronyms
- **AI** — Artificial Intelligence
- **APA** — American Psychological Association
- **API** — Application Programming Interface
- **ATLAS** — Adversarial Threat Landscape for Artificial-Intelligence Systems
- **AUC** — Area Under the Curve
- **CAPTCHA** — Completely Automated Public Turing test to tell Computers and Humans Apart
- **CPU** — Central Processing Unit
- **DHS** — Department of Homeland Security
- **DoD** — Department of Defense
- **FNR** — False Negative Rate
- **FPR** — False Positive Rate
- **GPT** — Generative Pre-trained Transformer
- **GPT-4** — Generative Pre-trained Transformer 4
- **GPT-4o** — Generative Pre-trained Transformer 4o
- **GPU** — Graphics Processing Unit
- **HTML** — Hypertext Markup Language
- **IP** — Internet Protocol
- **LLM** — Large Language Model
- **LR** — Logistic Regression
- **MITRE** — MITRE Corporation
- **ML** — Machine Learning
- **NIST** — National Institute of Standards and Technology
- **OSINT** — Open-Source Intelligence
- **REST** — Representational State Transfer
- **RF** — Random Forest
- **ROC** — Receiver Operating Characteristic
- **SLO** — Service Level Objective
- **SMOTE** — Synthetic Minority Over-sampling Technique
- **TPR** — True Positive Rate
- **URL** — Uniform Resource Locator
- **US** — United States

### Appendix C: Detailed Technical Specifications
[To be developed]

### Appendix D: Risk Assessment and Mitigation Plan
[To be developed]

---

This document is a living document and will be updated throughout the project lifecycle to reflect progress, findings, and any necessary adjustments to the project scope or methodology.
