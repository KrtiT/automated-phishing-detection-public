# Project Timeline
## Automated Phishing Detection for Frontier AI Inference

**Student:** Krti Tallam  
**Advisor:** Professor Mheish  
**Duration:** July 2025 - December 2025

---

## Project Schedule with Biweekly Milestones

| Meeting | Date | Period Activities | Key Deliverables |
|---------|------|-------------------|------------------|
| 1 | **Jul 26** | Project kickoff, initial planning | Project scope defined, repository created |
| 2 | **Aug 9** | Literature review, threat modeling | (Completed) Annotated bibliography (30+ sources), OWASP/MITRE threat model v1 |
| 3 | **Aug 23** | Requirements analysis, methodology refinement | (Completed) Week 1 snapshot plan, labeling protocol, evaluation framework |
| 4 | **Sep 6** | System design, architecture planning | (Completed) Data pipeline scripts, baseline training plan, advisor sign-off |
| 5 | **Sep 20** | Core implementation begins | (Completed) Sept–Nov data freeze materialized, 60-label audit log |
| 6 | **Oct 4** | Feature development, API integration | (Completed) Logistic/RF baselines + evaluation artifacts |
| 7 | **Oct 18** | Testing framework, initial evaluation | (In progress) Pytest harness + deployment monitoring artifacts |
| 8 | **Nov 1** | Full evaluation, performance tuning | (Completed) ROC/CM figures, calibration notes, burst latency |
| 9 | **Nov 15** | Paper writing, documentation | (In progress) Chapter 3–4 updates awaiting Word sync |
| 10 | **Dec 6** | Refinement, final testing | (Upcoming) Extended burst replay + monitoring hookups |
| 11 | **Dec 20** | Project wrap-up, presentation prep | (Upcoming) Final manuscript, defense deck |

---

## Detailed Phase Breakdown

### Phase 1: Foundation (Jul 26 - Aug 23)
- **Literature Review**: Comprehensive survey of 25+ papers
- **Threat Modeling**: AI-specific attack taxonomy
- **Requirements**: System specifications and constraints
- **Methodology**: Research approach and validation plan

### Phase 2: Design & Development (Aug 23 - Oct 4)
- **Architecture**: System design and component specification (completed Aug 23)
- **Prototype**: Alpha version with core functionality (Week 4 data pipeline + labeling audit)
- **Integration**: FastAPI service + ASGI replay harness standing in for TensorFlow Serving
- **Feature Engineering**: 15-feature inline vector + SMOTE pipeline implemented in `train_baselines.py`

### Phase 3: Testing & Evaluation (Oct 4 - Nov 15)
- **Test Suite**: Minimal pytest harness pending (target Nov 2025)
- **Performance**: Latency + mirrored replay benchmarks logged (`burst_latency.json`, `mirrored_latency.json`)
- **Security**: OWASP/MITRE labeling protocol + jailbreak similarity checks operational
- **Evaluation**: Statistical analysis complete (ROC/CM, McNemar, feature ablations)

### Phase 4: Documentation & Delivery (Nov 15 - Dec 20)
- **Paper Writing**: Markdown draft updated through Chapter 4; Word doc pending sync
- **Documentation**: README/plan updates ongoing; deployment notes captured in `reports/`
- **Presentation**: Outline started in `presentations/Advising_Session_Update.md`
- **Final Delivery**: To include monitoring traces, automated tests, and advisor-ready appendices

## Progress to Date (as of Nov 23, 2025)

1. **Weeks 1–2** – Literature review, threat model, and policy mapping completed; Week 1 snapshot plan approved.
2. **Weeks 3–4** – Data ingestion scripts run, Sept–Nov freeze sealed, 60 manual labels logged with OWASP/MITRE/NIST rationale, inter-rater κ = 1.0.
3. **Weeks 5–6** – Logistic/RF baselines trained + evaluated; ROC/CM, calibration, and ablation artifacts produced; FastAPI service online.
4. **Weeks 7–8** – Burst replay harness measures avg 7.4 ms (p95 11.7 ms); deployment notes, sandbox constraints, and calibration rationale documented.
5. **Weeks 9–10 (current)** – Manuscript integration, monitoring field definitions, and automated tests underway to satisfy Chapter 3 §3.6 + Testing Protocol commitments.

---

## Critical Milestones

1. **August 23**: Requirements and methodology finalized
2. **September 20**: Working alpha prototype demonstrated
3. **October 18**: Testing framework operational
4. **November 15**: Evaluation complete, paper draft ready
5. **December 20**: Project defense ready

## Risk Buffer

- Each phase includes 20% time buffer for unexpected challenges
- Parallel tracks allow flexibility in task scheduling
- Regular biweekly checkpoints ensure early issue detection

---

**Note**: Timeline aligned with D.Eng. program requirements and biweekly advisor meetings
