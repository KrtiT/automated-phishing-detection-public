# Rolling Updates - Professor Mheish
## Automated Phishing Detection for Frontier AI Inference

**Student:** Krti Tallam  
**Program:** D.Eng. in AI & ML (Aug 2024 - Summer 2026)  
**Advisor:** Professor Mheish  

---

##  Meeting #1: July 26, 2025

### Summary
- **Status:**  Completed
- **Key Discussion:** Project kickoff and scope definition
- **Decision:** Focus on TensorFlow Serving for initial prototype
- **Action Items:** Complete literature review of 25+ papers by next meeting

### Progress Since Last Meeting
- Initial project setup
- Repository structure created
- Comprehensive documentation framework established
- 16 core documents prepared

### Next Steps (by August 9)
1. Complete literature review (target: 25+ papers)
2. Refine threat model based on recent AI security incidents
3. Set up development environment
4. Begin preliminary experiments with TensorFlow Serving

---

##  Meeting #2: August 9, 2025

### Summary
- **Status:**  Completed
- **Key Discussion:** Presented annotated bibliography (30+ papers), OWASP/MITRE threat model draft, and initial labeling policy alignment with NIST AI RMF.
- **Decisions:** Proceed with Sept–Nov telemetry snapshot; advisor approved OWASP/NIST labeling matrix and requested explicit timeline for data freeze.

### Deliverables Reviewed
- Annotated bibliography (`research/` folder) ✔
- Threat model extract (`reports/Threat_Model_Extract.md`) ✔
- Updated timeline (`reports/Praxis_Timeline.md`) ✔
- Environment setup checklist ✔

### Action Items (completed by Aug 23)
1. Produce Week 1 data snapshot plan with file inventory + access controls.
2. Draft labeling protocol referencing OWASP LLM Top 10, MITRE ATLAS, NIST AI RMF.
3. Stand up preprocessing scripts aligned to the proposed features.

---

##  Meeting #3: August 23, 2025

### Summary
- **Status:**  Completed
- **Focus:** Methodology + data governance sign-off; advisor pushed for explicit time ranges and labeling workflow transparency.
- **Outcome:** Approved Week 1 Data Freeze Log, LabelingProtocol.md, and Testing Protocol. Green-lit move into data ingestion + baseline training.

### Deliverables
- Requirements/methodology draft ✔
- Week1_Data_Freeze_Log.md ✔
- LabelingProtocol.md ✔
- Testing_Protocol.md (outlines future pytest suite) ✔

### Decisions & Follow-ups
1. Lock Sept 1–Nov 23 2025 as the snapshot window.
2. Minimum 30 phishing + 30 benign manual labels with OWASP/MITRE/NIST rationale before Week 4.
3. Provide DeploymentValidationPlan.md with concrete FastAPI/burst steps before Week 5.

---

##  Meeting #4: September 6, 2025

### Summary
- **Status:**  Completed
- **Focus:** Design approval, implementation start, and governance instrumentation.
- **Outcome:** Advisor signed off on preprocessing scripts, baseline configuration, and FastAPI deployment plan; requested rolling annotation log and inter-rater sampling.

### Highlights
- Data pipeline scripts running against sanitized extracts.
- Annotation template + audit log structure reviewed.
- FastAPI prototype scope agreed (logistic baseline first, RF as challenger).

### Action Items (due Sep 20)
1. Ingest Sept–Nov data freeze, save processed CSVs, and document stats.
2. Complete 60 manual annotations with OWASP/MITRE/NIST rationale + κ calculation.
3. Train logistic/RF baselines, export baseline_metrics.json, and prep ROC/CM pipeline.

---

##  Meeting #5: September 20, 2025

### Summary
- **Status:**  Completed
- **Focus:** Alpha prototype (data + labels) and initial modeling results.
- **Outcome:** Delivered 60-label audit log, interRaterStats.json (κ = 1.0), frozen datasets (v2025.08.10), and initial logistic/RF training runs. Advisor asked to push calibration + deployment validation into October.

### Reviewed Deliverables
- `data/processed/v2025.08.10/*.csv` ✔
- `data/labels/label_audit_log.csv`, `interRaterPairs.csv`, `interRaterStats.json` ✔
- Preliminary `baseline_metrics.json` ✔
- ROC/CM figure prototypes ✔

### Next Actions (Weeks 5–6)
1. Complete calibration, ROC/CM exports, feature ablations, and McNemar tests.
2. Implement FastAPI service + burst replay harness and log latency metrics.
3. Draft Chapter 4 results narrative + deployment validation notes.

---

##  Meeting #6: October 4, 2025

### Summary
- **Status:**  Scheduled
- **Focus:** Mid-project review

### Target Deliverables
- [ ] Beta prototype
- [ ] Testing framework
- [ ] Preliminary results
- [ ] Publication timeline

---

##  Meeting #7: October 18, 2025

### Summary
- **Status:**  Scheduled
- **Focus:** Testing and evaluation progress

### Target Deliverables
- [ ] Complete test suite
- [ ] Evaluation results
- [ ] Performance benchmarks
- [ ] Paper outline

---

##  Meeting #8: November 1, 2025

### Summary
- **Status:**  Scheduled
- **Focus:** Results analysis and paper writing

### Target Deliverables
- [ ] Full evaluation results
- [ ] Statistical analysis
- [ ] Paper draft (50%)
- [ ] Conference selection

---

##  Meeting #9: November 15, 2025

### Summary
- **Status:**  Scheduled
- **Focus:** Paper draft review

### Target Deliverables
- [ ] Complete paper draft
- [ ] Final system refinements
- [ ] Presentation outline
- [ ] Submission plan

---

##  Meeting #10: December 6, 2025

### Summary
- **Status:**  Completed
- **Focus:** Label provenance, dataset scale, feature selection clarity
- **Outcome:** Session grade **Yellow**; advisor is satisfied with pace but wants clearer documentation around labeling and feature selection plus a plan to enlarge the dataset.

### Highlights
- Relying on original phishing/benign labels from source datasets is acceptable; explicitly document that no relabeling was performed.
- Time ranges, filenames, and identifier structures are already noted; keep them in the methodology for reproducibility.
- Current dataset is ~1.6k rows; gather more samples to strengthen training/generalization.

### Decisions & Follow-ups
1. Add explicit label-provenance notes and sampling window details to the methodology/data sheets.
2. Define any feature-selection algorithm/criteria used and include correlation/heat map visuals with a short explanation of how they informed choices.
3. Attempt to grow the dataset before the next training/eval cycle and refresh dataset statistics.
4. Prepare concise bullets for the next meeting covering: response to prior feedback, progress, issues/setbacks, and tasks for the next two weeks.
5. Spring meetings will happen on Fridays preceding the Saturday dates sent via invite (next up: Dec 19).

### Action Items (before Dec 19)
1. Update methodology/docs with label provenance and sampling window details.
2. Produce correlation/heat map visual and note feature-selection rationale.
3. Acquire additional samples and refresh dataset_stats.json.
4. Draft the 4-bullet update (response to feedback, progress, issues, next tasks).

---

##  Meeting #11: December 20, 2025

### Summary
- **Status:**  Scheduled
- **Focus:** Project wrap-up and next steps

### Target Deliverables
- [ ] Project defense ready
- [ ] All deliverables submitted
- [ ] Future work plan
- [ ] Publication submission

---

##  Overall Project Metrics

### Progress Indicators
- **Literature Review:** 15/25 papers (60%)
- **Documentation:** 16/16 complete (100%)
- **Implementation:** 0/10 modules (0%)
- **Testing:** 0/50 test cases (0%)
- **Meetings Attended:** 1/11 (9%)

### Risk Status
- **Technical Risks:**  Medium (mitigation plans in place)
- **Timeline Risk:**  Low (on schedule)
- **Resource Risk:**  Medium (GPU access pending)

### Key Achievements
1.  Comprehensive project documentation
2.  Clear scope and objectives
3.  Advisor alignment on approach
4.  Literature review in progress

### Current Blockers
- None currently

---

##  Notes for Next Update
For each session, prepare concise bullets on:
1. Response to the advisor's previous feedback
2. Progress since the last meeting
3. Issues encountered and setbacks
4. Tasks for the next two weeks

---

**Document Purpose:** This rolling update document provides Professor Mheish with a quick overview of project progress, upcoming milestones, and meeting preparations. Updated biweekly after each advisor meeting.
