# Questions for Professor M - Meeting #3
## Automated Phishing Detection for Frontier AI Inference

**Student:** Krti Tallam  
**Date Prepared:** August 14, 2025  
**Meeting Date:** August 22, 2025  

---

## 1. Methodology and Technical Approach

### 1.1 Threat Model Rigor
**Question:** Is the mathematical formulation in Section 2.2 of the Methodology outline appropriately rigorous for a D.Eng. thesis?

**Context:** I've defined the threat model as:
- Attack surface: $E$ (endpoint) processing requests $R = \{r_1, r_2, ..., r_n\}$
- Each request containing: query payload $q_i$, tokens $t_i$, metadata $m_i$

**Follow-up:** Should I expand this to include formal security proofs or is the current level sufficient?

### 1.2 Feature Engineering Scope
**Question:** The baseline implementation uses 12 features. Should I expand to 20-30 features for the final implementation, or focus on feature quality over quantity?

**Current features:**
- URL-based: 5 features
- Payload-based: 4 features  
- AI-specific: 3 features

**Trade-off:** More features vs. latency constraints

---

## 2. Implementation Strategy

### 2.1 Ensemble Architecture
**Question:** Given our <200ms latency constraint, should the LLM component be:
a) Part of the main ensemble (slower but more integrated)
b) Separate verification layer for high-risk queries only
c) Omitted entirely in favor of lighter models

**Context:** PhishSense-1B achieves 97.5% accuracy but adds ~150ms latency

### 2.2 Development Priorities
**Question:** What should be my next implementation priority?
1. Complete ensemble (XGBoost + LightGBM)
2. Adversarial robustness testing
3. Real dataset integration
4. API endpoint development

**Current status:** Random Forest baseline at 89% accuracy, 0.15ms latency

---

## 3. Dataset and Evaluation

### 3.1 Dataset Access
**Question:** For VirusTotal academic access, should I:
- Apply now (may take 2-3 weeks)
- Use only freely available datasets
- Find alternative sources

**Impact:** VirusTotal provides the most comprehensive malware/phishing data

### 3.2 Evaluation Metrics
**Question:** Beyond accuracy and latency, should I include:
- Cost-per-query metrics (given API rate limits)
- Energy consumption metrics
- Model interpretability scores
- Adversarial robustness scores

**Concern:** Too many metrics might dilute the focus

### 3.3 AI-Specific Dataset Creation
**Question:** Since no existing dataset has AI endpoint phishing examples, is my augmentation strategy sufficient?
- Synthetic generation based on patterns
- Modification of traditional phishing
- Manual creation of ~100 examples

**Alternative:** Should I conduct a red team exercise to generate real examples?

---

## 4. Research Scope and Timeline

### 4.1 Scope Boundaries
**Question:** Should the thesis include:
- Only detection (current focus)
- Detection + automated response
- Detection + prevention strategies
- Full security framework

**Timeline impact:** Each addition adds ~3-4 weeks

### 4.2 Novel Contribution Clarity
**Question:** Is our novel contribution sufficiently clear?
- "First framework specifically for AI inference endpoint phishing"
- Should I emphasize the real-time aspect more?
- Should I focus more on the AI-specific features?

---

## 5. Publication Strategy

### 5.1 Target Venue
**Question:** Given our progress, which venue should we target?
- **IEEE S&P 2026** (December 2025 deadline, top-tier, competitive)
- **USENIX Security 2026** (February 2026 deadline, top-tier)
- **AISec Workshop** (October 2025 deadline, specialized, faster)
- **IEEE TDSC Journal** (rolling submissions, longer review)

**Trade-off:** Prestige vs. timeline vs. fit

### 5.2 Paper Focus
**Question:** Should the paper emphasize:
- Technical innovation (ensemble + adversarial robustness)
- Systems contribution (production-ready implementation)
- Empirical study (comprehensive evaluation)
- Position paper (AI security threats)

---

## 6. Technical Clarifications

### 6.1 Baseline Performance
**Question:** Is 89% accuracy on synthetic data acceptable for a baseline, or should I achieve 93%+ before moving to ensemble?

### 6.2 Latency Measurement
**Question:** For latency reporting, should I measure:
- Pure model inference time (current: 0.15ms)
- End-to-end including feature extraction
- Full API round-trip time

### 6.3 Production Considerations
**Question:** How much emphasis on production deployment?
- Docker containerization
- Kubernetes scaling configs
- CI/CD pipeline
- Monitoring/alerting

---

## 7. Academic Requirements

### 7.1 Literature Review Completeness
**Question:** With 42 papers, is the literature review sufficient, or should I aim for 50-60?

### 7.2 Mathematical Notation
**Question:** Should I maintain consistent mathematical notation with a specific paper/textbook?

### 7.3 Code Documentation
**Question:** What level of code documentation is expected?
- Current: Inline comments + docstrings
- Add: Formal API documentation
- Add: Architecture decision records

---

## 8. Project Management

### 8.1 Meeting Frequency
**Question:** Given the Yellow grade, should we increase meeting frequency to weekly check-ins?

### 8.2 Progress Indicators
**Question:** What specific milestones would move us from Yellow to Green?
- Complete implementation?
- 95% accuracy achieved?
- Real dataset integration?

### 8.3 Risk Mitigation
**Question:** Biggest risks you see in the project? How can I proactively address them?

---

## 9. Specific Technical Issues

### 9.1 CAPTCHA-Cloaking
**Question:** PhishDecloaker addresses CAPTCHA-cloaking. Should this be a major focus for our work too, or keep it as future work?

### 9.2 Adversarial Examples
**Question:** What's the minimum adversarial robustness testing needed for credibility?
- Basic FGSM attacks
- PGD attacks
- Adaptive attacks
- Full red team exercise

### 9.3 Feature Attribution
**Question:** Should I include SHAP/LIME analysis for model interpretability, or is that out of scope?

---

## 10. Administrative Questions

### 10.1 Conference Travel
**Question:** If the paper is accepted, is there funding for conference presentation?

### 10.2 Compute Resources
**Question:** For LLM experiments, can I access department GPU clusters, or should I budget for cloud compute?

### 10.3 External Collaboration
**Question:** The PhishDecloaker authors offered to share insights. Should I reach out, or keep the work independent?

---

## Questions Priority Order (for limited time):
1. **Ensemble architecture decision** (#2.1)
2. **Publication venue** (#5.1)
3. **Next implementation priority** (#2.2)
4. **Scope boundaries** (#4.1)
5. **Progress indicators for Green grade** (#8.2)

---

**Note**: I've prepared detailed questions but will prioritize based on available meeting time. The most critical decisions needed are around implementation strategy and publication planning.