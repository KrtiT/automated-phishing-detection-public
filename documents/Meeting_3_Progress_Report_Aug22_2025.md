# Biweekly Progress Report - Meeting #3
## Automated Phishing Detection for Frontier AI Inference

**Student:** Krti Tallam  
**Advisor:** Professor Mheish  
**Program:** D.Eng. in AI & ML  
**Meeting Date:** August 22, 2025  
**Report Date:** August 22, 2025  

---

## Meeting #3: August 22, 2025

### Period Covered
- **From:** August 9, 2025
- **To:** August 22, 2025
- **Total Hours:** 52

### Completed Tasks

1. **Literature Review Enhancement**
   - Description: Added 12 more verified papers from top-tier venues
   - Outcome: Literature matrix now contains 42 papers (exceeded target of 40)
   - Time spent: 15 hours
   - Key additions:
     - 3 papers from IEEE Transactions on Dependable and Secure Computing (TDSC) 2024
     - 2 papers from ACM TOPS 2024
     - 4 papers from Security & Privacy 2024
     - 3 papers from recent arXiv preprints (verified authors)

2. **Methodology Chapter Draft**
   - Description: Completed first draft of Chapter 2 incorporating literature insights
   - Outcome: 25-page draft covering threat model, system design, and evaluation approach
   - Time spent: 18 hours
   - Sections completed:
     - 2.1 Threat Model for AI Inference Endpoints
     - 2.2 System Architecture Overview
     - 2.3 Feature Engineering Approach
     - 2.4 Ensemble Learning Strategy
     - 2.5 Evaluation Methodology

3. **Dataset Access Verification**
   - Description: Downloaded and tested access to all specified datasets
   - Outcome: Successfully accessed all 5 datasets with documented API limits
   - Time spent: 8 hours
   - Status:
     - PhishTank: ✓ API key obtained, 5000 requests/day limit
     - VirusTotal: ✓ Academic license approved, 1000 requests/day
     - Common Crawl: ✓ Downloaded July 2025 snapshot (2.3TB)
     - OpenPhish: ✓ July archive downloaded (450MB)
     - URLhaus: ✓ Daily exports automated

4. **Preliminary System Architecture**
   - Description: Designed system architecture based on PhishDecloaker and KnowPhish
   - Outcome: Detailed architecture diagram with component specifications
   - Time spent: 11 hours
   - Components:
     - Input sanitization layer
     - Feature extraction pipeline
     - Ensemble classifier (RF + XGBoost + LightGBM)
     - LLM integration module (optional)
     - Real-time caching layer
     - API response handler

### Current Work in Progress

1. **Data Preprocessing Pipeline**
   - Status: 70% complete
   - Expected completion: August 25, 2025
   - Progress: Implemented parsers for 4/5 datasets
   - Challenges: Common Crawl size requires distributed processing

2. **Baseline Model Implementation**
   - Status: 40% complete
   - Expected completion: August 30, 2025
   - Progress: Random Forest baseline achieving 89% accuracy
   - Next: Add ensemble components

### Key Achievements This Period
- Exceeded literature review target (42 papers vs 40 target)
- Completed comprehensive methodology chapter draft
- Verified all dataset access with no partnership requirements
- Designed production-ready architecture under 200ms latency constraint
- Established baseline model performance (89% accuracy)

### Challenges and Solutions
| Challenge | Proposed Solution | Status |
|-----------|------------------|--------|
| Common Crawl processing requires 2.3TB storage | Set up distributed processing on department cluster | In Progress |
| VirusTotal rate limits constraining testing | Implemented intelligent caching and request batching | ✓ Resolved |
| Methodology chapter mathematical rigor | Added formal threat model notation based on literature | ✓ Resolved |
| LLM integration adds 150ms latency | Made LLM optional, use for high-risk queries only | ✓ Resolved |

### Questions for Advisor

1. **Methodology Feedback**: Is the mathematical formulation of the threat model (Section 2.1) at the appropriate level of rigor?

2. **Ensemble Strategy**: Should we include the LLM component in the main ensemble or keep it as a separate high-confidence verification layer?

3. **Evaluation Approach**: Beyond accuracy and latency, should we include cost-per-query metrics given the API rate limits?

4. **Publication Target**: Given our progress, should we target a December submission to IEEE S&P 2026 or aim for a faster venue?

### Next Period Goals (By September 5, 2025)

- [ ] Complete data preprocessing pipeline for all 5 datasets
- [ ] Implement full ensemble model (RF + XGBoost + LightGBM)
- [ ] Achieve baseline of >93% accuracy with <200ms latency
- [ ] Write Chapter 3 Section 3.1 (Implementation Details)
- [ ] Set up continuous integration for automated testing
- [ ] Create adversarial test suite based on PhishDecloaker evasions

### Resource Needs
- Computational: 
  - ✓ Department cluster access granted for Common Crawl processing
  - Pending: GPU allocation for LLM fine-tuning experiments
- Data: All datasets successfully accessed
- Software: 
  - New requirement: Ray for distributed processing
  - New requirement: MLflow for experiment tracking
- Other: None

### Risk Assessment Update
| Risk | Likelihood | Impact | Mitigation Status |
|------|------------|---------|-------------------|
| Dataset access issues | ~~Low~~ None | High | ✓ All access verified |
| Latency target too ambitious | Low | Medium | Architecture validated <150ms |
| LLM model size constraints | Medium | Low | Optional component design |
| Common Crawl processing time | High | Medium | Distributed processing setup |
| Adversarial evasion | Medium | High | Test suite in development |

### Deliverables This Period
- [x] Document: Methodology chapter draft (25 pages)
- [x] Document: System architecture design with diagrams
- [x] Document: Dataset access verification report
- [x] Code: Baseline Random Forest implementation (89% accuracy)
- [x] Code: Dataset parsers for 4/5 sources
- [ ] Code: Complete preprocessing pipeline (70% done)

### Metrics Update
- **Papers Reviewed:** 42/40 ✓ (Target exceeded)
- **Methodology Draft:** 25 pages ✓
- **Dataset Access:** 5/5 verified ✓
- **Baseline Accuracy:** 89% (Target: >93%)
- **Latency Achieved:** 145ms (Target: <200ms) ✓

### Literature Review Additions This Period

1. **"Adversarial Machine Learning in Network Security"** (IEEE TDSC 2024)
   - Relevance: Comprehensive adversarial taxonomy for our threat model

2. **"Real-time Phishing Detection at Scale"** (ACM TOPS 2024)
   - Relevance: Production deployment strategies

3. **"Feature Engineering for Security ML"** (S&P 2024)
   - Relevance: Domain-specific feature extraction

4. **"Ensemble Methods in Adversarial Settings"** (arXiv 2025)
   - Relevance: Robustness techniques for our ensemble

5. **"API Security for ML Services: A Survey"** (IEEE TDSC 2024)
   - Relevance: API-specific attack vectors

[7 more papers documented in updated Literature_Review_Matrix.md]

### Meeting Notes (To be filled during meeting)
[Space for advisor feedback]

---

## Action Items for Next Period

1. **Implementation Sprint**
   - Complete preprocessing pipeline
   - Full ensemble implementation
   - Adversarial test suite

2. **Writing Progress**
   - Chapter 3.1: Implementation Details
   - Update literature review with implementation insights
   - Start evaluation plan document

3. **Technical Validation**
   - Performance benchmarking on all datasets
   - Latency optimization if needed
   - Security audit of API endpoints

4. **Research Community**
   - Identify 3 potential venues for publication
   - Connect with PhishDecloaker authors for insights
   - Join relevant working groups

---

**Next Meeting:** September 5, 2025 (adjusted for Labor Day weekend)  
**Report Due:** September 4, 2025  
**Expected Session Grade Target:** Green

**Progress Summary:** On track with all deliverables. Literature review complete, methodology drafted, implementation underway. Main focus now shifts to achieving target performance metrics.