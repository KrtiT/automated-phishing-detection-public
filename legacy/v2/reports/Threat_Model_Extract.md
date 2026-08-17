# Threat Model for AI Phishing Detection
## Extract for August 9, 2025 Meeting

**Student:** Krti Tallam  
**Advisor:** Professor Mheish  
**Date:** August 9, 2025

---

## Attack Taxonomy

We categorize AI-targeted phishing attacks into four primary types:

### 1. Model Extraction Attacks
- **Description**: Queries designed to reverse-engineer model parameters
- **Risk Level**: High
- **Example**: Sequential queries that probe decision boundaries
- **Detection Strategy**: Pattern analysis of query sequences

### 2. Inference Manipulation
- **Description**: Inputs crafted to produce specific malicious outputs
- **Risk Level**: Critical
- **Example**: Adversarial prompts causing harmful content generation
- **Detection Strategy**: Semantic analysis of query intent

### 3. Resource Exhaustion
- **Description**: Computationally expensive queries for denial-of-service
- **Risk Level**: Medium
- **Example**: Extremely long sequences or complex computations
- **Detection Strategy**: Resource usage monitoring

### 4. Data Leakage Probes
- **Description**: Attempts to extract training data information
- **Risk Level**: High
- **Example**: Membership inference attacks
- **Detection Strategy**: Statistical anomaly detection

## Attacker Capabilities

We assume attackers with the following capabilities:

- **Access Level**: Black-box access to AI inference APIs only
- **Knowledge**: General AI architectures but not specific implementations
- **Resources**: Ability to generate large numbers of queries (up to 10K/hour)
- **Adaptability**: Can analyze responses and modify attack strategies
- **Tools**: Access to open-source attack frameworks

## Threat Scenarios

### Scenario 1: Startup Model Theft
- Attacker targets small AI company's proprietary model
- Uses systematic queries to extract model behavior
- Economic damage: $1-5M in lost IP

### Scenario 2: Service Disruption
- Coordinated resource exhaustion attack
- Targets critical AI services during peak hours
- Operational impact: Service downtime, reputation damage

### Scenario 3: Data Privacy Breach
- Extraction of personally identifiable information from training data
- Regulatory implications under GDPR/CCPA
- Legal and financial consequences

## Risk Prioritization Matrix

| Attack Type | Likelihood | Impact | Priority |
|-------------|------------|---------|----------|
| Model Extraction | High | High | Critical |
| Inference Manipulation | Medium | Critical | Critical |
| Resource Exhaustion | High | Medium | High |
| Data Leakage | Medium | High | High |

## Next Steps
- Develop detection algorithms for each attack type
- Create synthetic attack dataset for testing
- Design real-time monitoring system
- Implement proof-of-concept for TensorFlow Serving