# IRB and Ethics Review Document
## Automated Phishing Detection for Frontier AI Inference

**Principal Investigator:** Krti Tallam  
**Institution:** George Washington University  
**Department:** School of Engineering and Applied Science  
**Project Duration:** 6 months  
**Date:** July 29, 2025  

---

## 1. Project Overview

This research project develops an automated system to detect and prevent phishing attacks targeting frontier AI inference systems. The project involves collecting and analyzing phishing attack data, developing detection algorithms, and testing the system's effectiveness.

## 2. Human Subjects Determination

### 2.1 Does this research involve human subjects?
**No** - This research primarily involves:
- Analysis of technical attack patterns
- Development of algorithmic detection methods
- Testing on synthetic and publicly available datasets
- No direct interaction with human participants

### 2.2 Data Sources
- **PhishTank Database**: Publicly available, anonymized phishing URLs
- **AI Attack Repositories**: Technical attack signatures without personal information
- **Synthetic Attack Generation**: Computer-generated attack patterns
- **System Logs**: Anonymized API request patterns from consenting organizations

## 3. Ethical Considerations

### 3.1 Data Privacy and Protection

#### Phishing Data Handling
- **No Personal Information**: All phishing samples will be sanitized to remove any personal data
- **URL Anonymization**: Domain information will be hashed when necessary
- **Attack Payload Sanitization**: Remove any embedded personal information from attack samples

#### AI System Data
- **Request Anonymization**: All API requests will be stripped of identifying information
- **IP Address Handling**: IP addresses will be hashed or removed
- **Timestamp Generalization**: Precise timestamps will be generalized to protect usage patterns

### 3.2 Responsible Security Research

#### Vulnerability Disclosure Policy
1. **Discovery Protocol**: Any vulnerabilities discovered during research will be documented
2. **Responsible Disclosure**: 
   - Notify affected vendors before publication
   - Provide 90-day disclosure window
   - Offer remediation assistance
3. **No Active Attacks**: Research will not involve launching actual attacks
4. **Defensive Focus**: All research aims to improve security, not exploit vulnerabilities

### 3.3 Dual-Use Considerations

#### Preventing Misuse
- **Access Controls**: Detection algorithms will include safeguards against reverse engineering
- **Documentation**: Clear statements about ethical use only
- **Licensing**: Released under license prohibiting malicious use
- **Monitoring**: Track usage and citations to identify potential misuse

## 4. Data Management Ethics

### 4.1 Collection Ethics
- **Consent**: Use only publicly available or explicitly consented data
- **Minimization**: Collect only data necessary for research objectives
- **Purpose Limitation**: Use data only for stated research purposes

### 4.2 Storage and Retention
- **Secure Storage**: Encrypted storage for all attack samples
- **Access Logging**: Audit trail for all data access
- **Retention Period**: Data deleted 1 year after project completion
- **Backup Ethics**: Secure backups with same protections as primary data

### 4.3 Sharing and Publication
- **Anonymization**: All shared data will be fully anonymized
- **Aggregation**: Individual attack patterns aggregated before sharing
- **Review Process**: Ethics review before any data release
- **Citation Requirements**: Require ethical use statement in citations

## 5. Algorithm Ethics

### 5.1 Bias Considerations
- **Attack Diversity**: Ensure training data includes diverse attack types
- **Geographic Representation**: Include attacks from various global regions
- **Temporal Balance**: Include both historical and recent attack patterns
- **Regular Audits**: Quarterly bias assessments of detection algorithms

### 5.2 Transparency
- **Explainable AI**: Detection decisions will be interpretable
- **Algorithm Documentation**: Full documentation of detection logic
- **Limitation Disclosure**: Clear statement of system limitations
- **Performance Metrics**: Transparent reporting of accuracy and errors

### 5.3 Fairness
- **Equal Protection**: System should protect all AI systems equally
- **No Discrimination**: Avoid bias against specific platforms or vendors
- **Resource Consideration**: Ensure system works for resource-constrained deployments

## 6. Testing Ethics

### 6.1 Synthetic Attack Generation
- **Controlled Environment**: All testing in isolated environments
- **No Real Targets**: Never test against production systems without permission
- **Attack Simulation**: Use only simulated attacks, not real malicious code
- **Safe Detonation**: Sandboxed environments for any payload analysis

### 6.2 Performance Testing
- **Consent Required**: Explicit consent for any real-world testing
- **Minimal Impact**: Ensure testing doesn't degrade system performance
- **Data Protection**: Test data handled with same care as research data
- **Result Anonymization**: All performance results anonymized before publication

## 7. Publication Ethics

### 7.1 Responsible Disclosure in Publications
- **Technical Details**: Balance detail with preventing misuse
- **Attack Descriptions**: Generalize attack patterns to prevent replication
- **Code Release**: Security review before releasing any code
- **Dataset Release**: Only release sanitized, anonymized datasets

### 7.2 Academic Integrity
- **Proper Attribution**: Cite all data sources and prior work
- **Contribution Clarity**: Clear statement of novel contributions
- **Reproducibility**: Provide sufficient detail for ethical reproduction
- **Peer Review**: Submit to venues with security ethics review

## 8. Stakeholder Considerations

### 8.1 AI System Operators
- **Benefit**: Improved security for their systems
- **Risk**: Potential performance impact
- **Mitigation**: Extensive performance testing and optimization

### 8.2 Security Researchers
- **Benefit**: New tools and techniques for defense
- **Risk**: Potential misuse of techniques
- **Mitigation**: Clear ethical guidelines and licensing

### 8.3 General Public
- **Benefit**: More secure AI systems serving the public
- **Risk**: Minimal direct risk
- **Communication**: Clear public benefit statements

## 9. Compliance

### 9.1 Regulatory Compliance
- **GDPR**: Ensure compliance with data protection regulations
- **CCPA**: Follow California privacy requirements
- **Sector-Specific**: Comply with any sector-specific regulations

### 9.2 Institutional Compliance
- **GWU Policies**: Adhere to all university research policies
- **Department Guidelines**: Follow SEAS research guidelines
- **Federal Requirements**: Comply with any federal research requirements

## 10. Ethics Review Checklist

- [x] No human subjects involved
- [x] Data anonymization procedures in place
- [x] Responsible disclosure policy defined
- [x] Dual-use considerations addressed
- [x] Bias mitigation strategies planned
- [x] Testing ethics protocols established
- [x] Publication ethics guidelines set
- [x] Compliance requirements identified
- [x] Stakeholder impacts assessed
- [x] Regular ethics review scheduled

## 11. Approval and Signatures

**Student Researcher:**  
Name: Krti Tallam  
Date: _______________  
Signature: _______________  

**Faculty Advisor:**  
Name: _______________  
Date: _______________  
Signature: _______________  

**IRB Determination:**  
[ ] Not Human Subjects Research - IRB Review Not Required  
[ ] Exempt - Category: _______________  
[ ] Expedited Review Required  
[ ] Full Board Review Required  

**Ethics Committee Notes:**  
_____________________________________  
_____________________________________  
_____________________________________  

---

**Document Version:** 1.0  
**Last Updated:** July 29, 2025  
**Next Review Date:** [Monthly during project]