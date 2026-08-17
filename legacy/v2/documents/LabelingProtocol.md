# Labeling Protocol for AI Inference Phishing Dataset

**Purpose.** Provide consistent instructions for annotating telemetry records so
labels align with OWASP LLM Top 10, MITRE ATLAS techniques, and NIST AI RMF
governance language referenced in Chapter 3.

## Workflow

1. **Ingest batch** – export candidate rows from `data/processed/v2025.08.10/*`
   into an annotation sheet that includes URL, timestamp, token counts, latency,
   and any session metadata.
2. **Primary review** – Krti Tallam applies the decision tree below, records the
   label (`phishing` or `benign`), OWASP category, MITRE tactic, rationale, and
   timestamp in `data/labels/label_audit_log.csv`.
3. **Secondary QA** – Advisor proxy spot-checks ≥20% of entries, adding their
   name in the `reviewer` column when they confirm or overturn a label. Disputed
   samples are resolved synchronously and logged in a notes column.
4. **Inter-rater check** – Every 50 samples, compute Cohen’s kappa between the
   two reviewers and document it in the annotation tracker; values <0.75 trigger
   a calibration meeting.

## Decision Tree

1. **Resource Exhaustion / Denial-of-Wallet (OWASP LLM09)**
   - Prompt tokens or completion tokens exceed historical p97.5 AND
     repeated >3× per 10 minutes OR latency >300 ms in bursts.
   - Label as `phishing`, MITRE ATLAS `TA0042`.
2. **Prompt Injection / Instruction Hijack (OWASP LLM01/LLM04)**
   - URL or payload contains known jailbreak strings, jailbreak similarity score
     ≥0.75, or explicit instructions to reveal system prompts.
   - Label as `phishing`, MITRE `TA0031` or `TA0027`.
3. **Insecure Output Handling (OWASP LLM05/LLM06)**
   - Payload attempts to embed scripts/HTML/SQL that would exploit downstream
     rendering or storage.
   - Label as `phishing`, MITRE `TA0027`.
4. **Benign**
   - Request follows standard prompt templates, token totals within the
     5th–95th percentile of the workload family, no similarity to jailbreak
     corpus, latency within normal bounds, and proper authentication context.
   - Label `benign`, OWASP `BENIGN`, MITRE `N/A`.

## Logging Requirements

- Every annotation must include a short free-text rationale referencing the
  above rules.
- File names or record IDs must match the raw data source (e.g.,
  `openphish_20250715_07`).
- Audit log entries should be committed alongside weekly updates so Chapter 3
  can cite traceability.

## Tooling

- Use the `data/labels/label_audit_log.csv` file for authoritative logging.
- For bulk batches, maintain a temporary spreadsheet but ensure entries are
  copied into the CSV before closing the session.
- Future work: integrate a lightweight Streamlit UI to guide reviewers through
  the decision tree and auto-populate OWASP/MITRE fields.
