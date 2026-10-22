# Project TRACE — SAR Analysis and Human-Review Workflow

## 1. Purpose and architecture

TRACE is a two-stage, AI-assisted Suspicious Activity Report (SAR) workflow for case analysis and narrative preparation. A **Risk Analyst Agent** examines structured customer/account/transaction evidence; a **human reviewer** decides whether to proceed; an approved case is then passed to a **Compliance Officer Agent** to prepare and validate a narrative.

```text
Selected customer + accounts + transactions
                 |
           DataLoader / CaseData
                 |
     Risk Analyst Agent (5 steps)
                 |
       Structured RiskAnalystOutput
                 |
        HUMAN REVIEW / DECISION
           /               \
       REJECT             PROCEED
         |                   |
   Record decision      Compliance Officer Agent
                             |
                      ReACT tool/validation loop
                             |
                       SAR document creation
                             |
                     Save local SAR document
```

**Key components**

| Component | Responsibility |
| --- | --- |
| `DataLoader` / `CaseData` | Assemble and validate the case and its relationships. |
| `RiskAnalystAgent` | Perform five-stage evidence-based analysis and return a `RiskAnalystOutput`. |
| Transaction chunk helper | Review subsets of transaction-heavy cases before downstream synthesis. |
| `ComplianceOfficerAgent` | Draft a concise, objective narrative, call validation utilities, and return structured output. |
| Jupyter workflow | Sequence cases, display findings, capture human decision, create and save documents. |
| `ExplainabilityLogger` / audit events | Record execution and decision history; persistent workflow audit logging must be wired into the notebook to be reliable. |

## 2. Execution order

For each selected customer:

1. Build `CaseData` from customer, account, and transaction records; enforce structural checks (including transactions belonging to the supplied accounts).
2. Call `risk_agent.analyze_case(case_data)`.
3. Display the Risk Analyst's **classification, confidence, risk level, reasoning, and supporting indicators** to the reviewer.
4. Capture a **reviewer ID, explicit yes/no decision, and rationale**. Persist the decision, timestamp, case ID, and the **exact findings shown** *before* continuing.
5. On rejection: record the rejection and stop the case's filing-preparation path.
6. On approval: call `compliance_agent.generate_compliance_narrative(case_data, risk_analysis)`.
7. Validate the generated narrative and structured response; if checks fail, log the failure and do not silently treat the document as complete.
8. Create the SAR document, save it under `../outputs/filed_sars/` (the current local convention), and persist the save outcome.
9. Record any failures and final workflow status. Process the next customer independently.

**Important:** Saving a file under `filed_sars` is **not proof of transmission to or acceptance by FinCEN**. The status should be `DOCUMENT_SAVED` until an actual filing interface and its acknowledgment are integrated.

## 3. Risk-analysis path and five stages

The planned/implemented risk path uses the following ordered steps:

| Step | Output focus |
| --- | --- |
| 1. Data Review | Customer, accounts, relevant transactions, facts, missing/conflicting data. |
| 2. Pattern Recognition | Suspicious patterns, transaction links, deviation from customer profile, red flags. |
| 3. Regulatory Mapping | Candidate SAR typologies and relevant regulatory indicators supported by facts. |
| 4. Risk Quantification | Severity, frequency, magnitude, gaps, evidence strength, escalation consideration. |
| 5. Classification Decision | Final typed `RiskAnalystOutput`, not a decision to file. |

**Analysis-path selection:** The risk agent can use a direct/compact input path for cases small enough to process reliably, or a **transaction-chunking** path for larger cases. The exact branch predicate, chunk size, and model/token thresholds are **not verified in this README** and should be documented next to the actual `analyze_case()` condition. Do not assume the choice is based on a fixed transaction count unless confirmed in code.

Step 5 must adhere to `RiskAnalystOutput`:

- `classification`: one of `Structuring`, `Sanctions`, `Fraud`, `Money_Laundering`, `Other`.
- `confidence_score`: number from 0.0 to 1.0.
- `reasoning`: at most 500 characters.
- `key_indicators`: list of supporting indicators.
- `risk_level`: one of `Low`, `Medium`, `High`, `Critical`.

The agent recommends a classification; **human judgment controls whether the workflow proceeds**.

## 4. Transaction chunking

Chunking addresses oversized or unreliable one-shot Step 1 prompts. The helper `_analyze_transaction_chunk(...)` prepares a compact payload consisting of selected customer context and a **subset of transactions**. Each LLM call is asked to return structured:

- `material_transactions`
- `notable_facts`
- `data_gaps`

Chunk results are combined for the broader case review, which then feeds the later analytical steps. Preserve transaction IDs and deduplicate overlapping facts; ensure every chunk is processed, including a final partial chunk. Record chunk boundaries, counts, failures, model settings, and token usage for reproducibility. Summarization is **not** a substitute for preserving the source transaction dataset.

**Observed during development:** Case sizes of approximately **90 and 121 transactions** produced long prompts (roughly **18k–24k characters** of compact case data). LLM responses sometimes ended with `finish_reason="length"`, including in chunked runs. Chunking reduces individual request size but **does not guarantee complete or valid JSON**; response validation and retry/failure handling remain necessary.

## 5. Compliance Officer ReACT loop

The Compliance Officer follows a **Reason → Action → Observation** control pattern:

1. Consider the risk findings, case facts, narrative requirements and gaps.
2. Produce a machine-readable **tool request** (an `ACTION`), rather than prose pretending to run a tool.
3. The **application executes** the allowed tool and sends the actual result back as an `OBSERVATION`.
4. Revise the narrative, validate again if needed, and emit a terminal structured result (`FINAL`) when checks pass.

Example tools include `get_regulatory_requirements`, `validate_word_count`, and the narrative-compliance validator. The prompt **does not execute Python tools on its own**: dispatch/parsing/loop termination must be implemented in application code. Bound iterations, reject unknown tools, validate action parameters, and fail explicitly on malformed responses.

The final output is validated against the project's `ComplianceOfficerOutput` model. Field definitions changed in development; the authoritative schema is the **current source model**, not historical prompt examples.

**Observed during development:** Some responses contained free-form `THOUGHT`/`ACTION` text, fenced JSON, or an apparent final narrative without the exact required `ACTION` or `FINAL` envelope. Smoke tests consequently reported parser/loop errors. A generated narrative is **not** a passing ReACT integration test unless actual tool execution and a valid terminal response are observed.

## 6. Validation rules

Validation occurs at multiple layers:

**Input data:** Pydantic types and cross-record relationships (`CaseData`, customers, accounts, transactions), including nonempty transactions and correct account/customer ownership.

**Risk Analyst:** Strict classification/risk-level enumerations; confidence within `[0, 1]`; reasoning length ≤500 characters; exact shape of the final typed response. Intermediate JSON is parsed and checked rather than assumed valid.

**Narrative:** The project helper currently describes a **120-word maximum** and checks for customer identification, suspicious activity description, transaction amounts/dates, and an explanation of suspicion. It also checks recognized regulatory terminology and maintains regulatory citation references. These are **prototype heuristics**, not proof of legal compliance. Validate facts against case evidence; do not fabricate thresholds, intent, counterparties, transactions or regulatory requirements. A positive boolean from a text-matching validator is not sufficient to authorize filing.

**Workflow:** Require an explicit, valid human decision and rationale; log attempts, rejection, document generation, validation outcomes, save outcomes, and exceptions. Never silently map an invalid decision to `REJECT`.

## 7. Human decision gate and auditability

The notebook asks whether to proceed after risk analysis. The durable audit record should include:

- UTC timestamp and unique event ID; case ID and reviewer identity.
- `PROCEED` or `REJECT` and the reviewer's explanation.
- The **exact findings presented**, plus case/evidence version or hash where practical.
- Compliance validation observations, document ID and save state, actual submission/acknowledgment (if any), and failures.

Write each event **immediately** to persistent storage; keep the in-memory `audit_decisions` list only as a convenient notebook summary. For a prototype, an append-only JSONL event log is useful. Production requires authenticated reviewer identity, controlled access, tamper-evidence, retention policies and recovery guarantees.

A human `PROCEED` decision authorizes the **next preparation stage** in this prototype; it should not be equated with a completed regulatory filing or an independently verified filing approval.

## 8. Outputs and observed results

Expected local output conventions include:

| Location | Purpose |
| --- | --- |
| `../outputs/case_data/` | Serialized case inputs used in development/testing. |
| `../outputs/audit_logs/` | Agent explainability logs and, if integrated, workflow decision events (`.jsonl`). |
| `../outputs/filed_sars/` | Locally generated SAR documents; **not necessarily filed**. |

The notebook returns `processed_cases`, `approved_sars`, `rejected_cases`, and `audit_decisions`. These represent the **current run's in-memory results** unless separately persisted.
--- 

