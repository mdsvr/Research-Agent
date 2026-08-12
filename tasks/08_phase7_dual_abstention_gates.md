# Task 08: Phase 7 — Dual Abstention Gates

## Phase Summary
- **Time Window:** H9 → H10 (1 Hour)
- **Goal:** Implement programmatic abstention logic to reliably refuse unanswerable and adversarial queries.
- **Deliverable:** `veritas/abstain.py` enforcing multi-stage refusal criteria.

## Dual Abstention Architecture

```
User Query
    │
    ▼
┌─────────────────────────┐
│ Hybrid Retrieval        │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐      max(rerank_score) < τ_lo
│ Gate A: Pre-Generation  ├───────────────────────────────┐
└───────────┬─────────────┘                               │
            │ (Pass)                                      │
            ▼                                             │
┌─────────────────────────┐                               │
│ LLM Generation & NLI    │                               │
└───────────┬─────────────┘                               │
            │                                             │
            ▼                                             │
┌─────────────────────────┐      supported_claims < min   │
│ Gate B: Post-Verify     ├───────────────────────────────┤
└───────────┬─────────────┘                               │
            │ (Pass)                                      │
            ▼                                             ▼
     Answer Output                               Abstain Response
```

## Gate Definitions & Signals

### Gate A: Pre-Generation Retrieval Sufficiency
- **Condition:** `max(reranker_scores) < tau_lo`
- **Function:** Aborts LLM execution when retrieval fails to identify contextually relevant documents. Saves API costs and prevents hallucination from weak context.

### Gate B: Post-Verification Evidential Sufficiency
- **Condition:** `count(supported_claims) < min_supported_claims` OR `LLM.insufficient_evidence == True`
- **Function:** Refuses queries where documents appear topically relevant but lack the specific required facts (adversarial case).

## Signal Reliability Hierarchy
1. **Cross-Encoder Score:** Strongest pre-generation retrieval quality signal.
2. **NLI Entailment Count:** Strongest post-verification factual evidence signal.
3. **LLM Self-Reported Confidence:** Moderate reliability.
4. **Raw Cosine Similarity:** Weak, uncalibrated baseline (avoid relying on cosine alone).

## Informative Abstention Output Format
Instead of a generic refusal, Veritas prints the nearest retrieved contexts and explains why evidence fell short:
```
I cannot answer this question based on the provided corpus.

Closest matching source passages:
  - [S4::c07] "Analysis of CVE-2024-3094 memory structures..." (relevance: 0.31)
  - [S9::c02] "Implementation notes for system update..." (relevance: 0.28)

Reason: The passages discuss XZ Utils architecture but do not disclose CVSS severity scores.
```

## Acceptance & Verification
- Verify that unanswerable queries hit Gate A.
- Verify that adversarial queries (related topic, missing facts) hit Gate B.

## Cut Strategy (If Behind Schedule)
- Non-negotiable phase. Thresholds can be estimated if time is constrained.
