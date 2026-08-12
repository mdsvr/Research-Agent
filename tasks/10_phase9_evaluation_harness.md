# Task 10: Phase 9 — Evaluation Harness & Ablation Framework

## Phase Summary
- **Time Window:** H12 → H14 (2 Hours)
- **Goal:** Implement a zero-dependency evaluation harness to compute citation metrics, abstention accuracy, and ablation grids.
- **Deliverable:** `eval/metrics.py` and `eval/run_eval.py` writing reports to `eval/results/`.

## Why Hand-Rolled over External Frameworks (RAGAS/TruLens)?
Hand-rolling guarantees zero version-conflict risks, complete metric transparency, offline execution, and exact adherence to ground-rule requirements for explainability.

## Mathematical Formulations

### 1. ALCE Citation Metrics
- **Citation Recall:**
  $$\text{Recall} = \frac{\sum \mathbb{I}(\text{joint\_premise} \implies \text{claim})}{\text{Total Claims}}$$
- **Citation Precision:** A citation $c_i$ is precise IF full citation set entails claim AND $(P \setminus \{c_i\})$ fails to entail claim.
  $$\text{Precision} = \frac{\text{Number of Precise Citations}}{\text{Total Citations Issued}}$$

### 2. Abstention Confusion Matrix & Metrics
| | Should Answer | Should Abstain |
|---|---|---|
| **Agent Answered** | True Answer (TA) | **False Answer (FA)** *(Critical Risk)* |
| **Agent Abstained** | False Refusal (FR) | True Refusal (TR) |

- **False Answer Rate:** $\text{FAR} = \frac{FA}{FA + TR}$ *(Primary metric to minimize)*
- **Over-Refusal Rate:** $\text{ORR} = \frac{FR}{FR + TA}$

### 3. Risk-Coverage & AUROC Metrics
Compute Area Under Risk-Coverage Curve (AURC) and Area Under ROC Curve (AUROC) evaluating abstention signal discrimination against `is_answerable`.

## Ablation Experiment Grid
| Configuration | Dense | BM25/RRF | Reranker | NLI Verifier |
|---|---|---|---|---|
| 1. Baseline Naive RAG | ✓ | ✗ | ✗ | ✗ |
| 2. Hybrid RRF | ✓ | ✓ | ✗ | ✗ |
| 3. + Reranker | ✓ | ✓ | ✓ | ✗ |
| 4. Veritas Full Pipeline | ✓ | ✓ | ✓ | ✓ |

## Execution Command
```bash
python -m veritas eval --offline
# Outputs metrics table and writes eval/results/ablation_results.json
```

## Acceptance & Verification
- Evaluation completes offline using cached fixtures.
- Results written to committed `eval/results/` directory.

## Cut Strategy (If Behind Schedule)
- Omit ablation grid; compute core abstention and citation metrics only.
