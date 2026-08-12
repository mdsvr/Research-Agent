# Task 11: Phase 10 — Threshold Calibration

## Phase Summary
- **Time Window:** H14 → H15 (1 Hour)
- **Goal:** Calibrate Gate A reranker threshold ($\tau_{lo}$) and Gate B threshold ($\text{min\_supported\_claims}$) against empirical risk-coverage curves.
- **Deliverable:** Calibrated `config.yaml` matching reported evaluation numbers.

## Calibration Methodology

### 1. Empirical Sweep Data Collection
Run full gold evaluation dataset across the un-gated pipeline, recording:
- Maximum cross-encoder reranker score ($S_{rerank}$).
- Number of NLI-supported claims ($N_{supported}$).
- True binary outcome ($Y \in \{0, 1\}$).

### 2. Risk-Coverage Parameter Sweep
Iterate $\tau_{lo} \in [0.0, 1.0]$ in increments of 0.02:
```python
def evaluate_threshold(scores: list[float], outcomes: list[bool], tau: float):
    answered = [s >= tau for s in scores]
    coverage = sum(answered) / len(scores)
    false_answers = sum(1 for a, o in zip(answered, outcomes) if a and not o)
    risk = false_answers / sum(answered) if sum(answered) > 0 else 0.0
    return coverage, risk
```

### 3. Threshold Selection Principle
- **Target Constraint:** Set maximum acceptable False Answer Rate (e.g., $\text{FAR} \le 10\%$).
- **Optimization:** Select the threshold $\tau_{lo}$ that maximizes coverage while strictly satisfying the target FAR constraint.
- *Example:* "Targeted $\le 10\%$ false-answer rate; highest coverage satisfying constraint achieved at $\tau_{lo} = 0.42$."

## Conservative Bias Policy
Given small sample sizes (30–60 queries), calibration estimates carry variance:
- Prefer over-refusal over false answers for research/clinical/legal tools.
- Set thresholds conservatively (higher $\tau_{lo}$) when variance is present.

## Configuration Persistence
Update `config.yaml` with calibrated values:
```yaml
abstain:
  min_rerank_score: 0.42
  min_supported_claims: 1
```

## Acceptance & Verification
- Re-run `python -m veritas eval` and verify that metrics match documented targets.

## Cut Strategy (If Behind Schedule)
- Set conservative heuristic default ($\tau_{lo} = 0.40$); document rationale in README.
