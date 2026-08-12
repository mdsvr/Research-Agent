# Task 07: Phase 6 — NLI Attribution Verifier

## Phase Summary
- **Time Window:** H7 → H9 (2 Hours)
- **Goal:** Build an independent, model-based NLI verifier that checks whether cited context strictly entails generated claims.
- **Deliverable:** `veritas/verify.py` returning verified/unverified verdicts and precision metrics.

## Entailment Formulation
For every claim emitted by the LLM:
- **Premise ($P$):** Concatenation of text from all cited chunks (`\n\n`.join(cited_chunks)).
- **Hypothesis ($H$):** Generated claim text.
- **NLI Task:** Evaluate $P \implies H$ entailment probability independently of the generation prompt.

## Model Selection & Tiering
- **Primary Verifier:** `lytang/MiniCheck-Flan-T5-Large` (770M parameters, state-of-the-art claim verification).
- **Fast/Low-Memory Fallback:** `cross-encoder/nli-deberta-v3-small` (Lightweight 3-way NLI model).

> **As built (differs from this plan):** MiniCheck is a Flan-T5 seq2seq model and is not
> loadable through `sentence_transformers.CrossEncoder`, which is what the verifier uses.
> The shipped verifier runs `cross-encoder/nli-deberta-v3-small` as its **primary** model,
> configured in `config.yaml` under `verify.model`. Every `Verdict` records the backend
> that scored it. Adding MiniCheck means a seq2seq scoring path in `veritas/verify.py`,
> not a config change. See the README for the current description.

## Implementation Details (`veritas/verify.py`)

### 1. Joint Premise Checking
```python
def verify_claim(claim: Claim, chunk_map: dict[str, Chunk], threshold: float = 0.5) -> Verdict:
    premise = "\n\n".join(chunk_map[cid].text for cid in claim.citations if cid in chunk_map)
    if not premise:
        return Verdict(supported=False, score=0.0)
    score = nli_entailment_score(premise, claim.text)
    return Verdict(supported=(score >= threshold), score=score)
```

### 2. Citation Necessity Verification (ALCE Precision Metric)
To verify that citation $C_i$ is necessary:
1. Evaluate entailment score with full citation set ($P$).
2. Remove $C_i$ to form reduced premise ($P \setminus \{C_i\}$).
3. If $P \setminus \{C_i\} \implies H$ still holds, citation $C_i$ is redundant (lowers citation precision).

### 3. Unsupported Claim Policy
- **Default Policy:** Flag unverified claims with `⚠ unverified` tag and display NLI score.
- **Config Option:** Optionally drop unverified claims prior to output rendering.

## Acceptance & Verification
- Unit test: Construct a plausible but unsupported claim; verify that NLI model assigns score < 0.5 and flags it.

## Cut Strategy (If Behind Schedule)
- **CRITICAL PHASE — DO NOT CUT.** Reduce reranking and preprocessing complexity elsewhere to protect this module.
