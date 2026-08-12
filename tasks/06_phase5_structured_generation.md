# Task 06: Phase 5 — Structured Generation & Fallback Chain

## Phase Summary
- **Time Window:** H5 → H7 (2 Hours)
- **Goal:** Prompt LLM to output structured claims with explicit source citations, protected by a provider fallback chain.
- **Deliverable:** `veritas/generate.py` emitting validated `AgentAnswer` objects.

## Prompt Engineering Design
The generation prompt enforces two crucial rules:
1. **Citation Minimization:** Demands minimal supporting citations to optimize citation precision.
2. **Explicit Escape Hatch:** Provides a clear path to set `insufficient_evidence: true` when context is inadequate, reducing hallucination.

```
You are a research assistant. Answer the QUESTION using ONLY the SOURCES below.

Rules:
- Every claim must be supported by at least one source.
- Cite using exact source IDs, e.g. ["S3::c12"].
- Cite the MINIMUM number of sources needed.
- If sources are insufficient, set insufficient_evidence to true and return empty claims.

SOURCES:
[S3::c12] <chunk_text>
[S7::c04] <chunk_text>

QUESTION: <question>
Return JSON matching schema: {"insufficient_evidence": bool, "claims": [{"text": str, "citations": [str]}]}
```

## Schema Definitions (`veritas/schemas.py`)
```python
from pydantic import BaseModel

class Claim(BaseModel):
    text: str
    citations: list[str]

class AgentAnswer(BaseModel):
    insufficient_evidence: bool
    claims: list[Claim]
    reasoning: str | None = None
```

## Constraint Tax Avoidance Strategy
Rather than using hard grammar-constrained decoding (which restricts model reasoning capability and degrades performance), Veritas employs **JSON mode + Pydantic validation + single retry + regex fallback**.

## Provider Fallback Cascade
1. **Gemini 2.0 Flash** (Primary API)
2. **OpenRouter API** (Secondary API)
3. **Ollama Local** (`http://localhost:11434`, key-free local inference)
4. **Offline Deterministic Fallback** (Extracts top reranked chunk verbatim)

*Note: All calls set `temperature: 0.0` for full reproducibility.*

## Acceptance & Verification
- Output parses into valid `AgentAnswer` objects.
- Hallucinated chunk IDs not present in prompt sources are automatically discarded.

## Cut Strategy (If Behind Schedule)
- Restrict to single LLM provider + regex citation extraction. (Retain Ollama path).
