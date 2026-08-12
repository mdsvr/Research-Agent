# Task 05: Phase 4 — Cross-Encoder Reranking

## Phase Summary
- **Time Window:** H4 → H5 (1 Hour)
- **Goal:** Re-rank top-20 candidate chunks into top-6 context windows using a cross-encoder model while generating a primary abstention signal.
- **Deliverable:** `veritas/rerank.py` emitting scored and ordered candidates.

## Retrieve-then-Rerank Architecture
Bi-encoder embedding models score queries and documents independently. Cross-encoders process `(query, document)` pairs simultaneously through full cross-attention layers, yielding significantly more accurate relevance scores.

## Dual Purpose: Ranking & Pre-Generation Abstention
1. **Precision Ranking:** Promotes truly relevant chunks to top positions for prompt injection.
2. **Abstention Signal:** The maximum reranker score across candidates serves as Gate A's primary relevance metric:
   - Cosine similarity is poorly calibrated (scores cluster closely).
   - Cross-encoder scores reflect genuine answer probability, making them suitable for thresholding unanswerable questions.

## Implementation Details (`veritas/rerank.py`)

### 1. Model Selection & Configuration
- **Primary Model:** `BAAI/bge-reranker-v2-m3`
- **Low-Resource Fallback:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Config Toggle:** Controlled via `rerank.enabled` in `config.yaml`.

### 2. Reranking Interface
```python
from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name: str):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 6) -> list[tuple[Chunk, float]]:
        pairs = [(query, c.text) for c in chunks]
        scores = self.model.predict(pairs)
        scored_chunks = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_k]
```

## Acceptance & Verification
- Verify that top-ranked chunks shift logically for test queries after reranking.
- Confirm log outputs record initial vs. final rank transformations (`veritas/trace.py`).

## Cut Strategy (If Behind Schedule)
- Disable reranking (`rerank.enabled: false`); pass RRF top-6 directly to prompt and use max cosine similarity as fallback Gate A score.
