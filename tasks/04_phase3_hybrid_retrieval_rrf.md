# Task 04: Phase 3 — Hybrid Retrieval & Reciprocal Rank Fusion

## Phase Summary
- **Time Window:** H2.5 → H4 (1.5 Hours)
- **Goal:** Implement a dual-stage dense + sparse retrieval pipeline fused via Reciprocal Rank Fusion (RRF).
- **Deliverable:** `veritas/index.py` returning top-20 candidate chunks per query.

## Architecture & Rationale
- **Dense Retriever (`BAAI/bge-small-en-v1.5`):** Captures semantic context and paraphrased queries.
- **Sparse Retriever (`BM25Okapi`):** Captures exact tokens (CVE identifiers, version numbers, function names).
- **Reciprocal Rank Fusion (RRF):** Merges rankings without requiring score normalization across heterogeneous distributions.

## Critical BGE Query Prefix Requirement
`BAAI/bge-small-en-v1.5` requires an explicit instruction prefix for queries (passages require no prefix):
```python
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
```
*Note: Omitting this prefix degrades retrieval performance significantly.*

## Algorithm Implementation (`veritas/index.py`)

### 1. Vector Index & Matrix Cache
- Encode all corpus chunks at ingest time.
- Save dense embedding matrix to disk (`data/index/embeddings.npy`) to eliminate runtime encoding overhead.

### 2. Reciprocal Rank Fusion Algorithm
```python
from collections import defaultdict

def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Combines multiple ranked lists of chunk_ids into a single fused ranking."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))
```

## Retrieval Pipeline Execution Flow
1. **Dense Querying:** Top-20 retrieved via cosine similarity of query embedding against `embeddings.npy`.
2. **Sparse Querying:** Top-20 retrieved via `BM25Okapi` over whitespace-tokenized chunks.
3. **Fusion:** Top-20 candidates selected via RRF score ($k=60$).

## Acceptance & Verification
- For 5 benchmark queries, target answer chunks appear within top-20 fused results.
- Disk caching avoids re-embedding corpus on subsequent CLI calls.

## Cut Strategy (If Behind Schedule)
- Drop BM25 and RRF; use dense vector retrieval exclusively.
