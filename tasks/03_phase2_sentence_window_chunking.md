# Task 03: Phase 2 — Sentence-Window Chunking Strategy

## Phase Summary
- **Time Window:** H1.5 → H2.5 (1 Hour)
- **Goal:** Transform raw documents into precise, sentence-aligned chunks containing explicit positional offset pointers.
- **Deliverable:** `veritas/chunking.py` emitting `Chunk` objects under 350 tokens.

## Theoretical Rationale: The Chunk Size Tension
- **Large Chunks (512+ tokens):** Higher retrieval recall, but poor citation precision (user cannot easily verify a claim within 500 words).
- **Small Chunks (~256 tokens):** High citation precision and ideal NLI premise length, but risks splitting facts across boundaries.
- **Resolution:** 256-token target size with 1-sentence overlap provides precise citation verification while retaining context across chunk seams.

## Chunk Data Architecture (`veritas/schemas.py`)
```python
from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str        # Formatted as "S3::c12" (Doc ID + Index)
    doc_id: str          # Formatted as "S3"
    doc_title: str       # Title derived from manifest
    text: str            # Clean chunk text content
    char_start: int      # Absolute character start offset in original file
    char_end: int        # Absolute character end offset in original file
    sent_range: tuple    # Sentence index tuple (start_sent, end_sent)
```

## Chunking Algorithm (`veritas/chunking.py`)
1. **Sentence Segmentation:** Tokenize document into sentences using regex boundary detection (`[.!?]\s+`).
2. **Greedy Accumulation:** Iterate through sentences, appending to the current chunk until target (~256 tokens) is reached.
3. **Overlap Retention:** Copy the final sentence of chunk $N$ as the first sentence of chunk $N+1$.
4. **Pointer Generation:** Calculate exact character indices (`char_start`, `char_end`) relative to original source document text.

## Code Example
```python
def chunk_document(doc_id: str, doc_title: str, text: str, target_tokens: int = 256) -> list[Chunk]:
    sentences = split_sentences(text)
    chunks = []
    # Accumulate sentences, track character offsets, construct Chunk instances
    return chunks
```

## Acceptance & Verification
- No chunk exceeds 350 tokens.
- `char_start` and `char_end` accurately slice the target chunk text from source document.
- Sentence overlap works cleanly across chunk transitions.

## Cut Strategy (If Behind Schedule)
- Fall back to simple 300-character fixed splits with 50-character sliding overlap.
