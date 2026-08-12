# Task 09: Phase 8 — Gold Dataset Creation

## Phase Summary
- **Time Window:** H10 → H12 (2 Hours)
- **Goal:** Construct a benchmark evaluation set (`eval/gold.jsonl`) containing 30–60 manually annotated test questions across three distinct challenge categories.
- **Deliverable:** Committed `eval/gold.jsonl` dataset file.

## Dataset Bucket Composition

### Bucket 1: Answerable Queries (~50% of dataset, 15–30 queries)
- **Definition:** Facts explicitly stated within the corpus documents.
- **Annotation:** Includes ground-truth chunk IDs (`gold_chunk_ids`) and reference answer text (`gold_answer`).

### Bucket 2: Completely Unanswerable Queries (~25% of dataset, 8–15 queries)
- **Definition:** Topic is entirely absent from the corpus.
- **Correct System Behavior:** Gate A abstention (`insufficient_evidence`).

### Bucket 3: Adversarial / Related-but-Insufficient Queries (~25% of dataset, 8–15 queries)
- **Definition:** Documents cover the general subject matter, but lack the specific fact requested.
- **Examples:**
  - *Missing Attribute:* Corpus describes vulnerability mechanics but omits CVSS score.
  - *Entity Swap:* Corpus covers Protocol A; query asks for Protocol B parameters.
  - *Temporal Trap:* Corpus covers 2023 guidelines; query asks for 2024 updates.
  - *False Premise:* Query assumes a fact contradicted by corpus text.
- **Correct System Behavior:** Gate B abstention.

## Benchmark JSONL Schema
```jsonl
{"qid": "q001", "question": "What function in XZ Utils contained the backdoor?", "is_answerable": true, "gold_answer": "The _get_cpuid function was intercepted...", "gold_chunk_ids": ["S1::c04"], "bucket": "answerable"}
{"qid": "q020", "question": "What is the official CVSS v3.1 score for CVE-2024-3094?", "is_answerable": false, "gold_answer": null, "gold_chunk_ids": [], "bucket": "adversarial", "note": "Corpus details mechanism but excludes CVSS numeric score"}
{"qid": "q035", "question": "What is the annual revenue of Microsoft in 2023?", "is_answerable": false, "gold_answer": null, "gold_chunk_ids": [], "bucket": "unanswerable", "note": "Corpus contains no financial data"}
```

## Dataset Construction Methodology
1. Read corpus documents systematically.
2. Formulate Bucket 1 questions directly from explicit facts.
3. Formulate Bucket 3 questions by identifying missing parameters within those same documents.

## Acceptance & Verification
- `eval/gold.jsonl` contains at least 30 valid JSON entries.
- All three buckets are represented with correct schema validation.

## Cut Strategy (If Behind Schedule)
- Reduce evaluation dataset size to 25 total queries (12 answerable, 6 unanswerable, 7 adversarial).
