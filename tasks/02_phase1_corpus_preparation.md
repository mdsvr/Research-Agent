# Task 02: Phase 1 — Corpus Selection & Preparation

## Phase Summary
- **Time Window:** H1 → H1.5 (0.5 Hours)
- **Goal:** Assemble and standardize a factual, domain-bounded text corpus of 8–15 source documents stored in `data/corpus/`.
- **Deliverable:** Committed document corpus, document loader module, and document metadata manifest (`data/manifest.json`).

## Domain Selection Rationale
Target Domain: **Public Cybersecurity Advisories / NIST Guidelines / OWASP Documentation / CVE Technical Analyses**.
- **Coherent Domain:** Enables focused, realistic technical questions.
- **Factual Density:** Packed with dates, version identifiers, software names, and vulnerability IDs.
- **Memorization Defense:** Technical parameters and obscure vulnerability metrics reduce the risk of the LLM answering purely from internal memory.

## Corpus Processing Pipeline

### 1. Document File Loader (`veritas/ingest.py`)
Implement robust file ingestion for `.txt`, `.md`, and `.pdf` files using `pypdf`:
- Strip repeating header/footer artifacts across PDF pages.
- Normalize whitespace (`\s+` to single space).
- Assign fixed document identifiers: `S1`, `S2`, `S3`, ...

### 2. Manifest Schema (`data/manifest.json`)
```json
[
  {
    "doc_id": "S1",
    "filename": "cve_2024_3094_analysis.md",
    "title": "XZ Utils Backdoor Detailed Analysis",
    "source_url": "https://nvd.nist.gov/vuln/detail/CVE-2024-3094",
    "char_count": 14250
  }
]
```

### 3. The LLM Memorization Test
Verify that evaluation questions cannot be answered by the LLM without context:
1. Select 3 specific factual queries (e.g., specific memory offsets or fixed patch version numbers).
2. Prompt raw LLM without context documents.
3. If the LLM answers correctly, replace the question with a more obscure parameter query.

## Command Line Interface Integration
```bash
python -m veritas ingest
# Output: Loaded 12 documents, 847 chunks written to data/index/
```

## Acceptance & Verification
- `data/corpus/` contains 8–15 valid files.
- Manifest accurately reflects document counts and file parameters.
- Raw LLM without context fails or provides incorrect details for test queries.

## Cut Strategy (If Behind Schedule)
- Fallback to 8 plain `.txt` files; omit PDF parsing logic.
