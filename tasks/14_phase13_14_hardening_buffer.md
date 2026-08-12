# Task 14: Phase 13 & 14 — System Hardening & Buffer

## Phase Summary
- **Time Window:** H20 → H24 (4 Hours total)
  - **Phase 13 (Hardening):** H20 → H22 (2 Hours)
  - **Phase 14 (Buffer & Final Submission):** H22 → H24 (2 Hours)
- **Goal:** Hardened pipeline against failure modes, validated repository git history, and executed pre-submission checklist.
- **Deliverable:** Battle-tested, submission-ready codebase.

## Defensive Error Handling Matrix

| Edge Case Failure | Implementation Guard |
|---|---|
| Empty Retrieval Results | Trigger Gate A abstention with clear refusal message; prevent pipeline crash. |
| Malformed LLM JSON Output | Single Pydantic retry → Regex extraction → Extractive fallback. |
| Hallucinated Chunk ID | Filter output citations against retrieved `chunk_ids` set before verification. |
| Query Exceeds Context Length | Truncate query gracefully and record warning in `veritas/trace.py`. |
| Corrupted Corpus PDF File | Log parsing error, skip unreadable file, and continue ingestion process. |
| API Rate Limit / Timeout | Automatic failover down provider chain (Gemini → OpenRouter → Ollama → Offline). |
| Verifier Out-Of-Memory (OOM) | Catch CUDA/Memory error; downgrade to `nli-deberta-v3-small`. |
| Encoding Artifacts in PDF | Strip non-printable characters during text normalization phase. |

## Pre-Submission Verification Workflow

### 1. Clean Environment Verification
```bash
# Execute in temporary directory
cd /tmp && git clone <repo-url> veritas-verify
cd veritas-verify
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m veritas ingest
python -m veritas ask "Test query" --offline
python -m veritas eval --offline
```

### 2. Git History Audit
- Verify commit history reflects continuous progress across the 24-hour window (not a single monolithic commit).
- Confirm `.env`, API keys, `__pycache__`, and binary vector caches are excluded via `.gitignore`.

### 3. Final Submission Deliverables Check
- [x] Repo is public.
- [x] `requirements.txt` pinned.
- [x] `data/corpus/`, `eval/gold.jsonl`, and `eval/results/` committed.
- [x] `samples/` directory populated with trace outputs.
- [x] Submission completed before 24-hour deadline.
