# Task 12: Phase 11 — Offline Reproducibility

## Phase Summary
- **Time Window:** H15 → H17 (2 Hours)
- **Goal:** Ensure reviewers can run ingestion, querying, and evaluation completely offline without API keys.
- **Deliverable:** Working `--offline` flag and committed response cache (`fixtures/cache.json`).

## Three-Tier Offline Execution Strategy

```
                       ┌─────────────────────────┐
                       │ Request Generation Call │
                       └───────────┬─────────────┘
                                   │
                     Has API Key?  ├─── Yes ───► Call Gemini / OpenRouter API
                                   │
                                   Wait No
                                   ▼
                   --offline set?  ├─── Yes ───► Read key from fixtures/cache.json
                                   │
                                   Wait No
                                   ▼
                 Ollama Available? ├─── Yes ───► Query http://localhost:11434
                                   │
                                   Wait No
                                   ▼
                     Deterministic Fallback ───► Extract top reranked chunk verbatim
```

## Tier Implementation Details

### 1. Cached Fixture Tier (`fixtures/cache.json`)
Key LLM responses by hash of model name and prompt:
$$\text{key} = \text{SHA256}(\text{model\_name} + \text{prompt\_string})$$
When running `--offline`, the generator reads directly from `fixtures/cache.json`.

### 2. Local Ollama Tier
Provide zero-key local inference compatibility via Ollama's OpenAI-compatible endpoint:
- Service URL: `http://localhost:11434/v1`
- Model: `llama3.1:8b` or `qwen2.5:7b`

### 3. Deterministic Extractive Fallback
If no API key exists, no cache hit occurs, and Ollama is inactive:
- Extract top reranked context chunk verbatim.
- Format as single self-citing claim.

## Clean-Clone Verification Procedure
Run the following test sequence in an isolated environment to verify reviewer setup:
```bash
cd /tmp && git clone <repo-url> && cd veritas-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m veritas ingest
python -m veritas ask "What functions were backdoored in XZ Utils?" --offline
python -m veritas eval --offline
```

## Acceptance & Verification
- Clean-clone sequence executes without network access or API keys.

## Cut Strategy (If Behind Schedule)
- Non-negotiable phase. Essential for submission scoring.
