# Task 01: Phase 0 — Scaffold & Dependency Pinning

## Phase Summary
- **Time Window:** H0 → H1 (1 Hour)
- **Goal:** Establish a fully runnable repository structure with pinned dependencies and a single centralized configuration file.
- **Deliverable:** Working CLI shell (`python -m veritas --help`) and committed environment setup.

## Step-by-Step Implementation

### 1. Git Initialization & Directory Setup
Initialize Git, construct package directory structure, and push initial commit immediately:
```bash
git init
mkdir -p veritas data/corpus eval/results fixtures samples
touch veritas/__init__.py
```

### 2. Dependency Pinning (`requirements.txt`)
Install exact dependency versions to prevent environment drift during evaluation:
```
sentence-transformers==2.7.0
rank-bm25==0.2.2
transformers==4.40.1
torch==2.3.0
pydantic==2.7.1
pyyaml==6.0.1
click==8.1.7
pypdf==4.2.0
numpy==1.26.4
scikit-learn==1.4.2
```

### 3. Centralized Configuration (`config.yaml`)
All runtime parameters and model parameters must reside in a single file to allow deterministic tuning:
```yaml
corpus_dir: data/corpus
chunk:
  target_tokens: 256
  overlap_sentences: 1
embedding:
  model: BAAI/bge-small-en-v1.5
retrieval:
  dense_k: 20
  bm25_k: 20
  rrf_k: 60
  final_k: 6
rerank:
  enabled: true
  model: BAAI/bge-reranker-v2-m3
verify:
  model: vectara/hallucination_evaluation_model
  support_threshold: 0.5
abstain:
  min_rerank_score: 0.35
  min_supported_claims: 1
llm:
  providers: [gemini, openrouter, ollama, offline]
  temperature: 0.0
```

### 4. Configuration Dataclass Loader (`veritas/config.py`)
Load `config.yaml` into strongly-typed Pydantic dataclasses to ensure static validation across all pipeline modules.

## Acceptance & Verification
- `python -m veritas --help` runs without import errors or missing module warnings.
- `requirements.txt` contains pinned versions (no `~=` or `>=`).
- Config options are accessible via `veritas.config.load_config()`.

## Cut Strategy (If Behind Schedule)
- Non-negotiable phase. Must complete cleanly before proceeding.
