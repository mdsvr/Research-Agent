# Task 00: Project Overview & Architecture

## Core Vision & Purpose
Veritas is a verified research agent that processes user questions over a fixed document corpus. Unlike standard RAG systems that display unverified citation markers ("citation theatre"), Veritas uses an independent NLI fact-checking model to verify that every output claim is strictly entailed by its cited text spans. If evidence is insufficient, the system explicitly abstains.

## Key Differentiator: Fact Entailment vs. Citation Theatre
- **Standard RAG (Citation Theatre):**
  `Retrieve chunks` → `LLM generates response + inserts [1], [2]` → `Output without verification`
- **Veritas Architecture (Verified Entailment):**
  `Retrieve chunks` → `LLM emits structured claims + chunk IDs` → `Independent NLI verifier checks (claim, chunk) pair` → `Filter unsupported claims` → `Abstain if insufficient evidence`

## Scope Lock-in Decisions
| Feature | Choice | Technical Rationale |
|---|---|---|
| Live Web Search | No | Avoid network flakiness; rubric focuses on corpus citation fidelity. |
| UI | CLI Only | Fast execution, zero framework bloat, rubric awards CLI equal credit. |
| Fine-Tuning | No | High time cost, non-essential for NLI-guided verification. |
| Citation Granularity | Sentence-Level | Optimal trade-off between atomic claim fragmentation and bulk chunk vagueness. |

## Repository Layout
```
veritas-agent/
├── README.md                  # Comprehensive setup & tradeoff documentation
├── PLAN.md                    # Primary 24-hour architecture blueprint
├── requirements.txt           # Exactly pinned dependencies
├── config.yaml                # Centralized project configuration
├── .env.example               # Template for API keys
├── veritas/                   # Core Python package
│   ├── cli.py                 # CLI entry point (ingest, ask, eval)
│   ├── config.py              # YAML config loader dataclass
│   ├── chunking.py            # Sentence-window chunking engine
│   ├── index.py               # Dense vector + BM25 index with RRF fusion
│   ├── rerank.py              # Cross-encoder reranker
│   ├── generate.py            # Provider-fallback structured generator
│   ├── verify.py              # NLI entailment verification module
│   ├── abstain.py             # Dual pre- and post-retrieval abstention gates
│   ├── pipeline.py            # End-to-end pipeline orchestrator
│   ├── schemas.py             # Pydantic data schemas
│   └── trace.py               # Detailed stage-by-stage execution logger
├── data/corpus/               # Source document repository
├── eval/                      # Gold evaluation dataset & evaluation harness
├── fixtures/                  # Cached LLM response fixtures for offline execution
└── samples/                   # Sample trace outputs (answered and abstained)
```

## Trace System Design (`veritas/trace.py`)
`trace.py` captures detailed metadata at every pipeline stage:
1. Top-K candidates returned by BM25 and Dense index.
2. Fused rankings produced by Reciprocal Rank Fusion (RRF).
3. Promoted and demoted candidates from the Cross-Encoder Reranker.
4. Exact context string injected into the generation prompt.
5. Verification scores and decisions emitted by the NLI model.
