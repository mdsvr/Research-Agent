# Task 13: Phase 12 — README & Tradeoff Notes Documentation

## Phase Summary
- **Time Window:** H17 → H20 (3 Hours)
- **Goal:** Author a comprehensive `README.md` containing quickstart steps, architecture diagrams, ablation tables, and measured tradeoff notes.
- **Deliverable:** Final `README.md` (worth 25% of total evaluation score).

## Required Document Structure

1. **Vision Statement:** Single-sentence core concept (§0).
2. **Quickstart Guide:**
   - Primary: Offline execution (zero API keys).
   - Secondary: API-key enabled execution.
3. **Architecture Diagram:** ASCII pipeline flowchart illustrating dual gates and NLI verifier.
4. **Sample Runs:** Real trace excerpts for (a) Answered query, (b) Unanswerable query, (c) Adversarial query.
5. **Ablation Results Table:** Empirical metrics grid comparing Naive RAG vs. Veritas.
6. **Retrieval & Tool Approach Note:** Dedicated heading explaining chunking, hybrid search, RRF, and cross-encoder reranking.
7. **Measured Tradeoff Notes:** Structured decision log (template below).
8. **Limitations:** Honest statement of known constraints (§4).

## Tradeoff Documentation Template

Every tradeoff entry must state: **Decision** → **Alternative Rejected** → **Empirical Reason**.

- **Embedding Model:** `bge-small-en-v1.5` over `all-MiniLM-L6-v2` due to higher MTEB retrieval performance at acceptable CPU latency.
- **Retrieval Fusion:** RRF over score normalization; avoids tuning arbitrary weights across non-comparable score distributions.
- **Chunk Size:** 256 tokens over 512 tokens; prioritizes human-verifiable citation precision over raw chunk recall.
- **Vector Storage:** NumPy/Disk cache over ChromaDB/Qdrant; 800 chunks do not justify vector database overhead.
- **Citation Granularity:** Sentence-level over atomic claim decomposition; avoids over-fragmentation and coreference errors.
- **Structured Decoding:** Pydantic validation + retry over strict grammar masking; avoids LLM reasoning tax.
- **Verification Engine:** a dedicated NLI cross-encoder over LLM-as-judge; faster, cheaper, offline-capable, and independent. (Planned as MiniCheck-Flan-T5-Large; shipped as `cross-encoder/nli-deberta-v3-small` — see task 07 for why.)
- **Abstention Metric:** Cross-encoder reranker score + NLI support count over raw cosine similarity.
- **Evaluation Harness:** Hand-rolled metrics over RAGAS; eliminates dependency fragility and ensures 100% code explainability.

## Acceptance & Verification
- Quickstart commands copy-paste cleanly and execute without error.
- All 11 tradeoff items contain measured numbers.

## Cut Strategy (If Behind Schedule)
- **CRITICAL PHASE — DO NOT CUT.** Allocate full time to complete documentation.
