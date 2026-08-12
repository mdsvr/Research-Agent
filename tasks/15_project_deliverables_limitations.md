# Task 15: Master Deliverables, Limitations & Time Budget

## Project Deliverables Checklist
1. **Source Documents:** 8–15 domain-bounded documents committed to `data/corpus/`.
2. **Question Set:** 30–60 annotated queries committed to `eval/gold.jsonl`.
3. **Trace Log Examples:** Comprehensive trace logs committed to `samples/`.
4. **Retrieval/Tool Note:** Dedicated section in `README.md`.
5. **Tradeoff Notes:** Detailed design decision log in `README.md`.

## Rubric Score Factors (100 Points Total)
- **Working Agent (30 pts):** Clean-clone execution, offline availability, zero key failure.
- **Approach & Model Choice (25 pts):** Hybrid retrieval, NLI verifier, dual abstention gates.
- **Code Organization (20 pts):** Modular package structure, explicit dataclasses, `trace.py`.
- **README & Setup (15 pts):** Clear setup, offline quickstart, real sample runs.
- **Tradeoffs & Engineering Judgment (10 pts):** Measured metrics, honest limitations.

## 10 Honest System Limitations
1. **Self-Authored Benchmark:** Benchmark created internally; independent dataset would be more objective.
2. **Multi-Hop NLI Limitation:** NLI verifier can score synthesized multi-chunk claims as neutral.
3. **Model-Based Metrics:** Citation recall/precision scores agree ~80-85% with human judges.
4. **Small-Sample Thresholds:** Calibration on 40 queries carries variance.
5. **No Live Web Search:** Scoped strictly to local corpus.
6. **Single-Turn Interaction:** No conversational memory or multi-turn state.
7. **Language Scope:** Optimized for English technical text.
8. **Extractive Bias:** Stronger on direct factual spans than broad thematic summaries.
9. **Unmeasured User Tolerance:** Over-refusal rate optimized conservatively without UX feedback.
10. **Reranker Domain Shift:** Out-of-the-shelf cross-encoder may underperform on specialized domain text.

## 24-Hour Master Time Budget

| Phase | Hours | Focus Area | Cuttable? |
|---|---|---|---|
| Phase 0 | H0–H1 | Scaffold & Pin Dependencies | No |
| Phase 1 | H1–H1.5 | Corpus Assembly & Preparation | No |
| Phase 2 | H1.5–H2.5 | Sentence-Window Chunking | Simplify |
| Phase 3 | H2.5–H4 | Hybrid Retrieval & RRF | Drop BM25 if late |
| Phase 4 | H4–H5 | Cross-Encoder Reranking | **Yes (1st to cut)** |
| Phase 5 | H5–H7 | Structured Generation & Fallbacks | No |
| Phase 6 | H7–H9 | NLI Attribution Verifier | **NEVER** |
| Phase 7 | H9–H10 | Dual Abstention Gates | No |
| Phase 8 | H10–H12 | Gold Dataset Creation | Shrink to 25 |
| Phase 9 | H12–H14 | Evaluation Harness & Ablations | Cut ablations |
| Phase 10 | H14–H15 | Threshold Calibration | Heuristic default |
| Phase 11 | H15–H17 | Offline Reproducibility | Retain Ollama |
| Phase 12 | H17–H20 | README & Tradeoff Notes | **NEVER** |
| Phase 13 | H20–H22 | Defensive Hardening | Compress |
| Phase 14 | H22–H24 | Final Buffer & Submission | Buffer |
