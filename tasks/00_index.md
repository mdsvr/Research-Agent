# Veritas Agent Task Index

This directory contains detailed task breakdowns for each phase and component of the Veritas Agent project. Each file provides a focused explanation of goals, theoretical rationale, step-by-step implementation, acceptance criteria, and contingency cut strategies, with all files strictly under 150 lines.

## Task Navigation

| Task File | Component / Phase | Time Window | Key Focus |
|---|---|---|---|
| [00_overview_and_architecture.md](file:///d:/Research%20agent/tasks/00_overview_and_architecture.md) | Vision & Architecture | Overview | NLI verification vs citation theatre, scope, repo layout, `trace.py` |
| [01_phase0_scaffold_and_pin.md](file:///d:/Research%20agent/tasks/01_phase0_scaffold_and_pin.md) | Phase 0 — Setup | H0 → H1 | Scaffolding repo, `requirements.txt`, `config.yaml` dataclass |
| [02_phase1_corpus_preparation.md](file:///d:/Research%20agent/tasks/02_phase1_corpus_preparation.md) | Phase 1 — Corpus | H1 → H1.5 | Document ingestion, PDF cleaning, manifest, memorization test |
| [03_phase2_sentence_window_chunking.md](file:///d:/Research%20agent/tasks/03_phase2_sentence_window_chunking.md) | Phase 2 — Chunking | H1.5 → H2.5 | 256-token sentence-window chunking, pointers (`char_start`/`char_end`) |
| [04_phase3_hybrid_retrieval_rrf.md](file:///d:/Research%20agent/tasks/04_phase3_hybrid_retrieval_rrf.md) | Phase 3 — Hybrid Search | H2.5 → H4 | Dense (BGE-small) + Sparse (BM25) fused via RRF ($k=60$) |
| [05_phase4_cross_encoder_reranking.md](file:///d:/Research%20agent/tasks/05_phase4_cross_encoder_reranking.md) | Phase 4 — Reranking | H4 → H5 | Cross-encoder reranking, retrieve-then-rerank, Gate A pre-signal |
| [06_phase5_structured_generation.md](file:///d:/Research%20agent/tasks/06_phase5_structured_generation.md) | Phase 5 — Generation | H5 → H7 | Citation-minimizing prompt, Pydantic schemas, provider fallbacks |
| [07_phase6_nli_attribution_verifier.md](file:///d:/Research%20agent/tasks/07_phase6_nli_attribution_verifier.md) | Phase 6 — NLI Verifier | H7 → H9 | Independent NLI entailment check (MiniCheck), ALCE precision |
| [08_phase7_dual_abstention_gates.md](file:///d:/Research%20agent/tasks/08_phase7_dual_abstention_gates.md) | Phase 7 — Dual Gates | H9 → H10 | Gate A (retrieval score) & Gate B (NLI evidence count) refusal |
| [09_phase8_gold_dataset_creation.md](file:///d:/Research%20agent/tasks/09_phase8_gold_dataset_creation.md) | Phase 8 — Gold Dataset | H10 → H12 | `eval/gold.jsonl` creation (Answerable, Unanswerable, Adversarial) |
| [10_phase9_evaluation_harness.md](file:///d:/Research%20agent/tasks/10_phase9_evaluation_harness.md) | Phase 9 — Eval Harness | H12 → H14 | ALCE Citation P/R/F1, Faithfulness, False Answer Rate, Ablation grid |
| [11_phase10_threshold_calibration.md](file:///d:/Research%20agent/tasks/11_phase10_threshold_calibration.md) | Phase 10 — Calibration | H14 → H15 | Risk-coverage parameter sweep, setting $\tau_{lo}$ for FAR target |
| [12_phase11_offline_reproducibility.md](file:///d:/Research%20agent/tasks/12_phase11_offline_reproducibility.md) | Phase 11 — Offline | H15 → H17 | Zero-API key offline execution, Ollama integration, response cache |
| [13_phase12_readme_tradeoff_notes.md](file:///d:/Research%20agent/tasks/13_phase12_readme_tradeoff_notes.md) | Phase 12 — README | H17 → H20 | README structure, quickstart, measured tradeoff notes template |
| [14_phase13_14_hardening_buffer.md](file:///d:/Research%20agent/tasks/14_phase13_14_hardening_buffer.md) | Phase 13 & 14 — Hardening | H20 → H24 | Failure mode matrix, clean-clone verification, pre-submission check |
| [15_project_deliverables_limitations.md](file:///d:/Research%20agent/tasks/15_project_deliverables_limitations.md) | Deliverables & Budget | Summary | Rubric score factors, 10 known limitations, 24-hr schedule summary |

## Summary of Execution Rules
- Every task file is self-contained and strictly under 150 lines.
- Phase dependencies follow chronological order from Task 01 to Task 14.
- High-priority non-negotiable phases: Task 01 (Scaffold), Task 07 (NLI Verifier), Task 08 (Abstention Gates), Task 12 (Offline Path), and Task 13 (README & Tradeoffs).
