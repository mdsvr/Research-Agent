from typing import Optional

from veritas.config import Config
from veritas.schemas import TraceLog
from veritas.index import search_hybrid, load_chunks
from veritas.rerank import rerank_chunks
from veritas.generate import generate_answer, check_quote_grounding
from veritas.verify import verify_claims
from veritas.abstain import check_gate_a, check_gate_b
from veritas.trace import save_trace


def run_pipeline(
    query: str,
    cfg: Config,
    qid: Optional[str] = None,
    offline: bool = False,
) -> TraceLog:
    """Hybrid search -> RRF -> rerank -> Gate A -> generate -> NLI verify -> Gate B -> trace."""
    trace = TraceLog(query=query, qid=qid)

    chunks = load_chunks(cfg)
    chunk_map = {c.chunk_id: c for c in chunks}

    # 1. Hybrid search & RRF
    fused_chunks, rrf_tuples, dense_ranking, bm25_ranking = search_hybrid(query, cfg, chunks=chunks)
    trace.dense_candidates = dense_ranking
    trace.bm25_candidates = bm25_ranking
    trace.rrf_fused = rrf_tuples
    if not fused_chunks:
        trace.abstained = True
        trace.abstain_reason = "Corpus index is empty or no candidates returned."
        save_trace(trace, cfg)
        return trace

    # 2. Cross-encoder reranking
    scored_chunks = rerank_chunks(query, fused_chunks, cfg)
    trace.reranked = [(c.chunk_id, score) for c, score in scored_chunks]
    trace.context_injected = [c.chunk_id for c, _ in scored_chunks]

    # 3. Gate A: pre-generation retrieval sufficiency
    should_abstain_a, reason_a = check_gate_a(scored_chunks, cfg)
    if should_abstain_a:
        trace.abstained = True
        trace.abstain_reason = reason_a
        save_trace(trace, cfg)
        return trace

    # 4. Structured generation
    answer = generate_answer(query, scored_chunks, cfg, offline=offline)

    # 4b. Quote grounding: a claim whose supporting span is not literally in its cited
    # chunk is discarded before any model gets a say. Cheap, deterministic, and it fires
    # on exactly the fabrications an entailment score is weakest at catching.
    answer, ungrounded = check_quote_grounding(answer, chunk_map)
    trace.ungrounded_quotes = ungrounded
    trace.final_answer = answer

    # 5. NLI verification
    verdicts = verify_claims(answer.claims, chunk_map, cfg)
    trace.verdicts = verdicts

    # 6. Gate B: post-verification evidential sufficiency
    should_abstain_b, reason_b = check_gate_b(answer, verdicts, scored_chunks, cfg)
    if should_abstain_b:
        trace.abstained = True
        trace.abstain_reason = reason_b
        save_trace(trace, cfg)
        return trace

    # Unverified claims are dropped from the answer but recorded, so a shrunken answer
    # is visible in the trace rather than silently disappearing.
    trace.dropped_claims = [c.text for c, v in zip(answer.claims, verdicts) if not v.supported]
    answer.claims = [c for c, v in zip(answer.claims, verdicts) if v.supported]
    trace.final_answer = answer

    save_trace(trace, cfg)
    return trace
