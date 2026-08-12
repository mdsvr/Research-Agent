from typing import List, Tuple, Optional
from veritas.config import Config
from veritas.schemas import Chunk, AgentAnswer, Verdict


def _closest_passages(scored_chunks: List[Tuple[Chunk, float]], n: int = 2) -> str:
    lines = [
        f"  - [{chunk.chunk_id}] \"{chunk.text[:120]}...\" (relevance: {score:.2f})"
        for chunk, score in scored_chunks[:n]
    ]
    return "\n".join(lines) if lines else "  (none)"


def check_gate_a(scored_chunks: List[Tuple[Chunk, float]], cfg: Config) -> Tuple[bool, Optional[str]]:
    """Gate A: pre-generation retrieval sufficiency. Abstains when the best reranker
    score falls below the calibrated threshold."""
    if not scored_chunks:
        return True, "No source document chunks were retrieved for the query."

    if not cfg.abstain.gate_a:
        return False, None

    max_score = max(s for _, s in scored_chunks)
    if max_score < cfg.abstain.min_rerank_score:
        return True, (
            f"Retrieved passages fell below the relevance threshold "
            f"(max score: {max_score:.2f} < {cfg.abstain.min_rerank_score:.2f}).\n"
            f"Closest matching passages found:\n{_closest_passages(scored_chunks)}\n"
            f"Reason: The corpus does not contain sufficiently relevant material to answer this question."
        )

    return False, None


def check_gate_b(
    answer: AgentAnswer,
    verdicts: List[Verdict],
    scored_chunks: List[Tuple[Chunk, float]],
    cfg: Config,
) -> Tuple[bool, Optional[str]]:
    """Gate B: post-verification evidential sufficiency. Abstains when the generator
    declared insufficient evidence, or when too few claims survived NLI verification."""
    if answer.insufficient_evidence:
        return True, (
            f"The generator declared insufficient evidence in context.\n"
            f"Closest passages:\n{_closest_passages(scored_chunks)}"
        )

    if not cfg.abstain.gate_b:
        return False, None

    supported_count = sum(1 for v in verdicts if v.supported)
    if supported_count < cfg.abstain.min_supported_claims:
        return True, (
            f"Post-verification check failed: {supported_count} of {len(verdicts)} generated "
            f"claims were verified by the NLI entailment model "
            f"(minimum required: {cfg.abstain.min_supported_claims}).\n"
            f"Closest matching passages:\n{_closest_passages(scored_chunks)}\n"
            f"Reason: Passages discuss the general topic but do not state the specific requested facts."
        )

    return False, None
