import math
from typing import List, Tuple
from veritas.config import Config
from veritas.schemas import Chunk

_RERANKER_MODEL = None

def get_reranker_model(model_name: str):
    """Loads the configured cross-encoder. A load failure is fatal rather than silently
    substituted: a different reranker has a different score scale, which would quietly
    invalidate the calibrated Gate A threshold."""
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        from sentence_transformers import CrossEncoder
        _RERANKER_MODEL = CrossEncoder(model_name)
    return _RERANKER_MODEL

def rerank_chunks(query: str, chunks: List[Chunk], cfg: Config) -> List[Tuple[Chunk, float]]:
    """Cross-encoder reranking over candidate chunks.

    Returns (Chunk, score) sorted descending, truncated to `final_k`. Scores are
    probabilities in [0, 1] so they are directly comparable to `abstain.min_rerank_score`.
    """
    if not chunks:
        return []

    if not cfg.rerank.enabled:
        # Ablation path: no reranker means no relevance score. Return a constant so
        # Gate A cannot fire on what would otherwise be a meaningless rank fraction.
        return [(c, 1.0) for c in chunks[:cfg.retrieval.final_k]]

    model = get_reranker_model(cfg.rerank.model)
    raw = [float(s) for s in model.predict([(query, c.text) for c in chunks])]

    # sentence-transformers sigmoids single-logit rerankers already; models that emit
    # raw logits are squashed here so the threshold means the same thing either way.
    scores = raw if all(0.0 <= s <= 1.0 for s in raw) else [1.0 / (1.0 + math.exp(-s)) for s in raw]

    return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:cfg.retrieval.final_k]
