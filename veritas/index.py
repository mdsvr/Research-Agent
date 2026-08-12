import os
import json
import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict

from veritas.config import Config
from veritas.schemas import Chunk

_EMBEDDING_MODEL = None

def get_embedding_model(model_name: str):
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL

def rrf(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion algorithm across multiple ranked lists."""
    scores: Dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))

def build_indices(chunks: List[Chunk], cfg: Config):
    """Encodes dense embeddings and tokenizes the BM25 corpus, caching both to disk."""
    if not chunks:
        return

    os.makedirs(cfg.index_dir, exist_ok=True)
    texts = [c.text for c in chunks]

    model = get_embedding_model(cfg.embedding.model)
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    np.save(os.path.join(cfg.index_dir, "embeddings.npy"), embeddings)

    tokenized_corpus = [t.lower().split() for t in texts]
    with open(os.path.join(cfg.index_dir, "bm25_corpus.json"), "w", encoding="utf-8") as f:
        json.dump(tokenized_corpus, f)

def load_chunks(cfg: Config) -> List[Chunk]:
    chunks_json_path = os.path.join(cfg.index_dir, "chunks.json")
    if not os.path.exists(chunks_json_path):
        return []
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Chunk(**d) for d in data]

def _load_embeddings(cfg: Config, chunks: List[Chunk]) -> np.ndarray:
    """Loads cached embeddings, refusing to use them if they are stale."""
    emb_path = os.path.join(cfg.index_dir, "embeddings.npy")
    if os.path.exists(emb_path):
        embeddings = np.load(emb_path)
        if embeddings.shape[0] == len(chunks):
            return embeddings
        raise RuntimeError(
            f"Stale index: {emb_path} holds {embeddings.shape[0]} vectors but "
            f"chunks.json holds {len(chunks)} chunks. Re-run `py -m veritas ingest`."
        )
    model = get_embedding_model(cfg.embedding.model)
    return model.encode([c.text for c in chunks], normalize_embeddings=True)

def search_hybrid(
    query: str, cfg: Config, chunks: List[Chunk] = None
) -> Tuple[List[Chunk], List[Tuple[str, float]], List[str], List[str]]:
    """Executes dense + BM25 search over the corpus and fuses the two rankings with RRF.

    Returns (fused_chunks, fused_scores, dense_ranking, bm25_ranking). The fused list is
    the reranker's candidate pool and is deliberately not truncated to `final_k` — that
    cut happens after reranking.
    """
    if chunks is None:
        chunks = load_chunks(cfg)
    if not chunks:
        return [], [], [], []

    chunk_map = {c.chunk_id: c for c in chunks}
    texts = [c.text for c in chunks]

    # 1. Dense search with BGE query-instruction prefix
    model = get_embedding_model(cfg.embedding.model)
    query_emb = model.encode([cfg.embedding.query_prefix + query], normalize_embeddings=True)[0]
    doc_embeddings = _load_embeddings(cfg, chunks)

    dense_scores = np.dot(doc_embeddings, query_emb)
    dense_ranking = [chunks[i].chunk_id for i in np.argsort(-dense_scores)[:cfg.retrieval.dense_k]]

    # 2. BM25 search
    bm25_path = os.path.join(cfg.index_dir, "bm25_corpus.json")
    tokenized_corpus = None
    if os.path.exists(bm25_path):
        with open(bm25_path, "r", encoding="utf-8") as f:
            tokenized_corpus = json.load(f)
    if not tokenized_corpus or len(tokenized_corpus) != len(chunks):
        tokenized_corpus = [t.lower().split() for t in texts]

    # ponytail: BM25Okapi is rebuilt per query — negligible for a corpus this size,
    # cache the instance if the corpus ever outgrows a few thousand chunks.
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ranking = [chunks[i].chunk_id for i in np.argsort(-bm25_scores)[:cfg.retrieval.bm25_k]]

    # 3. Reciprocal Rank Fusion
    if cfg.retrieval.hybrid:
        fused_scores = rrf([dense_ranking, bm25_ranking], k=cfg.retrieval.rrf_k)
    else:
        fused_scores = rrf([dense_ranking], k=cfg.retrieval.rrf_k)

    fused_ids = [cid for cid in fused_scores if cid in chunk_map]
    fused_chunks = [chunk_map[cid] for cid in fused_ids]
    fused_score_tuples = [(cid, fused_scores[cid]) for cid in fused_ids]

    return fused_chunks, fused_score_tuples, dense_ranking, bm25_ranking
