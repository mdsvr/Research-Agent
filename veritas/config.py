import os
import yaml
from pydantic import BaseModel
from typing import List, Optional

class ChunkConfig(BaseModel):
    target_tokens: int = 96
    overlap_sentences: int = 1

class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    query_prefix: str = "Represent this sentence for searching relevant passages: "

class RetrievalConfig(BaseModel):
    hybrid: bool = True          # False = dense-only (ablation)
    dense_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    final_k: int = 6

class RerankConfig(BaseModel):
    enabled: bool = True
    model: str = "BAAI/bge-reranker-v2-m3"

class VerifyConfig(BaseModel):
    enabled: bool = True
    model: str = "cross-encoder/nli-deberta-v3-small"
    support_threshold: float = 0.5
    allow_heuristic_fallback: bool = False
    # Index of the "supported" output column. Left null, it is read from the model's own
    # id2label map, which only works for checkpoints that name the label. Binary
    # fact-checkers (MiniCheck) ship id2label={0: LABEL_0, 1: LABEL_1} and need it stated.
    entail_index: Optional[int] = None
    # Characters of source context kept either side of a claim's quote when building the
    # entailment premise. Too wide and the NLI model loses the claim in the noise; too
    # narrow and it accepts plausible-sounding substitutions. See verify.build_premise.
    premise_window_chars: int = 160

class AbstainConfig(BaseModel):
    gate_a: bool = True
    gate_b: bool = True
    min_rerank_score: float = 0.35
    min_supported_claims: int = 1

class LLMConfig(BaseModel):
    providers: List[str] = ["groq", "gemini", "openrouter", "ollama", "offline"]
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-3.6-flash"
    openrouter_model: str = "google/gemini-2.0-flash-001"
    ollama_model: str = "llama3.1:8b"
    temperature: float = 0.0
    max_retries: int = 1  # one re-ask on malformed JSON before falling to the next provider

class Config(BaseModel):
    corpus_dir: str = "data/corpus"
    index_dir: str = "data/index"
    manifest_path: str = "data/manifest.json"
    trace_dir: str = "traces"
    chunk: ChunkConfig = ChunkConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    rerank: RerankConfig = RerankConfig()
    verify: VerifyConfig = VerifyConfig()
    abstain: AbstainConfig = AbstainConfig()
    llm: LLMConfig = LLMConfig()

def load_config(config_path: str = "config.yaml") -> Config:
    if not os.path.exists(config_path):
        return Config()
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config(**data)
