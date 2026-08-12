from pydantic import BaseModel, Field
from typing import List, Optional, Tuple

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_title: str
    text: str
    char_start: int
    char_end: int
    sent_range: Tuple[int, int]

class Claim(BaseModel):
    text: str
    citations: List[str] = Field(default_factory=list)
    # Verbatim span from a cited chunk that supports this claim. Checked against the
    # corpus by exact (whitespace-normalised) substring match — a model cannot talk its
    # way past this the way it can past an entailment score.
    quote: Optional[str] = None

class AgentAnswer(BaseModel):
    insufficient_evidence: bool = False
    claims: List[Claim] = Field(default_factory=list)
    reasoning: Optional[str] = None
    # Which generator produced this answer: a provider name, "cache", or
    # "extractive-fallback". Evaluation refuses to report headline metrics for
    # answers that did not come from a real generator.
    generator: str = "unknown"

class Verdict(BaseModel):
    claim_text: str
    citations: List[str]
    supported: bool
    score: float
    precise_citations: List[str] = Field(default_factory=list)
    # Model that produced `score`, so a heuristic can never be read as NLI entailment.
    backend: str = "unknown"

class TraceLog(BaseModel):
    qid: Optional[str] = None
    query: str
    dense_candidates: List[str] = Field(default_factory=list)
    bm25_candidates: List[str] = Field(default_factory=list)
    rrf_fused: List[Tuple[str, float]] = Field(default_factory=list)
    reranked: List[Tuple[str, float]] = Field(default_factory=list)
    context_injected: List[str] = Field(default_factory=list)
    verdicts: List[Verdict] = Field(default_factory=list)
    final_answer: Optional[AgentAnswer] = None
    dropped_claims: List[str] = Field(default_factory=list)
    ungrounded_quotes: List[str] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: Optional[str] = None
