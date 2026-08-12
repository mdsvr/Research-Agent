import math
from typing import List, Dict, Tuple, Optional

import numpy as np

from veritas.config import Config
from veritas.generate import normalise_span
from veritas.schemas import Chunk, Claim, Verdict

# Cached (predict_fn, backend_name). `backend_name` is recorded on every Verdict so a
# heuristic score can never be mistaken for a real NLI entailment probability.
_NLI: Optional[Tuple[object, str]] = None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _word_overlap(premise: str, hypothesis: str) -> float:
    h_words = set(hypothesis.lower().split())
    p_words = set(premise.lower().split())
    return len(h_words & p_words) / max(len(h_words), 1)


def get_nli_scorer(cfg: Config):
    """Loads the configured NLI model and returns (scorer, backend_name).

    `scorer` maps a list of (premise, hypothesis) pairs to entailment probabilities.
    Three-way NLI models are softmaxed and the entailment column is selected using the
    model's own ``id2label`` map rather than a hardcoded index.
    """
    global _NLI
    if _NLI is not None:
        return _NLI

    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(cfg.verify.model)
        id2label = dict(getattr(model.model.config, "id2label", {}) or {})
        entail_idx = cfg.verify.entail_index
        if entail_idx is None:
            entail_idx = next(
                (int(i) for i, label in id2label.items()
                 if "entail" in str(label).lower() or "support" in str(label).lower()),
                None,
            )
        num_labels = int(getattr(model.model.config, "num_labels", 1))

        if num_labels > 1 and entail_idx is None:
            raise ValueError(
                f"{cfg.verify.model} exposes {num_labels} labels ({id2label}) but none "
                f"is an entailment label; set verify.entail_index in config.yaml to the "
                f"column that means 'supported', or use a model that names its labels."
            )
        if entail_idx is not None and not 0 <= entail_idx < max(num_labels, 1):
            raise ValueError(
                f"verify.entail_index={entail_idx} is out of range for {cfg.verify.model}, "
                f"which has {num_labels} labels ({id2label})."
            )

        def scorer(pairs: List[Tuple[str, str]]) -> List[float]:
            raw = np.asarray(model.predict(pairs))
            if raw.ndim == 1:  # single-logit model: already a relevance probability
                return [float(1.0 / (1.0 + math.exp(-s))) if not 0.0 <= s <= 1.0 else float(s)
                        for s in raw]
            return [float(_softmax(row)[entail_idx]) for row in raw]

        _NLI = (scorer, cfg.verify.model)
        return _NLI

    except Exception as e:
        if not cfg.verify.allow_heuristic_fallback:
            raise RuntimeError(
                f"Failed to load NLI verifier '{cfg.verify.model}': {e}\n"
                f"Fix the model name or set verify.allow_heuristic_fallback: true in "
                f"config.yaml to run with the (much weaker) word-overlap heuristic."
            ) from e

        print(f"Warning: NLI model '{cfg.verify.model}' unavailable ({e}). "
              f"Falling back to word-overlap heuristic — scores are NOT entailment probabilities.")

        def scorer(pairs: List[Tuple[str, str]]) -> List[float]:
            return [_word_overlap(p, h) for p, h in pairs]

        _NLI = (scorer, "word-overlap-heuristic")
        return _NLI


def compute_entailment_score(premise: str, hypothesis: str, cfg: Config) -> float:
    """Entailment probability for premise -> hypothesis."""
    if not premise.strip() or not hypothesis.strip():
        return 0.0
    scorer, _ = get_nli_scorer(cfg)
    return scorer([(premise, hypothesis)])[0]


def _with_title(doc_title: str, body: str) -> str:
    """Heads a premise with its source document's title.

    A chunk in isolation says "The vulnerability ... CVSS v3.1 Base Score: 10.0"; the
    identifier it refers to lives in the document title. Without the title, a claim binding
    the two ("The CVSS score for CVE-2024-3094 is 10.0") scored 0.002 — the NLI model will
    not resolve the reference across sentences. With it, 0.998, and adversarial claims that
    name a topic with no such value in the text stayed flat at 0.0005. The title is part of
    the source document, so this adds context, never evidence.
    """
    return f"{doc_title}\n\n{body}" if doc_title else body


def build_premise(claim: Claim, chunks: List[Chunk], cfg: Config) -> str:
    """Premise text for one claim against a set of cited chunks.

    When the claim carries a quote that is verbatim in one of these chunks, the premise is
    a window of the source centred on that quote rather than every cited chunk end to end.
    Measured reason: `nli-deberta-v3-small` degrades badly on long multi-topic premises —
    a true claim scored 0.05 against its full 575-char chunk and 0.93 against a 315-char
    window of the same chunk. The bare quote alone over-accepts (a false claim sharing the
    quote's phrasing scored 0.98), so the surrounding context is kept deliberately.
    """
    if not claim.quote:
        # Only safe to name one document when every cited chunk came from it.
        titles = {c.doc_id: c.doc_title for c in chunks}
        joint = "\n\n".join(c.text for c in chunks)
        return _with_title(next(iter(titles.values())), joint) if len(titles) == 1 else joint

    joint = "\n\n".join(c.text for c in chunks)
    # Same normaliser the grounding check uses: a quote that cleared that check must still
    # locate its window here, or the premise silently widens back to every cited chunk.
    needle = normalise_span(claim.quote).lower()
    for i, chunk in enumerate(chunks):
        normalised = normalise_span(chunk.text)
        position = normalised.lower().find(needle)
        if position == -1:
            continue
        pad = cfg.verify.premise_window_chars
        start = max(0, position - pad)
        end = min(len(normalised), position + len(needle) + pad)
        # Window only, headed by the source document's title. Keeping the other cited
        # chunks alongside the window was measured and is worse: a two-chunk claim scored
        # 0.92 from the window alone and 0.006 with the sibling chunk appended. Premise
        # length hurts this model more than extra context helps it.
        return _with_title(chunk.doc_title, normalised[start:end].strip())

    return joint


def verify_claims(
    claims: List[Claim],
    chunk_map: Dict[str, Chunk],
    cfg: Config,
) -> List[Verdict]:
    """Verifies each claim against its cited chunks.

    Support is the entailment probability of the premise built by `build_premise`.
    Citation precision follows ALCE: a citation is precise if dropping it leaves the
    remainder unable to entail the claim.
    """
    if not cfg.verify.enabled:
        return [
            Verdict(claim_text=c.text, citations=c.citations, supported=True, score=1.0,
                    precise_citations=list(c.citations), backend="disabled")
            for c in claims
        ]

    scorer, backend = get_nli_scorer(cfg)
    verdicts: List[Verdict] = []

    for claim in claims:
        cited_chunks = [chunk_map[cid] for cid in claim.citations if cid in chunk_map]
        if not cited_chunks:
            verdicts.append(Verdict(
                claim_text=claim.text, citations=claim.citations, supported=False,
                score=0.0, precise_citations=[], backend=backend,
            ))
            continue

        # One batched call: the full premise, then each leave-one-out premise. Dropping the
        # chunk the quote came from also drops the window, so necessity still resolves
        # against the same construction.
        premises = [build_premise(claim, cited_chunks, cfg)]
        for cid in claim.citations:
            remaining = [c for c in cited_chunks if c.chunk_id != cid]
            premises.append(build_premise(claim, remaining, cfg) if remaining else "")

        scores = [0.0 if not p.strip() else None for p in premises]
        nonempty = [(i, p) for i, p in enumerate(premises) if p.strip()]
        if nonempty:
            for (i, _), s in zip(nonempty, scorer([(p, claim.text) for _, p in nonempty])):
                scores[i] = s

        full_score = scores[0]
        is_supported = full_score >= cfg.verify.support_threshold

        precise_citations: List[str] = []
        if is_supported:
            for cid, reduced_score in zip(claim.citations, scores[1:]):
                if reduced_score < cfg.verify.support_threshold:
                    precise_citations.append(cid)

        verdicts.append(Verdict(
            claim_text=claim.text,
            citations=claim.citations,
            supported=is_supported,
            score=full_score,
            precise_citations=precise_citations,
            backend=backend,
        ))

    return verdicts
