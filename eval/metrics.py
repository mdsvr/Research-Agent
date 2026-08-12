"""Evaluation metrics.

Two families, deliberately kept apart:

* **Gold-referenced** metrics score the agent against human-annotated answers in
  ``eval/gold.jsonl``. These are the ones that can actually be wrong.
* **Self-referenced** metrics (NLI self-entailment) score the agent's claims against the
  chunks the agent itself chose to cite. They measure internal consistency only, and are
  named so they cannot be mistaken for accuracy.
"""

from typing import List, Dict, Any


def compute_abstention_metrics(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Answer/refusal confusion matrix.

    FAR = FA / (FA + TR), the share of unanswerable questions that got an answer.
    ORR = FR / (FR + TA), the share of answerable questions that got refused.
    """
    ta = fa = fr = tr = 0
    for item in outcomes:
        if item["is_answerable"]:
            fr += item["abstained"]
            ta += not item["abstained"]
        else:
            tr += item["abstained"]
            fa += not item["abstained"]

    total = len(outcomes)
    return {
        "ta": ta, "fa": fa, "fr": fr, "tr": tr,
        "accuracy": round((ta + tr) / total, 4) if total else 0.0,
        "false_answer_rate": round(fa / (fa + tr), 4) if (fa + tr) else 0.0,
        "over_refusal_rate": round(fr / (fr + ta), 4) if (fr + ta) else 0.0,
    }


def compute_gold_citation_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Micro-averaged citation precision/recall against annotated gold chunk ids.

    The denominator for recall spans *every* answerable question, including ones the
    agent refused. Silently excluding refused questions would let an agent inflate recall
    by answering only what it finds easy.
    """
    tp = predicted_total = gold_total = 0
    scored = 0

    for rec in records:
        if not rec["is_answerable"]:
            continue  # unanswerable questions have no gold citations to match
        scored += 1
        gold = set(rec["gold_chunk_ids"])
        predicted = set(rec["predicted_chunk_ids"])
        tp += len(gold & predicted)
        predicted_total += len(predicted)
        gold_total += len(gold)

    precision = tp / predicted_total if predicted_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "questions_scored": scored,
        "citations_predicted": predicted_total,
        "citations_gold": gold_total,
    }


_STOPWORDS = {"the", "a", "an", "of", "in", "to", "and", "for", "is", "are", "was", "were",
              "on", "by", "with", "that", "this", "it", "as", "at", "from", "or"}


def token_recall(gold_answer: str, answer_text: str) -> float:
    """Share of gold-answer content words present in the agent's answer.

    A blunt lexical proxy: it cannot tell a correct paraphrase from a wrong answer that
    happens to share vocabulary. `eval.grade` scores it against human grades so the size
    of that gap is measured rather than assumed.
    """
    gold_words = {w.strip(".,;:()`\"'").lower() for w in gold_answer.split()}
    gold_words = {w for w in gold_words if w and w not in _STOPWORDS}
    answer_words = {w.strip(".,;:()`\"'").lower() for w in answer_text.split()}
    return len(gold_words & answer_words) / len(gold_words) if gold_words else 0.0


def compute_gold_answer_recall(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean `token_recall` over answerable questions. See its docstring for the caveat."""
    scores = [token_recall(rec["gold_answer"], rec["answer_text"]) for rec in records
              if rec["is_answerable"] and rec.get("gold_answer")]

    return {
        "mean_token_recall": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "questions_scored": len(scores),
    }


def compute_retrieval_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generator-independent retrieval quality: did the context window contain the gold
    chunk, and how highly was it ranked."""
    hits, reciprocal_ranks, scored = 0, [], 0

    for rec in records:
        if not rec["is_answerable"] or not rec["gold_chunk_ids"]:
            continue
        scored += 1
        gold = set(rec["gold_chunk_ids"])
        context = rec["context_injected"]
        if gold & set(context):
            hits += 1
            reciprocal_ranks.append(
                1.0 / min(i + 1 for i, cid in enumerate(context) if cid in gold)
            )
        else:
            reciprocal_ranks.append(0.0)

    return {
        "recall_at_final_k": round(hits / scored, 4) if scored else 0.0,
        "mrr": round(sum(reciprocal_ranks) / scored, 4) if scored else 0.0,
        "questions_scored": scored,
    }


def compute_self_consistency_metrics(verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """NLI self-entailment of claims against their own cited chunks (ALCE-style).

    NOT an accuracy metric: a claim copied verbatim from a chunk scores 1.0 whether or
    not it answers the question. Use alongside the gold-referenced metrics.
    """
    if not verdicts:
        return {"entailed_rate": 0.0, "citation_necessity": 0.0, "claims": 0,
                "backends": []}

    entailed = sum(1 for v in verdicts if v.get("supported"))
    total_citations = sum(len(v.get("citations", [])) for v in verdicts)
    necessary = sum(len(v.get("precise_citations", [])) for v in verdicts)

    return {
        "entailed_rate": round(entailed / len(verdicts), 4),
        "citation_necessity": round(necessary / total_citations, 4) if total_citations else 0.0,
        "claims": len(verdicts),
        "backends": sorted({v.get("backend", "unknown") for v in verdicts}),
    }


def compute_auroc(scores: List[float], labels: List[bool]) -> float:
    """AUROC of the abstention signal against the binary answerability label."""
    if len(set(labels)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(labels, scores)), 4)
