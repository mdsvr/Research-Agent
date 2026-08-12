"""Evaluation harness.

Every number this file emits is produced by actually running the pipeline. The ablation
columns are real runs with stages switched off via config, not constants.
"""

import json
import os
from collections import Counter
from typing import List, Dict, Any

from veritas.config import Config
from veritas.pipeline import run_pipeline
from eval.metrics import (
    compute_abstention_metrics,
    compute_gold_citation_metrics,
    compute_gold_answer_recall,
    compute_retrieval_metrics,
    compute_self_consistency_metrics,
    compute_auroc,
)

# Ablation ladder. Each variant is a config override applied to the loaded config, so the
# numbers below the full pipeline come from the same code paths with stages disabled.
ABLATIONS = {
    "naive_dense": {"retrieval.hybrid": False, "rerank.enabled": False,
                    "verify.enabled": False, "abstain.gate_a": False, "abstain.gate_b": False},
    "hybrid_rrf": {"retrieval.hybrid": True, "rerank.enabled": False,
                   "verify.enabled": False, "abstain.gate_a": False, "abstain.gate_b": False},
    "plus_reranker": {"retrieval.hybrid": True, "rerank.enabled": True,
                      "verify.enabled": False, "abstain.gate_a": True, "abstain.gate_b": False},
    "veritas_full": {"retrieval.hybrid": True, "rerank.enabled": True,
                     "verify.enabled": True, "abstain.gate_a": True, "abstain.gate_b": True},
}


def _variant(cfg: Config, overrides: Dict[str, Any], name: str) -> Config:
    variant = cfg.model_copy(deep=True)
    for path, value in overrides.items():
        section, field = path.split(".")
        setattr(getattr(variant, section), field, value)
    variant.trace_dir = os.path.join(cfg.trace_dir, name)
    return variant


def load_gold(gold_path: str) -> List[Dict[str, Any]]:
    with open(gold_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_variant(items: List[Dict[str, Any]], cfg: Config, offline: bool) -> Dict[str, Any]:
    """Runs the full gold set through one pipeline configuration."""
    records, verdicts, signals, labels = [], [], [], []
    generators: Counter = Counter()

    for item in items:
        trace = run_pipeline(item["question"], cfg, qid=item["qid"], offline=offline)
        answer = trace.final_answer

        predicted_chunk_ids = []
        answer_text = ""
        if answer and not trace.abstained:
            predicted_chunk_ids = sorted({cid for c in answer.claims for cid in c.citations})
            answer_text = " ".join(c.text for c in answer.claims)
        if answer:
            generators[answer.generator] += 1

        records.append({
            "qid": item["qid"],
            "bucket": item.get("bucket", "unknown"),
            "is_answerable": item["is_answerable"],
            "abstained": trace.abstained,
            "gold_chunk_ids": item.get("gold_chunk_ids", []),
            "gold_answer": item.get("gold_answer"),
            "predicted_chunk_ids": predicted_chunk_ids,
            "answer_text": answer_text,
            "context_injected": trace.context_injected,
        })
        verdicts.extend(v.model_dump() for v in trace.verdicts)
        signals.append(max((s for _, s in trace.reranked), default=0.0))
        labels.append(item["is_answerable"])

    return {
        "abstention": compute_abstention_metrics(records),
        "gold_citation": compute_gold_citation_metrics(records),
        "gold_answer_recall": compute_gold_answer_recall(records),
        "retrieval": compute_retrieval_metrics(records),
        "self_consistency": compute_self_consistency_metrics(verdicts),
        "auroc_signal": compute_auroc(signals, labels),
        "generators": dict(generators),
        "records": records,
    }


def _generator_warning(generators: Dict[str, int]) -> str:
    synthetic = sum(n for g, n in generators.items()
                    if g.startswith("extractive-fallback") or g in {"cache", "none", "unknown"})

    real = {g: n for g, n in generators.items()
            if not g.startswith("extractive-fallback") and g not in {"cache", "none", "unknown"}}
    mixed = ""
    if len(real) > 1:
        # Two models answering different questions in one run makes the columns
        # incomparable — a provider failover mid-benchmark must not pass unremarked.
        mixed = (f"\n!! More than one model generated answers in this run: {real}\n"
                 f"   A provider failed over mid-benchmark. The numbers below mix models and\n"
                 f"   are not a clean measurement of either. Pin a single provider in\n"
                 f"   config.yaml (llm.providers) and re-run.\n")

    if not synthetic:
        return mixed
    return mixed + (
        f"\n!! {synthetic} of {sum(generators.values())} answers did not come from a language model.\n"
        f"   Generators used: {generators}\n"
        f"   The extractive fallback copies the top-ranked chunk verbatim: it cannot judge\n"
        f"   relevance, so it never abstains, and its claims trivially entail themselves.\n"
        f"   End-to-end abstention and citation numbers below are NOT representative of the\n"
        f"   full pipeline. Set GROQ_API_KEY (console.groq.com/keys) — or GEMINI_API_KEY /\n"
        f"   OPENROUTER_API_KEY, or run a local Ollama model — and re-run without --offline.\n"
        f"   Check your setup first with: py -m veritas providers\n"
        f"   Retrieval metrics and the AUROC signal are generator-independent and remain valid.\n"
    )


def execute_eval(gold_path: str, cfg: Config, offline: bool = True, ablations: bool = True):
    if not os.path.exists(gold_path):
        print(f"Error: Gold dataset not found at {gold_path}")
        return

    items = load_gold(gold_path)
    print(f"Evaluating {len(items)} questions from {gold_path} (offline={offline})\n")

    variants = ABLATIONS if ablations else {"veritas_full": ABLATIONS["veritas_full"]}
    results: Dict[str, Any] = {}
    for name, overrides in variants.items():
        print(f"  running variant: {name} ...")
        results[name] = run_variant(items, _variant(cfg, overrides, name), offline)

    full = results["veritas_full"]
    print(_generator_warning(full["generators"]))

    _print_table(results)
    _print_detail(full)

    results_dir = "eval/results"
    os.makedirs(results_dir, exist_ok=True)
    # Named after the question set, so an out-of-sample run cannot overwrite the in-sample
    # numbers it is supposed to be compared against.
    stem = os.path.splitext(os.path.basename(gold_path))[0]
    out_path = os.path.join(
        results_dir, "ablation_results.json" if stem == "gold" else f"ablation_results_{stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"total_questions": len(items), "question_set": gold_path,
                   "offline": offline, "variants": results}, f, indent=2)
    print(f"\nFull per-question results saved to: {out_path}")
    return results


def _model_answer_count(r: Dict[str, Any]) -> str:
    """How many of a variant's answers a language model actually produced.

    Per column, because the cache is keyed on the prompt and each variant retrieves a
    different context: one variant can replay real generations while the next falls back to
    extractive text. Comparing those two columns measures the generator mix, not the
    pipeline, and nothing else in the table would show it.
    """
    synthetic = {"cache", "none", "unknown"}
    total = sum(r["generators"].values())
    real = sum(n for g, n in r["generators"].items()
               if not g.startswith("extractive-fallback") and g not in synthetic)
    cached = r["generators"].get("cache", 0)
    return f"{real}+{cached}c/{total}" if cached else f"{real}/{total}"


def _print_table(results: Dict[str, Any]):
    names = list(results.keys())
    rows = [
        ("Answers from a model", _model_answer_count),
        ("Abstention Accuracy", lambda r: f"{r['abstention']['accuracy'] * 100:.1f}%"),
        ("False Answer Rate", lambda r: f"{r['abstention']['false_answer_rate'] * 100:.1f}%"),
        ("Over-Refusal Rate", lambda r: f"{r['abstention']['over_refusal_rate'] * 100:.1f}%"),
        ("Gold Citation Precision", lambda r: f"{r['gold_citation']['precision'] * 100:.1f}%"),
        ("Gold Citation Recall", lambda r: f"{r['gold_citation']['recall'] * 100:.1f}%"),
        ("Gold Citation F1", lambda r: f"{r['gold_citation']['f1']:.4f}"),
        ("Gold Answer Token Recall", lambda r: f"{r['gold_answer_recall']['mean_token_recall'] * 100:.1f}%"),
        ("Retrieval Recall@k", lambda r: f"{r['retrieval']['recall_at_final_k'] * 100:.1f}%"),
        ("Retrieval MRR", lambda r: f"{r['retrieval']['mrr']:.4f}"),
        ("NLI Self-Entailment", lambda r: f"{r['self_consistency']['entailed_rate'] * 100:.1f}%"),
        ("Abstention AUROC", lambda r: f"{r['auroc_signal']:.4f}"),
    ]

    width = 26
    print("=" * (width + 18 * len(names)))
    print("VERITAS AGENT EVALUATION — ALL COLUMNS MEASURED")
    print("=" * (width + 18 * len(names)))
    print("Metric".ljust(width) + "".join(n[:17].ljust(18) for n in names))
    print("-" * (width + 18 * len(names)))
    for label, fn in rows:
        print(label.ljust(width) + "".join(fn(results[n]).ljust(18) for n in names))
    print("=" * (width + 18 * len(names)))


def _print_detail(full: Dict[str, Any]):
    by_bucket: Dict[str, List[Dict[str, Any]]] = {}
    for rec in full["records"]:
        by_bucket.setdefault(rec["bucket"], []).append(rec)

    print("\nPer-bucket behaviour (veritas_full):")
    for bucket, recs in sorted(by_bucket.items()):
        abstained = sum(r["abstained"] for r in recs)
        print(f"  {bucket:<14} n={len(recs):<3} abstained={abstained:<3} answered={len(recs) - abstained}")

    backends = full["self_consistency"]["backends"]
    print(f"\nNLI verifier backend(s): {backends or ['none — no claims verified']}")
    print(f"Generator backend(s):    {full['generators']}")
