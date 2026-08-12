"""Gate A threshold calibration.

Sweeps every threshold that could change a decision and reports the resulting
false-answer / over-refusal trade-off. The previous midpoint-of-the-gap heuristic is only
meaningful when the two score distributions are linearly separable; a sweep reports the
truth either way.
"""

import json
from typing import List, Dict, Any

from veritas.config import Config, load_config
from veritas.index import search_hybrid, load_chunks
from veritas.rerank import rerank_chunks


def collect_signals(cfg: Config, gold_path: str) -> List[Dict[str, Any]]:
    """Max reranker score per gold question. Retrieval only — no generation needed."""
    with open(gold_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    # A threshold tuned on the holdout set stops being an out-of-sample measurement, and
    # nothing downstream would show that it happened. Enforced here rather than left to
    # whoever passes --gold, including a concatenated file.
    held_out = [i["qid"] for i in items if i.get("split") == "holdout"]
    if held_out:
        print(f"Skipping {len(held_out)} holdout question(s) — calibration must not see them: "
              f"{', '.join(held_out[:5])}{' ...' if len(held_out) > 5 else ''}")
        items = [i for i in items if i.get("split") != "holdout"]
    if not items:
        raise SystemExit(f"{gold_path} contains only holdout questions; nothing to calibrate on.")

    chunks = load_chunks(cfg)
    data = []
    for item in items:
        fused, _, _, _ = search_hybrid(item["question"], cfg, chunks=chunks)
        scored = rerank_chunks(item["question"], fused, cfg)
        data.append({
            "qid": item["qid"],
            "is_answerable": item["is_answerable"],
            "bucket": item.get("bucket", "unknown"),
            "max_score": max((s for _, s in scored), default=0.0),
        })
    return data


def run_calibration(cfg: Config = None, gold_path: str = "eval/gold.jsonl"):
    cfg = cfg or load_config("config.yaml")
    data = collect_signals(cfg, gold_path)

    print("\n--- MAX RERANKER SCORE PER QUERY ---")
    for d in sorted(data, key=lambda d: -d["max_score"]):
        label = "ANSWERABLE" if d["is_answerable"] else f"UNANSWERABLE ({d['bucket']})"
        print(f"[{d['qid']}] {label:<30} max rerank score: {d['max_score']:.4f}")

    answerable = [d["max_score"] for d in data if d["is_answerable"]]
    unanswerable = [d["max_score"] for d in data if not d["is_answerable"]]
    print(f"\nAnswerable   min={min(answerable):.4f}  max={max(answerable):.4f}  n={len(answerable)}")
    print(f"Unanswerable min={min(unanswerable):.4f}  max={max(unanswerable):.4f}  n={len(unanswerable)}")
    separable = min(answerable) > max(unanswerable)
    print(f"Linearly separable by this signal: {separable}")

    # Candidate thresholds: just above each observed score, plus 0.
    candidates = sorted({0.0} | {round(d["max_score"] + 1e-6, 6) for d in data})
    print("\n--- THRESHOLD SWEEP (Gate A only) ---")
    print(f"{'tau':>10}  {'FAR':>7}  {'ORR':>7}  {'accuracy':>9}")
    best = None
    for tau in candidates:
        fa = sum(1 for d in data if not d["is_answerable"] and d["max_score"] >= tau)
        tr = sum(1 for d in data if not d["is_answerable"] and d["max_score"] < tau)
        fr = sum(1 for d in data if d["is_answerable"] and d["max_score"] < tau)
        ta = sum(1 for d in data if d["is_answerable"] and d["max_score"] >= tau)
        far = fa / (fa + tr) if (fa + tr) else 0.0
        orr = fr / (fr + ta) if (fr + ta) else 0.0
        acc = (ta + tr) / len(data)
        print(f"{tau:10.4f}  {far:7.3f}  {orr:7.3f}  {acc:9.3f}")
        if best is None or acc > best[1] or (acc == best[1] and far < best[2]):
            best = (tau, acc, far, orr)

    tau, acc, far, orr = best
    print(f"\nBest Gate A threshold on this gold set: min_rerank_score = {tau:.4f}")
    print(f"  accuracy={acc:.3f}  FAR={far:.3f}  ORR={orr:.3f}")
    print("  Note: calibrated on 30 self-authored questions — treat as indicative, not tuned.")
    return best


if __name__ == "__main__":
    run_calibration()
