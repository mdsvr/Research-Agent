"""Head-to-head benchmark for the NLI attribution verifier.

The verifier is the pipeline's ceiling: a claim it cannot entail is dropped, however true
it is. `eval/verifier_bench.jsonl` labels 24 claim/citation pairs drawn from this corpus —
positives include the multi-hop and imperative-restatement cases the small model failed on,
negatives carry a *verbatim* quote with one falsified value, which is the over-acceptance
case. Pairs are scored through the pipeline's own `build_premise`, so what a candidate sees
here is exactly what it would see in a run.

    py -m eval.bench_verifier                    # every candidate in CANDIDATES
    py -m eval.bench_verifier --model NAME       # one checkpoint, e.g. before adopting it

Every candidate is downloaded on first use; the large ones are ~1.6 GB.
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List

from veritas import verify
from veritas.config import Config, load_config
from veritas.index import load_chunks
from veritas.schemas import Claim

BENCH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "verifier_bench.jsonl")

# (checkpoint, entail_index). entail_index=None reads the label from the model's id2label.
CANDIDATES = [
    ("cross-encoder/nli-deberta-v3-small", None),          # the original verifier
    ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", None),
    ("tasksource/deberta-small-long-nli", None),
    ("lytang/MiniCheck-DeBERTa-v3-Large", 1),              # binary fact-checker: 1 = supported
]


def load_bench() -> List[Dict[str, Any]]:
    with open(BENCH_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score_model(model: str, entail_index, items, chunk_map, cfg: Config) -> Dict[str, Any]:
    variant = cfg.model_copy(deep=True)
    variant.verify.model = model
    variant.verify.entail_index = entail_index
    variant.verify.enabled = True
    variant.verify.allow_heuristic_fallback = False
    verify._NLI = None  # module-level scorer cache: one process, several checkpoints

    claims = [Claim(text=i["claim"], citations=i["citations"], quote=i.get("quote"))
              for i in items]
    started = time.time()
    verdicts = verify.verify_claims(claims, chunk_map, variant)
    elapsed = time.time() - started

    rows = [{"vid": i["vid"], "gold": i["supported"], "score": v.score, "note": i["note"]}
            for i, v in zip(items, verdicts)]
    tau = variant.verify.support_threshold
    tp = sum(r["gold"] and r["score"] >= tau for r in rows)
    fn = sum(r["gold"] and r["score"] < tau for r in rows)
    fp = sum(not r["gold"] and r["score"] >= tau for r in rows)
    tn = sum(not r["gold"] and r["score"] < tau for r in rows)

    pos = [r["score"] for r in rows if r["gold"]]
    neg = [r["score"] for r in rows if not r["gold"]]
    return {
        "model": model, "rows": rows, "seconds": round(elapsed, 1),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": (tp + tn) / len(rows),
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,      # true claims kept
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,  # false claims dropped
        "mean_pos": sum(pos) / len(pos), "mean_neg": sum(neg) / len(neg),
        "separation": min(pos) - max(neg),  # >0 means one threshold splits the set cleanly
        "best_tau": _best_tau(rows),
    }


def _best_tau(rows) -> float:
    """Threshold maximising accuracy on this set. Reported to show whether the configured
    0.5 is leaving anything on the table, not as a value to copy into config."""
    candidates = sorted({0.0} | {round(r["score"] + 1e-6, 6) for r in rows})
    best = max(candidates, key=lambda t: sum((r["score"] >= t) == r["gold"] for r in rows))
    return round(best, 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="score only this checkpoint")
    parser.add_argument("--entail-index", type=int, default=None,
                        help="output column meaning 'supported' (binary fact-checkers)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    chunk_map = {c.chunk_id: c for c in load_chunks(cfg)}
    items = load_bench()
    missing = {cid for i in items for cid in i["citations"] if cid not in chunk_map}
    if missing:
        raise SystemExit(f"Bench cites chunk ids that are not in the index: {sorted(missing)}. "
                         f"Run `py -m veritas ingest` first.")

    candidates = [(args.model, args.entail_index)] if args.model else CANDIDATES
    n_pos = sum(i["supported"] for i in items)
    print(f"{len(items)} labelled pairs ({n_pos} supported, {len(items) - n_pos} not), "
          f"threshold={cfg.verify.support_threshold}\n")

    results = []
    for model, entail_index in candidates:
        try:
            results.append(score_model(model, entail_index, items, chunk_map, cfg))
        except Exception as e:
            print(f"  {model}: FAILED — {type(e).__name__}: {e}\n")

    print(f"{'model':<44}{'acc':>6}{'recall':>8}{'spec':>7}{'mean+':>8}{'mean-':>8}"
          f"{'sep':>8}{'tau*':>7}{'sec':>7}")
    print("-" * 103)
    for r in results:
        print(f"{r['model'][:43]:<44}{r['accuracy']:6.2f}{r['recall']:8.2f}{r['specificity']:7.2f}"
              f"{r['mean_pos']:8.3f}{r['mean_neg']:8.3f}{r['separation']:8.3f}"
              f"{r['best_tau']:7.3f}{r['seconds']:7.1f}")

    for r in results:
        errors = [row for row in r["rows"]
                  if (row["score"] >= cfg.verify.support_threshold) != row["gold"]]
        print(f"\n{r['model']} — {len(errors)} error(s):")
        for row in errors:
            kind = "DROPPED TRUE CLAIM" if row["gold"] else "ACCEPTED FALSE CLAIM"
            print(f"  [{row['vid']}] {kind:<20} score={row['score']:.3f}  {row['note']}")

    out = os.path.join(os.path.dirname(BENCH_PATH), "results", "verifier_bench.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
