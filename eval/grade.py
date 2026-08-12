"""Human grading of answer correctness, and a check on the lexical proxy.

`Gold Answer Token Recall` is the only answer-correctness number the harness produces on
its own, and it is lexical: it cannot separate a correct paraphrase from a wrong answer
sharing vocabulary with the gold one. This module puts a human in that loop —

    py -m veritas grade            # write/refresh eval/human_grades.jsonl
    py -m veritas grade --score    # read the filled-in grades back

— and then scores token recall *against* the grades, so how much the proxy can be trusted
is measured rather than assumed. The sheet deliberately omits the token-recall number:
showing a grader the score they are meant to validate would anchor them to it.
"""

import json
import os
from typing import Any, Dict, List

from eval.metrics import token_recall

GRADES_PATH = "eval/human_grades.jsonl"
RESULTS_PATH = "eval/results/ablation_results.json"
VALID_GRADES = ("correct", "partial", "wrong")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _questions(gold_paths: List[str]) -> Dict[str, str]:
    return {item["qid"]: item["question"]
            for path in gold_paths if os.path.exists(path)
            for item in _read_jsonl(path)}


def emit(results_path: str = RESULTS_PATH, out_path: str = GRADES_PATH,
         gold_paths: List[str] = None) -> int:
    """Writes one row per answered answerable question, for a human to fill in.

    Existing grades are carried over by qid: re-running after a new evaluation must never
    discard work someone did by hand.
    """
    if not os.path.exists(results_path):
        raise SystemExit(f"No evaluation results at {results_path}. Run `py -m veritas eval` first.")

    with open(results_path, "r", encoding="utf-8") as f:
        records = json.load(f)["variants"]["veritas_full"]["records"]

    questions = _questions(gold_paths or ["eval/gold.jsonl", "eval/holdout.jsonl"])
    existing = {row["qid"]: row.get("grade") for row in _read_jsonl(out_path)} \
        if os.path.exists(out_path) else {}

    rows = [{
        "qid": rec["qid"],
        "question": questions.get(rec["qid"], "(question text not found)"),
        "gold_answer": rec["gold_answer"],
        "agent_answer": rec["answer_text"],
        "grade": existing.get(rec["qid"]),  # one of VALID_GRADES, or null until graded
        "grader_note": "",
    } for rec in records
        if rec["is_answerable"] and rec.get("gold_answer") and not rec["abstained"]]

    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    carried = sum(1 for row in rows if row["grade"])
    print(f"Wrote {len(rows)} rows to {out_path} ({carried} grade(s) carried over).")
    print(f'Fill in "grade" on each row with one of: {", ".join(VALID_GRADES)}.')
    print("  correct = states the gold fact; partial = right but incomplete or hedged;")
    print("  wrong   = states something the gold answer does not support.")
    print("Then: py -m veritas grade --score")
    return len(rows)


def score(grades_path: str = GRADES_PATH) -> Dict[str, Any]:
    """Human accuracy, plus what token recall looked like for each grade."""
    if not os.path.exists(grades_path):
        raise SystemExit(f"No grading sheet at {grades_path}. Run `py -m veritas grade` first.")

    rows = _read_jsonl(grades_path)
    bad = [r["qid"] for r in rows if r.get("grade") and r["grade"] not in VALID_GRADES]
    if bad:
        raise SystemExit(f"Unrecognised grade on {bad}. Use one of: {', '.join(VALID_GRADES)}.")

    graded = [r for r in rows if r.get("grade")]
    if not graded:
        raise SystemExit(f"No rows graded yet in {grades_path}.")

    counts = {g: sum(1 for r in graded if r["grade"] == g) for g in VALID_GRADES}
    n = len(graded)
    strict = counts["correct"] / n
    lenient = (counts["correct"] + 0.5 * counts["partial"]) / n

    by_grade = {}
    for g in VALID_GRADES:
        recalls = [token_recall(r["gold_answer"], r["agent_answer"])
                   for r in graded if r["grade"] == g]
        by_grade[g] = round(sum(recalls) / len(recalls), 4) if recalls else None

    print(f"\nHuman grades: {n} of {len(rows)} rows graded")
    for g in VALID_GRADES:
        print(f"  {g:<8} {counts[g]:>3}")
    print(f"\nHuman answer accuracy   strict={strict * 100:.1f}%  "
          f"lenient(partial=0.5)={lenient * 100:.1f}%")

    print("\nMean Gold Answer Token Recall, by human grade:")
    for g in VALID_GRADES:
        value = f"{by_grade[g] * 100:.1f}%" if by_grade[g] is not None else "n/a (none)"
        print(f"  {g:<8} {value}")

    correct, wrong = by_grade["correct"], by_grade["wrong"]
    if correct is not None and wrong is not None:
        gap = correct - wrong
        verdict = ("token recall separates them; it is a usable proxy on this set"
                   if gap >= 0.25 else
                   "token recall barely separates them — do not read it as accuracy")
        print(f"\n  gap (correct - wrong) = {gap * 100:+.1f} points — {verdict}")
    else:
        print("\n  Need at least one 'correct' and one 'wrong' grade to judge the proxy.")

    return {"n_graded": n, "counts": counts, "strict": round(strict, 4),
            "lenient": round(lenient, 4), "token_recall_by_grade": by_grade}
