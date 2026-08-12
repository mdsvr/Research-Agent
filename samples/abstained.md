# Sample Runs: Abstentions

Verbatim console output from `py -m veritas ask "..." --offline`.

---

## Gate A — retrieval sufficiency

**Query:** `What is the capital city of France?`

```
[ABSTAINED]
Retrieved passages fell below the relevance threshold (max score: 0.00 < 0.35).
Closest matching passages found:
  - [S3::c02] "- Nested lookups allowing bypasses like `${jndi:${lower:l}${lower:d}ap://...}`.

## Remediation and Mitigation
1. Upgrad..." (relevance: 0.00)
  - [S3::c00] "# Technical Deep Dive: Log4Shell Vulnerability (CVE-2021-44228)

## Overview
CVE-2021-44228, dubbed "Log4Shell", is a re..." (relevance: 0.00)
Reason: The corpus does not contain sufficiently relevant material to answer this question.
```

The refusal is informative: it names the closest passages the corpus does contain and
the score they scored, so a reader can tell "not in this corpus" from "retrieval failed".

---

## What Gate A does and does not catch

Measured over the 30-question gold set (`py -m veritas calibrate`), with the reranker's
maximum score as the signal:

| Bucket | n | Max reranker score range | Caught by Gate A at τ=0.35 |
|---|---|---|---|
| Answerable | 15 | 0.886 – 0.998 | 0 (no over-refusals) |
| Unanswerable — topic absent from corpus | 5 | 0.0001 – 0.040 | 5 of 5 |
| Adversarial — topic present, fact absent | 10 | 0.111 – 0.976 | 3 of 10 |

The two distributions are **not** linearly separable: seven adversarial questions retrieve
a genuinely on-topic passage that simply does not contain the requested fact, and score as
high as an answerable question. Raising τ to 0.90 would cut the false-answer rate to 13%
but start refusing real questions.

That gap is Gate B's job, and the measured ladder shows it closing: false-answer rate falls
from **33.3% at Gate A alone to 6.7% with Gate B and the verifier**, on 30 questions all
answered live by `llama-3.1-8b-instant`. Of the ten adversarial questions, nine are refused.

The cost is real and is reported alongside it: two answerable questions are also refused
(13.3% over-refusal), and out of sample that cost is worse — four of twelve, 33.3%. See the
evaluation tables in the README.
