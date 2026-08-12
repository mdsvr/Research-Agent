# Veritas Agent — Verified Research Agent

> Takes a question plus a fixed corpus of source documents and produces an answer where
> every claim carries a citation to a specific text span, verified by an independent NLI
> entailment model — or an explicit, informative refusal when the corpus does not support
> an answer.

---

## ⚡ Quickstart

```bash
py -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt

py -m veritas ingest                                       # build chunks + indices
py -m veritas ask "What functions were targeted in the XZ Utils backdoor?" --offline
py -m veritas eval --offline                               # full benchmark + ablations
py -m veritas eval --gold eval/holdout.jsonl               # out-of-sample question set
py -m veritas calibrate                                    # Gate A threshold sweep
py -m veritas grade                                        # hand-grade answer correctness
py -m eval.bench_verifier                                  # compare NLI verifier models
py test_veritas.py                                         # self-check (42 assertions)
```

### Running with a generator (Groq)

The pipeline needs a language model to produce claims. With no provider reachable it
falls back to a **deterministic extractive fallback** that quotes the top-ranked passage
verbatim. That fallback is clearly labelled everywhere it appears, and the evaluation
harness refuses to present its end-to-end numbers as representative — see
[Evaluation](#-evaluation-results) below.

Groq is the default provider: free tier, fast, and OpenAI-compatible JSON mode.

1. Get a key at <https://console.groq.com/keys> (starts `gsk_`).
2. Copy `.env.example` to `.env` and paste in the keys you have — no quotes, no spaces:

   ```ini
   GROQ_API_KEY=gsk_your_key_here
   GEMINI_API_KEY=your_key_here        # optional
   OPENROUTER_API_KEY=your_key_here    # optional
   ```

   `.env` is loaded automatically on `import veritas`, so every command and the notebook
   pick it up. It is gitignored. Exported shell variables take precedence over it, so
   `$env:GROQ_API_KEY = "gsk_..."` still works if you prefer that.
3. Confirm it works, then run:

   ```bash
   py -m veritas providers                # which providers are configured & reachable
   py -m veritas ask "What are the primary controls recommended to prevent SQL injection?"
   py -m veritas eval                     # note: no --offline
   ```

Model is `llm.groq_model` in `config.yaml`, currently `llama-3.1-8b-instant` — the model the
shipped ablation ladder was measured with, chosen because its free-tier daily budget is large
enough to answer every question in every column. Stronger, with far smaller daily budgets:
`openai/gpt-oss-120b` and `llama-3.3-70b-versatile`.

`llm.providers` ships pinned to `[groq, offline]`, deliberately. With a cascade, a mid-run
rate limit hands the remaining questions to a *different* model and the columns quietly stop
describing one system — the harness warns, but only after the tokens are spent. Widen it to
`[groq, openrouter, gemini, ollama, offline]` for interactive use. A provider with no key set
is skipped silently; malformed JSON is re-asked once (`llm.max_retries`) before falling
through to the prompt-hash replay cache and then the extractive fallback.

**Rate limits.** A burst 429 is waited out (exponential backoff, `retry-after` honoured, up
to 300s per question). A 429 the provider *labels* as a per-day budget — Groq says
`tokens per day (TPD)` in the body — is not waited out at all: that provider is skipped for
the rest of the process, and if no other provider can answer, **the run aborts** instead of
degrading to extractive answers. The label is the signal, not the wait length: a spent daily
budget on a rolling window can ask for 2 seconds while an ordinary per-minute limit asks for
140. Deciding this by duration cost one benchmark run here before it was fixed.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌────────────────────────────┐
│ Hybrid Dense + BM25 Search │  bge-small-en-v1.5 + BM25Okapi
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Reciprocal Rank Fusion     │  RRF k=60 (candidate pool, not truncated to final_k)
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│ Cross-Encoder Reranker     │  bge-reranker-v2-m3 → top 6, scores in [0,1]
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐   max(rerank score) < τ_lo (0.35)
│ Gate A: Retrieval Check    ├──────────────────────────────────────┐
└─────────────┬──────────────┘                                      │
              │ pass                                                │
              ▼                                                     │
┌────────────────────────────┐                                      │
│ Structured Generation      │  Groq JSON mode + Pydantic, claim+quote schema
└─────────────┬──────────────┘                                      │
              ▼                                                     │
┌────────────────────────────┐                                      │
│ Quote Grounding Check      │  exact substring match, no model involved
└─────────────┬──────────────┘                                      │
              ▼                                                     │
┌────────────────────────────┐                                      │
│ NLI Attribution Verifier   │  MiniCheck-DeBERTa-v3-Large, quote    │
│                            │  window headed by the doc title       │
└─────────────┬──────────────┘                                      │
              ▼                                                     │
┌────────────────────────────┐   supported claims < 1               │
│ Gate B: Evidential Check   ├──────────────────────────────────────┤
└─────────────┬──────────────┘                                      │
              │ pass                                                ▼
      Verified Answer                                  Informative Abstention
```

---

## 📊 Evaluation Results

30 annotated questions in [`eval/gold.jsonl`](eval/gold.jsonl): 15 answerable, 5
unanswerable (topic absent from the corpus), 10 adversarial (topic present, requested fact
absent). This is the **tuning** set — the Gate A threshold and the generation prompt were
both shaped against it. A further 24 questions in [`eval/holdout.jsonl`](eval/holdout.jsonl)
were never used for tuning; see [Out-of-sample check](#out-of-sample-check).
Corpus: 8 documents → 26 chunks.

**Every column below is a real run, from one model.** The ablation variants are the same
pipeline with stages disabled via config overrides (`eval/run_eval.py:ABLATIONS`), executed
on the same 30 questions, every answer generated live by `llama-3.1-8b-instant`. The top row
is the receipt: no cached replays, no extractive fallbacks, nothing to discount.

| Metric | Naive dense | + Hybrid RRF | + Reranker & Gate A | Veritas full |
|---|---|---|---|---|
| **Answers from a model** | **30/30** | **30/30** | **22/22** | **22/22** |
| Abstention accuracy | 80.0% | 76.7% | 83.3% | **90.0%** |
| False answer rate (FAR) | 40.0% | 46.7% | 33.3% | **6.7%** |
| Over-refusal rate (ORR) | 0.0% | 0.0% | 0.0% | 13.3% |
| Gold citation precision | 93.8% | 82.3% | 83.3% | 88.2% |
| Gold citation recall | 83.3% | 77.8% | 83.3% | 83.3% |
| Gold citation F1 | 0.8824 | 0.8000 | 0.8333 | 0.8571 |
| Gold answer token recall | 59.0% | 65.8% | 60.6% | 57.3% |
| Retrieval recall@6 | 100.0% | 93.3% | 100.0% | 100.0% |
| Retrieval MRR | 0.8111 | 0.8667 | 0.9333 | 0.9333 |
| NLI self-entailment | 100.0% | 100.0% | 100.0% | 79.2% |
| Abstention signal AUROC | 0.5000 | 0.5000 | 0.9556 | 0.9556 |

Produced by `py -m veritas eval` (no `--offline`); per-question detail in
[`eval/results/ablation_results.json`](eval/results/ablation_results.json), archived at
[`gold_ladder_llama31_8b.json`](eval/results/gold_ladder_llama31_8b.json).

**The headline is the FAR column: 40.0% → 6.7%.** Of the 15 questions whose answer is not in
the corpus, the full pipeline answers exactly one. Gate A alone gets FAR to 33.3%; Gate B and
the verifier take it the rest of the way, which is the gap Gate A structurally cannot close —
an adversarial question retrieves a genuinely on-topic passage and scores like a real one.

### ⚠️ How to read these numbers

- **The refusals are not free, and the cost is 13.3% ORR.** Two answerable questions are
  refused by the full pipeline that `plus_reranker` answers. Both are diagnosed, neither is
  a mystery: q001's claim *"The functions within `liblzma` were targeted"* scored 0.331 —
  the verifier ceiling, item 6 under Limitations — and on q015 the generator quoted a span
  it had not cited, so quote grounding dropped it. That trade (one false answer avoided per
  two real answers lost) is the tuning knob, and it is `abstain.min_supported_claims` plus
  `verify.support_threshold`, not a mystery of the model.
- **`naive_dense` refusing 60% of unanswerable questions is the prompt, not the gates.**
  With both gates disabled the only refusal path left is the generator setting
  `insufficient_evidence` itself, which the prompt spends most of its budget on. The gates
  are what take the remaining 40% down to 6.7% — they are not carrying the whole load, and
  the table would be dishonest if it implied they were.
- **The one surviving false answer (q030) is not a hallucination.** Asked for the memory
  leaked by Heartbleed *in OpenSSL 1.1.1* — a version the bug does not affect — the model
  answered "up to 64 kilobytes ... in OpenSSL", dropping the version. The claim is true and
  the verifier scored it 0.949, correctly: it *is* entailed by the cited source. The
  verifier compares claim against source and never sees the question, so a true-but-not-
  responsive answer is structurally invisible to it. This is the same failure the verifier
  bench isolates as v22, and it is the honest ceiling of attribution-only verification.
- **Retrieval metrics and AUROC are generator-independent** and are unchanged across runs.
- **FAR of 33.3% at `plus_reranker` is Gate A working alone.** Seven of the ten adversarial
  questions retrieve a genuinely on-topic passage (reranker score up to 0.98) that simply
  lacks the requested fact. Gate A cannot separate them from real questions; that gap is
  precisely what Gate B exists for, and the 33.3% → 6.7% step is it closing.
- **Citation metrics are scored against annotated gold chunk ids**, not against the
  agent's own choices. Recall counts an answerable question the agent refused as a miss,
  so refusing cannot buy recall.
- **NLI self-entailment is reported separately** (`self_consistency` in the results JSON)
  and is explicitly *not* an accuracy metric.
- **Gold answer token recall is a lexical proxy.** `py -m veritas grade` is what checks
  whether it tracks correctness on a given run; see [Answer correctness](#answer-correctness).

### Out-of-sample check

[`eval/holdout.jsonl`](eval/holdout.jsonl) is 24 further questions over the same corpus —
12 answerable, 4 unanswerable, 8 adversarial — written after the thresholds and the prompt
were fixed. Nothing has been tuned on them, and `calibrate` refuses to read any question
carrying `split: holdout`, so that stays true by construction rather than by discipline.

Same ladder, same generator, same code — run with `py -m veritas eval --gold
eval/holdout.jsonl`. Every answer is live model output.

| Metric | Naive dense | + Hybrid RRF | + Reranker & Gate A | Veritas full |
|---|---|---|---|---|
| **Answers from a model** | **24/24** | **24/24** | **15/15** | **15/15** |
| Abstention accuracy | 70.8% | 75.0% | **79.2%** | 75.0% |
| False answer rate (FAR) | 58.3% | 50.0% | 33.3% | **16.7%** |
| Over-refusal rate (ORR) | 0.0% | 0.0% | 8.3% | 33.3% |
| Gold citation precision | 90.9% | 90.9% | 100.0% | 100.0% |
| Gold citation recall | 83.3% | 83.3% | 75.0% | 66.7% |
| Gold answer token recall | 54.0% | 52.3% | 44.5% | 41.1% |
| Retrieval recall@6 | 100.0% | 100.0% | 100.0% | 100.0% |
| Retrieval MRR | 0.7778 | 0.8194 | 0.9028 | 0.9028 |
| Abstention signal AUROC | 0.5000 | 0.5000 | 0.9028 | 0.9028 |

**What generalises and what does not.** The FAR reduction holds — 58.3% → 16.7%, a 3.5×
cut, close to the in-sample 6× — and retrieval is unchanged (recall@6 100%, MRR 0.9028 vs
0.9333, AUROC 0.9028 vs 0.9556). The **over-refusal cost does not**: ORR triples from 13.3%
in-sample to 33.3% out of sample, and that is enough to make `veritas_full`'s abstention
accuracy (75.0%) *lower* than `plus_reranker`'s (79.2%) on this set. Four of twelve
answerable holdout questions are refused. This is the single number that a tuning set would
have hidden, and it is why the holdout exists.

All four refusals are diagnosed, and only one is a threshold artefact:

| | Cause |
|---|---|
| h04, h09 | The generator paraphrased its own quote (`env` for `environment variable`), so grounding dropped it. Working as intended. |
| h05 | Verifier ceiling: *"OpenSSL version 1.0.1g fixes the Heartbleed bug"* scores 0.266 against a source phrased as an instruction ("Upgrade OpenSSL to version 1.0.1g"). |
| h07 | Gate A: max reranker score 0.32 against τ_lo = 0.35. The one genuine threshold miss, and it is 0.03 wide. |

**Both surviving false answers are the same architectural blind spot**, and it is the same
one as gold's q030. h21 asks after how many days AC-2 requires accounts to be *deleted*; the
model answered that they are *disabled* after 90 days — true, cited, entailed at 0.778.
h24 asks SwiftSafe's default alignment; the model returned the allocator signature, entailed
at 0.856. Across both question sets, **three of three** false answers are true-but-not-
responsive rather than fabricated. Quote grounding and attribution verification are working
exactly as designed and are structurally blind to this, because neither ever sees the
question. Limitation 7.

Detail: [`holdout_ladder_llama31_8b.json`](eval/results/holdout_ladder_llama31_8b.json).

### Verifier choice

The verifier sets the ceiling on the whole pipeline: a true claim it cannot entail is
dropped, and if every claim is dropped the agent refuses a question it could have answered.
[`eval/verifier_bench.jsonl`](eval/verifier_bench.jsonl) labels 24 claim/citation pairs from
this corpus — 13 supported, 11 not — scored through the pipeline's own `build_premise`, so a
candidate sees exactly what it would see in a run. The positives include the multi-hop cases
the original verifier failed; the negatives carry a **verbatim** quote with one falsified
value, which quote grounding cannot catch.

| Verifier (`py -m eval.bench_verifier`) | Accuracy | True claims kept | False claims dropped |
|---|---|---|---|
| **`MiniCheck-DeBERTa-v3-Large`** (current) | **0.92** | **92%** | 91% |
| `DeBERTa-v3-base-mnli-fever-anli` | 0.88 | 77% | **100%** |
| `deberta-small-long-nli` | 0.75 | 85% | 64% |
| `cross-encoder/nli-deberta-v3-small` (was) | 0.67 | 46% | 91% |

The old verifier dropped 7 of 13 true claims — including "Inactive accounts are
automatically disabled after 90 days" at 0.285, quoted verbatim from its own cited chunk —
and still accepted a claim the quoted span directly contradicts. Swapping it doubles true-claim
retention at unchanged specificity. MiniCheck is trained for "does this document support this
sentence" rather than strict textual entailment, which is the question the pipeline is
actually asking; that is where the multi-hop and title-coreference cases were being lost. The
threshold was **not** re-tuned on this set (0.5 throughout, where the accuracy-maximising
value would have been 0.077) — tuning the threshold on the same 24 pairs used to choose the
model is how a benchmark stops measuring anything.

### Answer correctness

`py -m veritas grade` writes [`eval/human_grades.jsonl`](eval/human_grades.jsonl), one row
per answered answerable question, and `py -m veritas grade --score` reads the grades back.
It reports human accuracy and — the point of it — mean token recall split by human grade,
so the lexical proxy is checked against the grades instead of standing in for them. The
sheet deliberately omits the token-recall number: showing a grader the score they are
validating anchors them to it.

Re-running after a new evaluation carries grades forward **only when the answer text is
unchanged**. A grade belongs to the answer it was given for, not to the question id — keyed
on qid alone, last run's verdict would ride along onto a different answer, which is the
substitution the rest of this codebase exists to refuse. Changed answers reset to ungraded
and the count is printed.

**The sheet in this repo is emitted and ungraded** — 13 rows from the shipped run, awaiting
a human. Until it is filled, `Gold answer token recall` in the tables above is an
unvalidated lexical proxy and should be read as one.

### Gate A threshold

`py -m veritas calibrate` sweeps every decision-changing threshold:

| Bucket | n | Max reranker score |
|---|---|---|
| Answerable | 15 | 0.886 – 0.998 |
| Unanswerable (topic absent) | 5 | 0.0001 – 0.040 |
| Adversarial (fact absent) | 10 | 0.111 – 0.976 |

The distributions are **not** separable. τ_lo = 0.35 is chosen as the largest threshold
that refuses zero answerable questions (ORR 0.0%) while still catching every
topic-absent question. τ = 0.90 would cut FAR to 13.3% at the cost of a 6.7% over-refusal
rate; on a 30-question self-authored set that difference is one question and not a
meaningful tuning signal.

The sweep runs on the tuning set only — questions carrying `split: holdout` are dropped
with a message before any model loads, so passing a concatenated file cannot quietly tune
the threshold on out-of-sample data.

**Out of sample, τ_lo = 0.35 costs exactly one question**: h07 ("Which command-and-control
domain did SUNBURST contact?") scores 0.32 and is refused. The threshold chosen for ORR 0.0%
in-sample delivers ORR 8.3% at Gate A out of sample — one question, 0.03 of reranker score.
That is the expected shape of a threshold fitted to 30 points, and it is small; the larger
out-of-sample cost comes from Gate B, not Gate A.

---

## 💡 Retrieval & Tool Approach

1. **Section-aware chunking.** Documents are split into atomic units — markdown headings,
   list items, table rows, and sentences — then packed into ~96-token windows with one
   unit of overlap. Each chunk stores a character span that slices back to the source
   exactly: `source[chunk.char_start:chunk.char_end] == chunk.text`, asserted in
   `test_veritas.py`.
2. **Hybrid search & fusion.** Dense retrieval (`bge-small-en-v1.5`, query-instruction
   prefixed) for semantics, `BM25Okapi` for exact tokens (CVE ids, version numbers),
   fused with Reciprocal Rank Fusion (k=60). The fused list is the reranker's candidate
   pool and is deliberately not truncated to `final_k` — that cut happens after reranking.
3. **Cross-encoder reranking.** `bge-reranker-v2-m3` re-orders candidates and keeps the
   top `final_k`. Scores are probabilities in [0,1], so `abstain.min_rerank_score` means
   the same thing regardless of which reranker is configured.
4. **Quote grounding.** Every claim must carry a verbatim supporting span from a chunk it
   cites. That span is checked by whitespace-normalised substring match against the cited
   chunks. A claim whose quote is not literally in the corpus — or is real text but from a
   chunk the claim did not cite — is discarded before any model gets a vote. It is the one
   check in the pipeline a language model cannot talk its way past.
5. **NLI attribution verification.** `MiniCheck-DeBERTa-v3-Large` scores whether the cited
   source supports each surviving claim. The supported column is read from the model's own
   `id2label` map where the checkpoint names its labels, or from `verify.entail_index` for
   binary fact-checkers that do not. Citation precision follows ALCE: a citation is precise
   if dropping it leaves the remainder unable to entail the claim. Which model to run here
   is a measured choice, not a preference — see [the verifier bench](#verifier-choice).

### Accuracy levers

The dominant failure mode measured on this benchmark is the **adversarial** bucket: the
retrieved passage is genuinely on-topic but does not contain the specific fact asked for.
Three things target it directly:

- **The prompt leads with it.** It states that being on-topic is not enough, that a
  requested number/identifier/version/score must appear verbatim, and carries a worked
  example of a correct refusal. Refusing is framed as a correct answer, not a failure.
- **The quote requirement makes fabrication expensive.** To answer at all the model must
  copy a real span; if the fact is not in the sources there is nothing to copy.
- **Grounding is checked deterministically**, so a confident-sounding invented quote is
  dropped whatever the entailment model thinks of it. If every claim is dropped, Gate B
  sees zero supported claims and the agent abstains.

---

## ⚖️ Design Tradeoffs

- **Embedding model:** `bge-small-en-v1.5` over `all-MiniLM-L6-v2` for MTEB retrieval
  performance at low CPU query latency.
- **Fusion:** RRF over score normalisation — avoids tuning blend weights across
  non-comparable cosine and BM25 scales.
- **Chunk size:** 96 tokens, not 256. The corpus documents are 110–240 words; at 256
  tokens every document collapsed into a single chunk, making citations document-level
  and the reranker score useless as a relevance signal.
- **Vector storage:** NumPy on disk over ChromaDB/Qdrant — exact search over 26 chunks is
  sub-millisecond. The loader refuses stale caches: a vector-count/chunk-count mismatch
  raises rather than silently mis-attributing citations.
- **Citation granularity:** section-level over atomic claim decomposition; avoids
  over-fragmentation and coreference errors.
- **Structured decoding:** JSON mode + Pydantic validation over grammar-constrained
  decoding.
- **Verification engine:** a dedicated entailment cross-encoder over LLM-as-judge — fast,
  offline-capable, independent of generator bias. Load failure is fatal by default; the
  word-overlap degradation is opt-in via `verify.allow_heuristic_fallback` and every
  verdict records which backend scored it, so a heuristic can never be read as NLI.
- **A fact-checker over a generic NLI model.** `MiniCheck-DeBERTa-v3-Large` is trained for
  "does this document support this sentence", which is the question actually being asked.
  Generic MNLI models answer a stricter one and treat a claim that needs a title, a
  coreference, or two sentences as neutral. Measured, that gap is most of the verifier's
  error: 46% → 92% of true claims retained at unchanged specificity.
- **An exhausted API quota stops the run.** A 429 that waiting cannot clear raises rather
  than falling through to the extractive fallback, because a benchmark quietly filled with
  answers no model produced is worse than a run that failed loudly.
- **Reranker load failure is fatal**, not silently substituted: a different reranker has a
  different score scale and would quietly invalidate the calibrated Gate A threshold.
- **Evaluation harness:** hand-rolled over RAGAS/TruLens, for full explainability.

---

## ⚠️ Known Limitations

1. **The benchmark is measured on one small generator.** Every column comes from
   `llama-3.1-8b-instant`, chosen because it is the only free-tier model whose daily budget
   covers all four ablation columns. A spot run of `openai/gpt-oss-120b` on the full pipeline
   reached **FAR 0.0%** (versus 6.7%) at **ORR 26.7%** (versus 13.3%) — a stronger model
   refuses more, and more accurately — but its 200k-token daily budget cannot fund a ladder,
   so it is a single archived column
   ([`gold_veritas_full_gptoss120b.json`](eval/results/gold_veritas_full_gptoss120b.json)),
   not a headline. Note that run predates the quote-grounding fix below, so its ORR is an
   overestimate. Numbers here will move with the generator; the pipeline is what is fixed.
2. **Both question sets are self-authored.** The holdout set is out-of-sample — nothing was
   tuned on it, and `calibrate` refuses to read it — but the same person wrote both it and
   the corpus. It removes the tuning-overfit reading of the numbers, not the author-bias
   one. A set written by someone else remains stronger evidence.
3. **Small-sample calibration.** τ_lo is calibrated on the 30 tuning questions; one
   question is 3.3 percentage points. Out of sample it costs one refusal (h07, scoring 0.32
   against a 0.35 threshold) — small, but the sets are 30 and 24 questions, so no threshold
   here should be read to two significant figures.
4. **Answer correctness is graded by hand on a sample, not on every run.**
   `gold_answer_token_recall` is bag-of-words overlap and stays in the table because it is
   cheap and reruns automatically; `py -m veritas grade` is what tells you whether it means
   anything on a given run, by comparing it against human grades. Trust the proxy only as
   far as that comparison shows it separating correct answers from wrong ones.
5. **Quote grounding refuses elided quotes, including correct ones.** Observed with
   `gpt-oss-120b`: a true claim whose supporting quote joined two real spans with an
   ellipsis. No such string exists in the corpus, so the claim was dropped and the agent
   abstained. Accepting `...`-joined segments would fix it and would still reject
   fabrications, but the check is deliberately left with no latitude to interpret what the
   model meant. Formatting *is* now tolerated — markdown emphasis is stripped before
   matching, after two over-refusals were traced to a model faithfully copying the rendered
   text of a bolded list item.
6. **The verifier still misses a claim that needs two sentences joined.** Measured on
   `eval/verifier_bench.jsonl`: the one true claim MiniCheck drops binds a parenthetical
   ("5 consecutive invalid attempts") to the window around it. Better than the 7 the
   previous model dropped, not zero. The same ceiling costs a real question in the shipped
   run: q001's *"The functions within `liblzma` were targeted"* scores 0.331.
7. **Attribution verification cannot see the question.** The verifier scores claim against
   source, so a claim that is true, cited and entailed passes even when it does not answer
   what was asked. The single false answer surviving the full pipeline (q030) is exactly
   this: the question fixed a version the vulnerability does not affect, and the model
   answered the un-versioned fact instead. Catching it needs a question-aware check —
   an answer-relevance judge — which is a different component, not a threshold change.
11. **The verifier accepts some plausible lexical substitutions.** Measured: it scores the
    false claim *"Heartbleed affects OpenSSL versions 1.1.1 through 1.1.1f"* at 0.92 against
    a premise stating 1.0.1 through 1.0.1f. Quote grounding does not catch this — the model
    can cite a real span and still substitute a near-identical value in the claim. Swapping
    the verifier narrowed this class (the old model also accepted a directly contradicted
    claim) without closing it. `DeBERTa-v3-base-mnli-fever-anli` scored 1.00 specificity on
    the same set at the cost of 15 points of recall; if false accepts matter more than
    dropped claims for your use, that is the configured trade to change.
12. **Claim mood changes the score more than claim truth does.** Identical content scored
    0.02 as an imperative ("Use parameterized queries…") and 0.99 as a declarative
    ("Parameterized queries are recommended…"). The generation prompt therefore carries a
    hard syntactic rule — a claim may never begin with a verb — because the verifier is
    only meaningful on propositions. A soft instruction was not enough: llama-3.3-70b
    ignored it and mirrored the imperative mood of the source documents.
6. **No live web search.** The agent operates strictly over the provided corpus.
7. **Single-turn.** No conversational state.
8. **English technical prose only.**
9. **Extractive bias.** Stronger on direct factual spans than on broad summarisation.
10. **Off-the-shelf reranker.** May need domain fine-tuning for non-standard technical text.

---

## 📁 Repository Structure

```
Research agent/
├── README.md                   # this file
├── .env                        # API keys — paste yours here; gitignored
├── .gitignore
├── config.yaml                 # centralised configuration
├── requirements.txt            # dependencies
├── test_veritas.py             # assert-based self-check (py test_veritas.py)
├── tasks/                      # 18 task specification files, incl. plan.md
├── veritas/
│   ├── __main__.py             # entry point for `py -m veritas`
│   ├── cli.py                  # ingest | ask | eval | calibrate | grade | providers
│   ├── config.py               # pydantic config model
│   ├── chunking.py             # unit-aware chunker with exact char offsets
│   ├── ingest.py               # corpus loader & indexer
│   ├── index.py                # dense + BM25 hybrid search & RRF
│   ├── rerank.py               # cross-encoder reranker
│   ├── generate.py             # provider cascade + structured generation
│   ├── verify.py               # NLI entailment verifier (ALCE citation precision)
│   ├── abstain.py              # Gate A / Gate B
│   ├── pipeline.py             # end-to-end orchestration
│   ├── schemas.py              # pydantic schemas
│   └── trace.py                # structured trace writer
├── data/
│   ├── corpus/                 # 8 source technical documents
│   ├── index/                  # chunks.json, embeddings.npy, bm25_corpus.json
│   └── manifest.json           # document metadata
├── eval/
│   ├── gold.jsonl              # 30 annotated benchmark questions (tuning set)
│   ├── holdout.jsonl           # 24 out-of-sample questions; nothing was tuned on these
│   ├── verifier_bench.jsonl    # 24 labelled claim/citation pairs for the NLI verifier
│   ├── bench_verifier.py       # head-to-head verifier comparison
│   ├── grade.py                # human grading sheet + proxy-vs-grade comparison
│   ├── metrics.py              # gold-referenced and self-referenced metrics
│   ├── run_eval.py             # evaluation + real ablation ladder
│   ├── calibrate.py            # Gate A threshold sweep (refuses holdout questions)
│   └── results/                # ablation_results*.json, verifier_bench.json
├── fixtures/
│   ├── cache.json              # prompt-hash replay cache (never keyed by question id)
│   └── curated_answers_UNUSED.json   # retired; see note below
├── samples/                    # verbatim sample runs
└── traces/                     # per-query pipeline traces (written at run time)
```

**`fixtures/curated_answers_UNUSED.json`** holds 30 hand-written answers that a previous
version of `generate.py` returned whenever the evaluation harness passed a question id.
That made the benchmark score the file rather than the pipeline. The cache is now keyed on
a SHA-256 of model + prompt only; the file is retained for reference and is never read.
`test_veritas.py` asserts that no non-hash key can enter the live cache.
