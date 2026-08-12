# PLAN.md — Veritas Agent

**Rooman 24-Hour AI Agent Challenge · Junior AI Research Associate**
**Chosen agent:** Research Agent (with Citations) — Category 4, Advanced

---

## 0. The one sentence

> **My agent takes a question plus a fixed corpus of provided source documents, and produces an answer where every claim carries a verified citation to a specific text span — or an explicit refusal when the corpus does not support an answer.**

Everything in this plan exists to serve that sentence. If a feature does not make that sentence more true, cut it.

### What "verified" means (the whole point of this build)

Most submissions will do this:

```
retrieve chunks → ask LLM to answer and add [1] [2] markers → print it
```

The `[1]` is whatever the LLM felt like typing. Nobody checked it. That is **citation theatre**.

We do this instead:

```
retrieve chunks → LLM emits structured claims with chunk IDs
              → a SEPARATE model checks each (claim, cited chunk) pair for entailment
              → unsupported claims are dropped or flagged
              → if too little survives, we refuse
```

The second model is the differentiator. It is a small NLI / fact-checking model that answers one question: *"Does this passage actually support this sentence?"* It has no incentive to be agreeable because it never saw the generation prompt.

### Scope decisions locked in now (do not revisit at hour 14)

| Decision | Choice | Why |
|---|---|---|
| Live web search | **No** | Brief permits it, rubric doesn't reward it, network flakiness during review kills the 30-point "working agent" score |
| UI | **No** (CLI only) | Brief explicitly says CLI is enough. A UI costs 3h and earns 0 rubric points |
| Fine-tuning | **No** | No rubric points, huge time sink |
| Multi-hop reasoning | **Best-effort, not a goal** | Say so honestly in tradeoffs |
| Citation granularity | **Sentence-level** | Claim-level decomposition is fragile; sentence-level is what Anthropic's Citations API ships |

---

## 1. Repository layout

Create this on the first commit. Reviewers score code organisation (20 points) and a clean tree communicates competence before they read a line.

```
veritas-agent/
├── README.md                  # written LAST, matters MOST
├── PLAN.md                    # this file — commit it, shows your thinking
├── requirements.txt           # exact pins
├── config.yaml                # every tunable in one place
├── .env.example               # API key names, no values
├── .gitignore
│
├── veritas/
│   ├── __init__.py
│   ├── cli.py                 # entry point: ingest / ask / eval
│   ├── config.py              # loads config.yaml into a dataclass
│   ├── chunking.py            # documents → chunks
│   ├── index.py               # dense + BM25 + RRF fusion
│   ├── rerank.py              # cross-encoder reranker
│   ├── generate.py            # LLM call, structured output, provider fallback
│   ├── verify.py              # NLI entailment verifier
│   ├── abstain.py             # the two abstention gates
│   ├── pipeline.py            # wires everything; the only file that knows the full flow
│   ├── schemas.py             # Pydantic models
│   └── trace.py               # structured logging of every stage
│
├── data/
│   └── corpus/                # your source documents (committed!)
│
├── eval/
│   ├── gold.jsonl             # the question set
│   ├── metrics.py             # citation P/R/F1, faithfulness, abstention
│   ├── run_eval.py
│   └── results/               # committed output — reviewers see numbers
│
├── fixtures/                  # cached LLM responses for offline runs
│   └── cache.json
│
└── samples/
    ├── answered.md            # 5+ example runs, answered
    ├── abstained.md           # 3+ example runs, refused
    └── trace_example.json     # one full pipeline trace
```

**Why `trace.py` exists:** for every query you log what BM25 returned, what dense returned, what RRF fused, what the reranker promoted/demoted, what was injected into the prompt, and what the verifier said. When an answer is wrong you can say *"retrieval found it at rank 7, the reranker demoted it to 12, so it never reached the prompt"* instead of shrugging. This single file is the most senior-engineer thing in the repo and takes 40 minutes.

---

## 2. Phase-by-phase build

Each phase below has: **goal → why it matters → exactly what to do → how you know it works → what to cut if you're behind.**

---

### PHASE 0 — Scaffold and pin (H0 → H1)

**Goal:** a repo that runs `python -m veritas --help` and prints usage.

**Why it matters:** the first commit timestamps the start of your 24-hour window. Late commits are not evaluated, so a clean early commit is your proof of honest work. Also: dependency hell at hour 20 has killed more hackathon projects than bad ideas.

**Do:**

1. `git init`, create the tree above with empty `__init__.py` files, push to a **public** GitHub repo immediately.
2. Create the venv and install everything **now**, before you write logic:

```
sentence-transformers
faiss-cpu           # or skip, use numpy
rank-bm25
transformers
torch
pydantic
pyyaml
click               # or typer
pypdf               # if your corpus has PDFs
numpy
scikit-learn        # for AUROC only
```

3. `pip freeze > requirements.txt` — **exact pins, not ranges.** Note your Python version in the README.
4. Write `config.yaml` with every knob you'll ever touch:

```yaml
corpus_dir: data/corpus
chunk:
  target_tokens: 256
  overlap_sentences: 1
embedding:
  model: BAAI/bge-small-en-v1.5
retrieval:
  dense_k: 20
  bm25_k: 20
  rrf_k: 60
  final_k: 6
rerank:
  enabled: true
  model: BAAI/bge-reranker-v2-m3
verify:
  model: vectara/hallucination_evaluation_model
  support_threshold: 0.5
abstain:
  min_rerank_score: null     # filled in at calibration, Phase 10
  min_supported_claims: 1
llm:
  providers: [gemini, openrouter, ollama, offline]
  temperature: 0.0
```

**Why one config file:** at hour 12 you will calibrate thresholds. If those numbers are scattered across five files you will miss one and your eval will not match your shipped behaviour. Also it makes the tradeoff notes trivial to write — you just narrate your own config.

**Done when:** `python -m veritas --help` prints, repo is public, first commit pushed.

**Cut if behind:** nothing. This phase is non-negotiable.

---

### PHASE 1 — Choose and prepare the corpus (H1 → H1.5)

**Goal:** 8–15 documents in `data/corpus/`, committed to the repo.

**Why it matters:** this is an agent-specific deliverable ("Source documents"). It must be in the repo or the reviewer cannot reproduce anything. It also determines how impressive your demo feels.

**Choosing well.** You want a corpus that is:

- **Bounded and coherent** — one domain, so questions are natural.
- **Factually dense** — dates, numbers, named entities. These make citation verification meaningful and make hallucination visible.
- **Not in the LLM's training data, ideally** — otherwise the model answers from memory and your retrieval could be broken without you noticing. This is a real trap.

**Strong options:**

| Option | Pro | Con |
|---|---|---|
| A set of arXiv papers on one narrow topic | Dense, technical, citation-natural | Possibly memorised by the LLM |
| Government/regulatory PDFs (RBI circulars, policy docs) | Dry, specific, unlikely memorised | Parsing can be messy |
| Company annual reports / 10-Ks | Numbers everywhere, great for adversarial questions | Long, needs good chunking |
| Your own domain: cybersecurity advisories, CVE writeups | Plays to your background, defensible in interview | Make sure it's public |

**Recommendation:** given your SwiftSafe background, a corpus of **public security advisories / NIST or OWASP documents / CVE analyses** is a strong pick — you can speak to it fluently if they interview you on it, and it is fact-dense.

**The memorisation test (do this, it takes 5 minutes):** pick three questions your corpus answers. Ask the raw LLM with **no context at all**. If it answers correctly, your corpus is memorised and your eval is compromised for those questions — either pick more obscure documents or lean on questions about specific numbers/dates that models get wrong from memory.

**Do:**
1. Drop files in `data/corpus/`.
2. Write a loader that handles `.txt`, `.md`, and `.pdf` (pypdf). Normalise whitespace, strip headers/footers if they repeat on every page.
3. Assign each document a stable ID: `S1`, `S2`, ... and record `{id, filename, title}` in a manifest.

**Done when:** `python -m veritas ingest` prints "Loaded 12 documents, 847 chunks."

**Cut if behind:** use 8 plain `.txt` files, skip PDF parsing entirely.

---

### PHASE 2 — Chunking (H1.5 → H2.5)

**Goal:** documents split into retrievable, citable units.

**Why it matters more here than in normal RAG:** a citation must point to something a human can verify in a few seconds. If your chunk is 1000 tokens, the citation "[S3]" is nearly useless — the reader still has to hunt. Small chunks make citations *precise*, and they make your NLI verifier's job easier (short premise, short hypothesis = higher accuracy).

**Strategy: sentence-window chunking.**

1. Split each document into sentences (regex on `[.!?]` followed by whitespace+capital is fine; or `nltk.sent_tokenize` if you want).
2. Greedily accumulate sentences until you approach ~256 tokens.
3. Carry **1 sentence of overlap** into the next chunk.

**Why 256 and not 512?** General RAG guidance favours 400–512 tokens for recall. But we are optimising for *citation traceability*, and there is a direct tension:

- Bigger chunks → better recall, worse citation precision, harder entailment checking.
- Smaller chunks → precise citations, but a fact split across a boundary becomes uncitable.

256 with sentence-aware boundaries is the compromise. **Write this tension down** — it is exactly the kind of reasoning the 10-point tradeoff section rewards.

**Why overlap?** A fact spanning a chunk boundary otherwise becomes unretrievable. Note honestly in your tradeoffs that overlap's benefit is contested — at least one benchmark found no measurable gain — so you treat it as cheap insurance rather than established fact.

**Critical implementation detail — keep the pointers:**

```python
@dataclass
class Chunk:
    chunk_id: str        # "S3::c12"  ← document + position, human-readable
    doc_id: str          # "S3"
    doc_title: str
    text: str
    char_start: int      # offset in original document
    char_end: int
    sent_range: tuple    # (first_sent_idx, last_sent_idx)
```

`char_start`/`char_end` let you show the reader exactly where the claim came from, which is what Anthropic's Citations API does to guarantee citations are real spans rather than model inventions. It costs you nothing to store and it makes your output feel genuinely trustworthy.

**Done when:** you can print any chunk with its source document and character range, and no chunk exceeds ~350 tokens.

**Cut if behind:** fixed 300-token character splits with 50-char overlap. Ugly but functional.

---

### PHASE 3 — Retrieval: dense + sparse + fusion (H2.5 → H4)

**Goal:** given a question, return the ~20 most plausibly relevant chunks.

**Why hybrid and not just embeddings:** dense embeddings are good at paraphrase and bad at exact tokens. Ask "what does CVE-2024-3094 affect?" and a dense retriever may return semantically-similar-but-wrong CVEs, because the embedding smooths over the exact identifier. BM25 nails exact identifiers, numbers, and rare terms. They fail differently, which is precisely why fusing them works.

**3a. Dense.**

- Model: `BAAI/bge-small-en-v1.5` (~130MB, 384-dim). Strong MTEB standing for its size, runs comfortably on CPU.
- Fallback if latency hurts: `all-MiniLM-L6-v2` (~22MB). Noticeably weaker but 5× faster.
- Encode all chunks once at ingest, cache the matrix to disk (`.npy`). Re-encoding on every run wastes reviewer patience.
- **BGE models want a query prefix.** BGE v1.5 expects queries encoded with an instruction prefix like `"Represent this sentence for searching relevant passages: "` while passages get none. Getting this wrong silently degrades retrieval by several points and is one of the most common quiet bugs in hackathon RAG. Check the model card and do it right.

**3b. Sparse.**

`rank_bm25.BM25Okapi` over lowercased, whitespace-tokenised chunks. Twenty lines. Do not over-engineer the tokeniser.

**3c. Fusion — Reciprocal Rank Fusion.**

Do **not** try to normalise BM25 scores against cosine similarity. They live on incomparable scales, the normalisation is corpus-dependent, and tuning the blend weight will eat two hours. RRF sidesteps this entirely by fusing on **rank**, not score:

```python
def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """rankings: list of ranked chunk_id lists, best first."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(sorted(scores.items(), key=lambda x: -x[1]))
```

That's the whole algorithm. `k=60` is the standard constant from the original paper; it damps the influence of top ranks so one retriever cannot dominate. Take dense top-20 and BM25 top-20 in, take top-20 fused out.

**Why this is a good tradeoff-notes point:** you can say "I chose RRF over score normalisation because score scales are incomparable and rank-based fusion requires no tuning — a deliberate choice to spend my calibration budget on the abstention threshold instead."

**Done when:** for five hand-written questions, the chunk you know contains the answer appears in the top 20. If it doesn't, fix this **now** — nothing downstream can recover from a retrieval miss.

**Cut if behind:** dense only. Say so honestly in tradeoffs and note it as the first thing you'd add back.

---

### PHASE 4 — Reranking (H4 → H5)

**Goal:** re-order the top-20 into a top-6 that actually goes in the prompt, and **produce your primary abstention signal.**

**Why a cross-encoder:** bi-encoders (your embedding model) encode question and chunk *separately* — they never see them together, so they can't reason about the interaction. A cross-encoder takes `(question, chunk)` as a single input and outputs a relevance score with full attention across both. Far more accurate, far slower — which is fine, because you only run it on 20 candidates, not the whole corpus. This is the classic retrieve-then-rerank cascade.

**Model:** `BAAI/bge-reranker-v2-m3`. If too heavy on CPU, `cross-encoder/ms-marco-MiniLM-L-6-v2` is a solid lighter option.

**The dual purpose — this is important.** The reranker score is not just for ordering. It is the best cheap signal you have for *"is the answer even in this corpus?"* A cross-encoder that has genuinely compared the question against the best available chunk and returns a low score is telling you something a cosine similarity cannot. Cosine similarity is **notoriously poorly calibrated** — every chunk in a coherent corpus is somewhat similar to every question about that corpus, so cosine scores cluster in a narrow band and thresholding them is close to arbitrary. This distinction is worth a paragraph in your tradeoff notes.

**Honest caveat to include:** off-the-shelf rerankers are trained on MS MARCO-style web data and sometimes *degrade* ranking on specialised technical corpora. Put it behind `rerank.enabled` in config, measure the delta on your gold set in Phase 9, and report the actual number — whichever way it goes. Reporting "I measured it and it hurt, so I disabled it" scores better than assuming it helped.

**Done when:** you log the before/after ordering for a query and can see the reranker moving the right chunk up.

**Cut if behind:** skip it. Use RRF top-6 directly, and use `max(dense_cosine)` as a weaker abstention signal. Note the downgrade explicitly.

---

### PHASE 5 — Generation with structured citations (H5 → H7)

**Goal:** the LLM returns claims, each tagged with the chunk IDs it used.

**5a. The prompt.** Number the chunks in the context explicitly so the model has unambiguous IDs to cite:

```
You are a research assistant. Answer the QUESTION using ONLY the SOURCES below.

Rules:
- Every claim must be supported by at least one source.
- Cite using the exact source IDs shown, e.g. ["S3::c12"].
- Cite the MINIMUM number of sources needed. Do not cite a source
  that is not necessary for the claim.
- If the sources do not contain enough information to answer,
  return an empty claims list and set insufficient_evidence to true.
- Do not use outside knowledge. Do not guess.

SOURCES:
[S3::c12] <chunk text>
[S7::c04] <chunk text>
...

QUESTION: <question>

Return JSON matching this schema: {...}
```

Two lines are doing heavy lifting:

- **"Cite the MINIMUM number of sources"** — citation *precision* penalises unnecessary citations. A model that dumps all six IDs on every sentence gets perfect recall and terrible precision. You must prompt against this.
- **"return an empty claims list and set insufficient_evidence"** — you are giving the model an explicit, structured escape hatch. Research on insufficient context consistently finds models hallucinate rather than abstain when not given a clean way out.

**5b. Structured output.** Define the schema in `schemas.py`:

```python
class Claim(BaseModel):
    text: str
    citations: list[str]          # chunk IDs

class AgentAnswer(BaseModel):
    insufficient_evidence: bool
    claims: list[Claim]
    reasoning: str | None = None
```

Use the provider's **JSON mode / function calling**, then validate with Pydantic, then retry once on failure, then fall back to a regex extractor.

**Deliberately do NOT use hard grammar-constrained decoding.** There's a documented "constraint tax" — strict token-masking to enforce a schema competes with the model's reasoning capacity and can lower extraction accuracy. Anthropic's API won't even allow strict JSON schemas combined with its Citations feature, because the two mechanisms conflict. JSON mode + validation + retry gets you 99% reliability without the tax. **This is a genuinely sophisticated tradeoff to explain** — most people assume more constraint is strictly better.

**5c. Validate the citations exist.** Before anything else, drop any citation ID that was not in the retrieved set. Models occasionally invent plausible-looking IDs. This is a two-line check that eliminates an entire class of failure.

**5d. Provider fallback.** You have built this before — reuse the pattern:

```
Gemini 2.0 Flash → OpenRouter → Ollama (local) → offline deterministic
```

The **Ollama tier is the one that matters for this challenge**, because it means a reviewer with no API key can still run your agent. Ollama exposes an OpenAI-compatible endpoint at `localhost:11434`, so this is a base-URL swap, not a code fork.

The **offline deterministic tier** — for the extractive case, return the top reranked chunk verbatim as a single claim citing itself. Not smart, but it always runs and it keeps `eval` reproducible.

Set `temperature=0` throughout. You want reproducibility, and a reviewer re-running your samples should get your samples.

**Done when:** you get valid `AgentAnswer` objects for ten different questions, including at least one where `insufficient_evidence=true`.

**Cut if behind:** single provider + regex citation extraction. But **do not cut the Ollama path** — it's worth more than most features here.

---

### PHASE 6 — The attribution verifier (H7 → H9)

**This is the heart of the project. Protect these two hours.**

**Goal:** independently confirm that each cited chunk actually supports its claim.

**The framing.** For each claim you have a hypothesis (the claim text) and a premise (the concatenation of its cited chunks). You are asking a natural language inference question: *does the premise entail the hypothesis?* The generating LLM does not get a vote. That independence is what makes the verification meaningful.

**6a. Model choice.** Three viable options, pick one and justify it:

| Model | Size | Strength | Use when |
|---|---|---|---|
| `lytang/MiniCheck-Flan-T5-Large` | 770M | Purpose-built for "does this doc support this claim"; reaches GPT-4-level accuracy at a fraction of the cost | **Default choice.** Best accuracy/size ratio |
| `vectara/hallucination_evaluation_model` (HHEM-2.1-Open) | FLAN-T5 base, <600MB RAM | Outputs a calibrated 0–1 probability — doubles as a confidence number | You want the score itself, not just a label. Weaker on summarised (vs extractive) answers |
| `cross-encoder/nli-deberta-v3-base` | 184M | Generic 3-way NLI, fastest | CPU-constrained, or as a fallback tier |

**Recommendation:** MiniCheck as primary, `nli-deberta-v3-small` as the low-resource fallback behind a config flag. Two tiers means the reviewer's laptop can't stop your demo.

**6b. The algorithm.**

```python
def verify_claim(claim: Claim, chunks: dict[str, Chunk]) -> Verdict:
    premise = "\n\n".join(chunks[cid].text for cid in claim.citations)
    score = nli_support_probability(premise, claim.text)   # 0..1
    return Verdict(
        supported = score >= cfg.verify.support_threshold,
        score = score,
    )
```

**Why concatenate rather than check each citation separately?** A claim synthesised from two chunks may not be entailed by either one alone. Checking the union is what ALCE does and it is the correct semantics for "these sources together support this."

**6c. Per-citation necessity (for precision).** After confirming the union entails the claim, test each citation by **removing** it and re-checking. If entailment survives without citation X, then X was unnecessary — it hurts precision. This is exactly the ALCE citation-precision definition. It costs one extra NLI call per citation, which is affordable at your scale, and it lets you either strip redundant citations from the output or report precision honestly.

**6d. What to do with unsupported claims.** Three policies — pick one, make it configurable, and explain the choice:

1. **Drop** — cleanest output, risks an incomplete answer.
2. **Flag** — keep with a ⚠ marker and the score. Most transparent, best for a demo.
3. **Regenerate** — retry with a "these claims lacked support" note. Costs latency; can loop.

**Recommendation:** flag by default (`⚠ unverified`), with drop available via config. A demo that visibly shows the verifier catching something is more persuasive than one that silently cleans up.

**6e. Known failure mode to document:** NLI models under-detect **multi-hop** and heavily **paraphrastic** support. A claim correctly synthesised from two chunks may register as "neutral." You mitigate by concatenating cited chunks, and you note the residual limitation honestly. You could add an LLM-as-judge fallback for claims scoring in an ambiguous middle band (say 0.35–0.5) if time allows — flexible and better at paraphrase, but slower and itself hallucination-prone.

**Done when:** you can construct a claim that is *plausible but unsupported* by its cited chunk, and watch the verifier catch it. **Save this example — it goes in your README.** It is the single most convincing artifact in your submission.

**Cut if behind:** never cut this. Cut the reranker, cut BM25, cut PDF parsing. Keep the verifier.

---

### PHASE 7 — The abstention gates (H9 → H10)

**Goal:** implement "the provided sources do not contain the answer" as a real, measurable decision rather than a prompt instruction.

**Why two gates.** A single point of failure will fail. The two gates catch different situations:

**Gate A — pre-generation (retrieval sufficiency).**
```
if max(reranker_scores) < τ_lo:  → ABSTAIN, don't even call the LLM
```
Catches: the corpus plainly has nothing on this topic. Saves an API call and removes any chance of the LLM confabulating from weak context.

**Gate B — post-verification (evidential sufficiency).**
```
if count(supported_claims) < min_supported_claims:  → ABSTAIN
```
Catches: chunks *looked* relevant (right topic!) but don't actually contain the asked fact. **This is the hard adversarial case** — related-but-insufficient context — and the one that separates you from the field. Gate A cannot catch it because retrieval scores look fine.

**Also honour** `insufficient_evidence=true` from the LLM as a third, softer signal.

**Signal quality, ranked** (worth putting in your tradeoff notes):

1. Reranker cross-encoder score — best cheap signal
2. NLI-verified support count — best semantic signal, catches what #1 can't
3. Self-consistency across N samples — robust, costs N×
4. LLM verbalised confidence — cheap, moderate
5. **Raw cosine similarity — weak, poorly calibrated, do not threshold on it alone**

That ranking, stated explicitly, demonstrates you understand *why* you built it this way rather than having stumbled into it.

**The abstention message must be useful:**

```
I cannot answer this from the provided sources.

The closest material I found was:
  [S4::c07] "…" (relevance 0.31)
  [S9::c02] "…" (relevance 0.28)

These discuss <topic> but do not state <the specific thing asked>.
```

A bare "I don't know" wastes the interaction. Showing *what you found and why it fell short* is the behaviour GopherCite pioneered and it is what a research associate should do.

**Done when:** three known-unanswerable questions abstain, and five known-answerable ones don't. Thresholds are still guesses at this point — Phase 10 fixes that.

---

### PHASE 8 — The gold question set (H10 → H12)

**Goal:** `eval/gold.jsonl` with 30–60 labelled questions.

**Why this is worth two full hours:** without it you cannot calibrate thresholds (Phase 10), cannot report metrics (Phase 9), and cannot claim anything about your system. It converts "I built a thing" into "I built a thing and measured it" — which is what the *Research Associate* title is asking for. It is also an explicit required deliverable ("A question set").

**The three buckets — deliberately unbalanced toward the interesting cases:**

**Bucket 1 — Answerable (~50%, 15–30 questions).**
Fact is clearly in the corpus. Record the gold supporting chunk IDs so you can compute citation recall.
> *"What encryption algorithm does the X protocol specify for key exchange?"*

**Bucket 2 — Unanswerable (~25%, 8–15 questions).**
Topic is absent from the corpus entirely. Correct behaviour: abstain.
> *"What is the annual revenue of Company Y?"* (corpus is technical advisories, contains no financials)

**Bucket 3 — Adversarial, related-but-insufficient (~25%, 8–15 questions). ← the one that matters.**
The corpus discusses the topic but not the specific asked fact. Correct behaviour: abstain. Naive RAG confidently hallucinates here, because retrieval returns high-scoring on-topic chunks.
> Corpus describes CVE-2024-XXXX's mechanism in detail but never states its CVSS score. Question: *"What is the CVSS score of CVE-2024-XXXX?"*

Also worth including in Bucket 3:
- **Near-miss entity swap:** corpus covers Protocol A; ask about Protocol B, mentioned only in passing.
- **Temporal trap:** corpus states a 2023 figure; ask for the 2024 one.
- **Superlative trap:** *"Which is the most severe vulnerability discussed?"* when the corpus never ranks them.
- **False-premise question:** asks about something the corpus implicitly contradicts.

**Format:**

```jsonl
{"qid":"q001","question":"...","is_answerable":true,"gold_answer":"...","gold_chunk_ids":["S3::c12"],"bucket":"answerable"}
{"qid":"q024","question":"...","is_answerable":false,"gold_answer":null,"gold_chunk_ids":[],"bucket":"adversarial","note":"corpus describes mechanism but never states CVSS score"}
```

The `note` field on adversarial questions is for the README — quoting two or three of these shows the reviewer you engineered the hard cases on purpose.

**How to write them fast:** read a document, write questions from what you see (answerable), then write questions about what is *conspicuously missing* from that same document (adversarial). The adversarial ones are easier to write than they sound, because gaps are visible once you're reading closely.

**Honesty rule:** you wrote both the questions and the system. Say so in the README. Note that a self-authored eval set is a limitation and that an independent set would be stronger. Reviewers reward this; the brief explicitly says stating limitations scores better than hiding them.

---

### PHASE 9 — Evaluation harness (H12 → H14)

**Goal:** `python -m veritas eval` prints a metrics table and writes `eval/results/`.

**Framework or hand-roll?** RAGAS, DeepEval, TruLens, promptfoo all exist. **Hand-roll it.** The formulas below are ~120 lines total, you get full control, zero version-conflict risk at hour 20, and you can explain every line — which the ground rules require ("You must be able to explain every part of your code"). Mention in tradeoffs that you evaluated adopting RAGAS and chose not to, and why. That reads as judgment, not ignorance.

**9a. Citation metrics (ALCE definitions).**

**Citation recall** — did the citations actually support the claim?
```
recall = (# claims whose cited chunks jointly entail the claim) / (# total claims)
```

**Citation precision** — was each citation necessary?
```
For each citation c on claim m:
    c is PRECISE if:
        (a) full cited set entails m, AND
        (b) removing c breaks entailment

precision = (# precise citations) / (# total citations)
```
Condition (b) is the part people skip. Without it, spraying citations everywhere scores perfectly.

**F1** = harmonic mean.

**State this caveat when you report:** these metrics are computed by an NLI model, and the original ALCE work found roughly 78–85% agreement with human judgment on these two measures. Report them as directional, not absolute. Saying this earns more credibility than a confident number would.

**9b. Faithfulness (RAGAS-style).**
```
faithfulness = (# answer claims supported by the retrieved context) / (# total answer claims)
```
Same verifier, but premise = *all* retrieved context rather than just the cited subset. Distinguishes "cited the wrong chunk" from "made it up entirely."

**9c. Abstention metrics — the headline numbers.**

```
                 │ Should answer │ Should abstain
─────────────────┼───────────────┼────────────────
Agent answered   │      TA       │      FA   ← the dangerous cell
Agent abstained  │      FR       │      TR

abstention_accuracy = (TA + TR) / total
false_answer_rate   = FA / (FA + TR)        # answered when it shouldn't have
over_refusal_rate   = FR / (FR + TA)        # refused when it could have answered
```

**False-answer rate is your money metric.** A research agent that confidently answers unanswerable questions is worse than useless. Break it out by bucket — your adversarial-bucket false-answer rate is the number that proves the whole design.

**9d. Risk–coverage (for Phase 10).**

```python
def risk_coverage(scores, correct, thresholds):
    pts = []
    for t in thresholds:
        answered = scores >= t
        coverage = answered.mean()
        risk = 1 - correct[answered].mean() if answered.any() else 0.0
        pts.append((coverage, risk))
    return pts   # AURC = trapezoidal integral, lower is better
```

Also compute **AUROC** of your abstention signal against the binary `is_answerable` label (`sklearn.metrics.roc_auc_score`). AUROC of 0.5 means your signal is noise; above 0.8 means it genuinely discriminates. Report it either way.

**9e. Ablations — cheap, high-value.** Run eval with:

| Configuration | What it isolates |
|---|---|
| dense only, no rerank, no verify | naive RAG baseline |
| + hybrid RRF | retrieval fusion gain |
| + reranker | reranking gain |
| + verifier & abstention | **your contribution** |

A four-row table showing false-answer rate collapsing as you add the verifier is the single most persuasive object in your submission. It is also directly the "approach and model choice" rubric line (25 points).

**Done when:** `eval` runs end-to-end offline and writes a committed results file.

---

### PHASE 10 — Calibration (H14 → H15)

**Goal:** replace guessed thresholds with measured ones.

**Do:**

1. Run the full gold set, recording for every question: max reranker score, supported-claim count, and whether the outcome was correct.
2. Sweep `τ_lo` across the reranker score range; at each value compute coverage and risk.
3. Plot / tabulate the risk–coverage curve.
4. **Pick τ by stating a target first:** *"I target ≤10% false-answer rate; the highest coverage achieving that is τ = 0.42."* Choosing a target and then reading off the threshold is principled. Picking a round number and rationalising it is not.
5. Write τ into `config.yaml`. Re-run eval to confirm the shipped config matches the reported numbers.

**Small-sample honesty.** With 40 questions, a threshold is noisy. Do three things:
- **Bias conservative** — when in doubt, abstain. False answers cost more than over-refusals for a research tool.
- **Leave-one-out** to gauge stability, if you have 20 spare minutes.
- **Say it out loud** in the README: *"τ was calibrated on 43 self-authored questions. It is indicative, not robust; a larger independent set would tighten it."*

This paragraph is worth more than a suspiciously precise threshold. It is the difference between someone who ran a script and someone who understands what the script's output means.

---

### PHASE 11 — Offline reproducibility (H15 → H17)

**Goal:** a reviewer with zero API keys can clone, install, and run everything.

**Why this is worth two hours:** README clarity and reproducibility is 15 points, and "working end-to-end" (30 points) is judged by *the reviewer actually running it*. A brilliant agent that needs a Gemini key the reviewer doesn't have scores near zero on 45 points. The brief says it twice: "Reviewers score what they can actually run. Make setup foolproof."

**Three layers:**

**Layer 1 — Ollama.** Document `ollama pull llama3.1:8b`. Ollama serves an OpenAI-compatible API on `localhost:11434`, so your provider abstraction handles it with a base-URL change. Add a timeout — CPU inference on long prompts is slow and a hung demo reads as broken.

**Layer 2 — Cached fixtures.** Key LLM responses by `sha256(model + prompt)` into `fixtures/cache.json`. Commit it. Now `--offline` replays your exact recorded runs with no model at all. This is what makes `eval` reproducible: the reviewer gets *your* numbers, not their own re-run's numbers.

**Layer 3 — Deterministic fallback.** No LLM available and no cache hit: return the top reranked chunk verbatim as a single self-citing claim. Degraded, honest, always runs.

**The clean-clone test — do this, it catches everything:**
```bash
cd /tmp && git clone <your-repo> && cd veritas-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m veritas ingest
python -m veritas ask "…" --offline
python -m veritas eval --offline
```
Every failure here is a failure the reviewer would have hit. Fix all of them. Budget 30 minutes; it always takes longer than you think.

---

### PHASE 12 — README and tradeoff notes (H17 → H20)

**Goal:** the document that carries 25 of your 100 points (15 README + 10 tradeoffs).

**Structure:**

1. **One-paragraph what-it-does** — including the one sentence from §0.
2. **Quickstart** — copy-pasteable, offline path first, API path second. First code block within the first screen.
3. **Architecture diagram** — the ASCII one from your plan is fine.
4. **Sample runs** — two answered (showing verified citations), one abstained on an unanswerable, **one abstained on an adversarial** (the impressive one). Real output, not idealised.
5. **Results table** — your ablation grid with real numbers.
6. **How it works** — one paragraph per stage: chunking, hybrid retrieval, RRF, reranking, structured generation, NLI verification, abstention gates.
7. **Retrieval / tool approach note** — a required deliverable; make it its own heading so the reviewer finds it.
8. **Tradeoff notes** — the template below.
9. **Limitations** — see §13.
10. **What I'd do with more time.**

**Tradeoff notes template** (fill each with your *measured* number, not a guess):

- **Embedding model** — BGE-small-en-v1.5 over all-MiniLM-L6-v2: chose ___ for [MTEB standing / size / CPU latency measured at ___ ms/query].
- **Retrieval** — hybrid RRF over dense-only: Recall@20 went from ___ to ___ on the gold set. Chose RRF over score normalisation because BM25 and cosine scales are incomparable and rank fusion needs no tuning.
- **Chunking** — 256-token sentence windows over 512: traded ___ recall for citation traceability, because a citation to a 512-token block isn't verifiable by a human in a few seconds.
- **Reranker** — measured ___ change in retrieval quality; kept/dropped because ___. Noted that off-the-shelf rerankers can degrade on out-of-domain corpora, so I measured rather than assumed.
- **Vector store** — NumPy/FAISS-Flat over ChromaDB: corpus is only ___ chunks; exact search is fast enough and a vector DB would be unjustified complexity.
- **Citation granularity** — sentence-level over atomic-claim: claim decomposition introduces over-fragmentation and pronoun-resolution errors; sentence-level is also what production systems (Anthropic Citations) ship.
- **Structured output** — JSON mode + Pydantic validation + retry, *not* grammar-constrained decoding: hard constraints impose a documented accuracy cost by competing with the model's reasoning, and I preferred validation-with-retry.
- **Verifier** — MiniCheck over LLM-as-judge: comparable accuracy at ~___× the cost, runs offline, and independence from the generator is the point.
- **Abstention signal** — reranker score + verified-claim count over cosine similarity: cosine is poorly calibrated on a topically coherent corpus. AUROC of my signal against the answerable label is ___.
- **Threshold** — τ = ___ chosen from the risk–coverage curve targeting ≤___% false-answer rate, calibrated on ___ questions.
- **Evaluation** — hand-rolled metrics over RAGAS: full control, no dependency risk, and I can explain every line as the ground rules require.

Each bullet is a decision, an alternative, and a reason. That is what "clear thinking and honest engineering" looks like on paper.

---

### PHASE 13 — Hardening (H20 → H22)

Work through these explicitly:

| Failure | Handling |
|---|---|
| Empty retrieval (no chunks pass) | Abstain with a clear message, don't crash |
| Malformed LLM JSON | Retry once → regex fallback → deterministic fallback |
| Hallucinated chunk ID | Drop the citation, log it, count it as a metric |
| Question longer than context | Truncate with a warning |
| Corpus document fails to parse | Skip it, log it, continue — never abort ingest |
| API timeout / rate limit | Fall through the provider chain |
| Verifier OOM | Fall back to the smaller NLI model |
| Unicode / encoding garbage in PDFs | Normalise at ingest, don't let it reach the prompt |

**Also do:** run the clean-clone test again. Skim every file once for a leftover `print()`, a hardcoded path, or a committed API key. `git log` should show steady commits across the window, not one dump at hour 23 — the brief explicitly rewards a genuine commit history.

---

### PHASE 14 — Buffer (H22 → H24)

Reserve this. Something will break.

If nothing breaks: record a 2-minute terminal demo (asciinema or a plain screen recording), add it to the README, do a final full eval run, commit the results, and submit early. Submitting at H22 with a working agent beats submitting at H23:59 with an untested last-minute change.

---

## 3. Time budget summary

| Hours | Phase | Cuttable? |
|---|---|---|
| 0–1 | Scaffold, pin deps | No |
| 1–1.5 | Corpus | No |
| 1.5–2.5 | Chunking | Simplify only |
| 2.5–4 | Hybrid retrieval + RRF | Drop BM25 if desperate |
| 4–5 | Reranker | **Yes, first to cut** |
| 5–7 | Structured generation | No |
| 7–9 | **NLI verifier** | **Never** |
| 9–10 | Abstention gates | No |
| 10–12 | Gold question set | Shrink to 25 questions |
| 12–14 | Eval harness | Cut ablations, keep core metrics |
| 14–15 | Calibration | Cut to a conservative guess + say so |
| 15–17 | Offline path | Keep Ollama at minimum |
| 17–20 | README + tradeoffs | **Never** |
| 20–22 | Hardening | Compress to 1h |
| 22–24 | Buffer | — |

**If you are 4 hours behind at H12:** cut the reranker, cut BM25, shrink the gold set to 25, cut ablations. Keep: generation → verification → abstention → eval → README. That minimal path still hits every stated capability and still has the one feature nobody else will have.

---

## 4. Known limitations to state honestly in the README

Being explicit here scores points ("Be honest about limitations. Clearly stating what doesn't work yet scores better than trying to hide it"). Each of these is real:

1. **Self-authored evaluation set.** I wrote both the questions and the system, which risks unconscious bias toward cases my design handles. An independently authored set would be stronger evidence.
2. **NLI verifiers miss multi-hop support.** A claim correctly synthesised across two chunks may register as "neutral" against their concatenation. Mitigated by joint-premise checking; not eliminated.
3. **Citation metrics are model-judged and noisy.** ALCE-style automatic scoring agrees with human judgment roughly 78–85% of the time. My numbers are directional.
4. **Threshold calibrated on a small sample.** τ from ~40 questions carries meaningful variance. Biased conservative deliberately.
5. **No live search.** Deliberately scoped out; the agent is only as good as the provided corpus.
6. **Single-turn only.** No conversation memory, no follow-up resolution.
7. **English only.** Embedding and NLI model choices are English-first.
8. **Extractive bias.** The system is stronger on questions answered by a contiguous span than on ones needing synthesis across many documents — a direct consequence of chunk-level verification.
9. **Over-refusal is unmeasured against user tolerance.** I optimised for low false-answer rate; a real deployment would need to know what refusal rate users actually accept.
10. **Reranker domain mismatch.** The cross-encoder was trained on web-search data, not my corpus's domain. I measured the effect rather than assuming it, but it remains a known risk.

---

## 5. The five things that decide your score

1. **It runs from a clean clone with no API key.** Nothing else matters if this fails.
2. **The verifier visibly catches something.** One screenshot of an unsupported claim being flagged is worth three paragraphs of description.
3. **The adversarial bucket abstains.** This is the capability nobody else will demonstrate.
4. **Real numbers in an ablation table.** Measurement is the difference between a build and research.
5. **Tradeoffs written as decisions with alternatives and reasons**, not as a feature list.

---

## 6. Pre-submission checklist

- [ ] Repo is **public**
- [ ] All commits inside the 24-hour window, spread across it
- [ ] `requirements.txt` exactly pinned; Python version documented
- [ ] Clean-clone test passes on a machine with no keys
- [ ] `data/corpus/` committed
- [ ] `eval/gold.jsonl` committed
- [ ] `eval/results/` committed with real numbers
- [ ] `samples/` has ≥5 answered and ≥3 abstained runs
- [ ] One sample shows the verifier rejecting a claim
- [ ] README quickstart is copy-pasteable, offline path first
- [ ] Retrieval/tool approach note has its own heading (required deliverable)
- [ ] Tradeoff notes have measured numbers, not placeholders
- [ ] Limitations section present and genuinely honest
- [ ] No API keys, no `.env`, no `__pycache__`, no hardcoded paths committed
- [ ] Submitted via HireAI app or public URL **before** the deadline
