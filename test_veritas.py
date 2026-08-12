"""Self-check for the logic that the fixes depend on. Run: `py test_veritas.py`

Deliberately assert-based and dependency-free apart from the package itself. The NLI
check is skipped when the model is not downloaded.
"""

import json
import os
import sys

from veritas.chunking import chunk_document, iter_units
from veritas.config import Config, load_config
from veritas.index import rrf
from veritas.schemas import Chunk, Claim, AgentAnswer, Verdict
from veritas.abstain import check_gate_a, check_gate_b
from eval.metrics import (
    compute_abstention_metrics,
    compute_gold_citation_metrics,
    compute_retrieval_metrics,
)

DOC = """# Title Heading

## Overview
CVE-2024-3094 affects XZ Utils 5.6.0 and 5.6.1. The payload targets `liblzma`, e.g. via
systemd. Andres Freund found it.

## Controls
- **First control:** do the thing.
- **Second control:** do the other thing.

CVSS v3.1 Base Score: 10.0 (Critical).
"""


def test_chunk_offsets_are_exact():
    chunks = chunk_document("S9", "Doc", DOC, target_tokens=20, overlap_sentences=1)
    assert chunks, "chunker produced nothing"
    for c in chunks:
        assert DOC[c.char_start:c.char_end] == c.text, (
            f"{c.chunk_id} offsets do not slice back to its text")
        assert c.char_end > c.char_start


def test_chunker_splits_structure_but_not_versions():
    units = [DOC[s:e] for s, e in iter_units(DOC)]
    assert "## Overview" in units, "markdown headings must be their own unit"
    assert any(u.startswith("- **First control:**") for u in units), "list items must not merge"
    # A version number must not be treated as a sentence boundary.
    assert not any(u.strip() in {"6.0 and 5.", "6.1."} for u in units), "split inside a version"
    assert any("e.g. via" in u or "e.g." in u and "systemd" in u for u in units), \
        "sentence split on the 'e.g.' abbreviation"


def test_every_character_of_a_chunk_comes_from_one_span():
    chunks = chunk_document("S9", "Doc", DOC, target_tokens=1000, overlap_sentences=0)
    assert len(chunks) == 1
    assert chunks[0].text == DOC[chunks[0].char_start:chunks[0].char_end]


def test_rrf_rewards_appearing_in_both_rankings():
    fused = rrf([["a", "b", "c"], ["c", "b", "a"]], k=60)
    assert abs(fused["a"] - fused["c"]) < 1e-12, "symmetric items must tie"

    # An item both retrievers return must outrank one only a single retriever found,
    # even when the single retriever ranked it first.
    fused = rrf([["shared", "x"], ["y", "shared"]], k=60)
    assert list(fused)[0] == "shared", f"consensus item should win, got {list(fused)}"


def test_cache_is_not_keyed_by_question_id():
    """The eval bypass: a qid-keyed lookup lets a hand-written file stand in for the pipeline."""
    from veritas import generate
    import inspect
    assert "qid" not in inspect.signature(generate.get_cached_response).parameters, \
        "get_cached_response must not accept a qid"
    assert "qid" not in inspect.signature(generate.generate_answer).parameters, \
        "generate_answer must not accept a qid"

    if os.path.exists(generate.CACHE_PATH):
        with open(generate.CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        for key in cache:
            assert len(key) == 64 and all(ch in "0123456789abcdef" for ch in key), \
                f"cache key {key!r} is not a prompt hash — qid-keyed fixtures are the bypass"


def test_extractive_fallback_is_labelled():
    from veritas.generate import deterministic_fallback
    chunk = Chunk(chunk_id="S1::c00", doc_id="S1", doc_title="T", text="Some text.",
                  char_start=0, char_end=10, sent_range=(0, 0))
    answer = deterministic_fallback([(chunk, 0.9)])
    assert answer.generator == "extractive-fallback", \
        "fallback answers must be distinguishable from model output"


def _probe_chunk(text="The CVSS v3.1 Base Score is 10.0 (Critical).", cid="S1::c00"):
    return Chunk(chunk_id=cid, doc_id=cid.split("::")[0], doc_title="T", text=text,
                 char_start=0, char_end=len(text), sent_range=(0, 0))


def test_quote_grounding_keeps_real_spans_and_drops_invented_ones():
    from veritas.generate import check_quote_grounding
    chunk_map = {"S1::c00": _probe_chunk()}

    answer = AgentAnswer(claims=[
        Claim(text="The score is 10.0.", citations=["S1::c00"],
              quote="CVSS v3.1 Base Score is 10.0"),
        Claim(text="The score is 7.5.", citations=["S1::c00"],
              quote="CVSS v3.1 Base Score is 7.5"),
    ])
    answer, ungrounded = check_quote_grounding(answer, chunk_map)
    assert [c.text for c in answer.claims] == ["The score is 10.0."]
    assert ungrounded == ["CVSS v3.1 Base Score is 7.5"]


def test_quote_grounding_survives_line_wrapping():
    from veritas.generate import check_quote_grounding
    chunk_map = {"S1::c00": _probe_chunk("Automatically disables inactive\naccounts after 90 days.")}
    answer = AgentAnswer(claims=[Claim(text="90 days.", citations=["S1::c00"],
                                       quote="disables inactive accounts after 90 days")])
    answer, ungrounded = check_quote_grounding(answer, chunk_map)
    assert not ungrounded, "whitespace differences must not fail a real quote"
    assert len(answer.claims) == 1


def test_quote_grounding_survives_markdown_emphasis():
    """Measured: gpt-oss-120b quotes the rendered text of a bolded list item, dropping the
    asterisks. Two of four over-refusals in the clean benchmark run were this."""
    from veritas.generate import check_quote_grounding
    source = ("1. **Zero Raw Pointers in Public API:** All reference passing must use safe "
              "wrappers (`SafeRef<T>` or `UniquePtr<T>`).")
    chunk_map = {"S7::c01": _probe_chunk(source, "S7::c01")}
    answer = AgentAnswer(claims=[Claim(
        text="SwiftSafe requires safe wrappers instead of raw pointers.", citations=["S7::c01"],
        quote="Zero Raw Pointers in Public API: All reference passing must use safe wrappers "
              "(SafeRef<T> or UniquePtr<T>).")])
    answer, ungrounded = check_quote_grounding(answer, chunk_map)
    assert not ungrounded, "formatting markers must not fail a faithful quote"
    assert len(answer.claims) == 1

    # Stripping markup must not let an invented span through.
    invented = AgentAnswer(claims=[Claim(text="x", citations=["S7::c01"],
                                         quote="Raw pointers are permitted in the public API.")])
    invented, ungrounded = check_quote_grounding(invented, chunk_map)
    assert ungrounded and not invented.claims


def test_premise_window_still_resolves_a_markdown_stripped_quote():
    """If the two normalisers disagree, a quote clears grounding and then silently loses its
    window, widening the premise back to every cited chunk."""
    from veritas.verify import build_premise
    cfg = Config()
    cfg.verify.premise_window_chars = 30
    source = ("Filler sentence about unrelated matters. " * 3
              + "1. **1-RTT Handshake:** Reduces connection handshake latency from 2-RTT to "
                "1-RTT for full handshakes. " + "More filler about ciphers. " * 3)
    chunk = Chunk(chunk_id="S8::c00", doc_id="S8", doc_title="T", text=source,
                  char_start=0, char_end=len(source), sent_range=(0, 0))
    claim = Claim(text="x", citations=["S8::c00"],
                  quote="1-RTT Handshake: Reduces connection handshake latency from 2-RTT to 1-RTT")
    premise = build_premise(claim, [chunk], cfg)
    assert "2-RTT to 1-RTT" in premise
    assert len(premise) < len(source), "the window did not resolve; premise widened to the chunk"


def test_quote_grounding_rejects_quote_from_an_uncited_chunk():
    """Quoting real corpus text but citing the wrong chunk is still a bad citation."""
    from veritas.generate import check_quote_grounding
    chunk_map = {"S1::c00": _probe_chunk(),
                 "S2::c00": _probe_chunk("Use parameterized queries.", "S2::c00")}
    answer = AgentAnswer(claims=[Claim(text="Use prepared statements.", citations=["S1::c00"],
                                       quote="Use parameterized queries")])
    answer, ungrounded = check_quote_grounding(answer, chunk_map)
    assert ungrounded and not answer.claims


def test_claims_without_a_quote_are_left_for_the_nli_verifier():
    from veritas.generate import check_quote_grounding
    answer = AgentAnswer(claims=[Claim(text="Something.", citations=["S1::c00"])])
    answer, ungrounded = check_quote_grounding(answer, {"S1::c00": _probe_chunk()})
    assert len(answer.claims) == 1 and not ungrounded


def test_env_file_is_gitignored():
    """A .env holding a live API key must never be committable."""
    if not os.path.exists(".env"):
        print("  SKIP env check (no .env)")
        return
    assert os.path.exists(".gitignore"), ".env exists but there is no .gitignore"
    with open(".gitignore", encoding="utf-8") as f:
        entries = {line.strip() for line in f}
    assert ".env" in entries, ".env must be listed in .gitignore"


def test_real_environment_wins_over_the_env_file():
    """An exported key must not be silently replaced by a stale one in .env."""
    import inspect
    from veritas import __file__ as pkg
    source = inspect.getsource(sys.modules["veritas"])
    if "load_dotenv" in source:
        assert "override=False" in source, "load_dotenv must not clobber exported variables"


def test_premise_narrows_to_the_quote_window():
    """A long multi-topic chunk drowns the claim; the window keeps the quote in context."""
    from veritas.verify import build_premise
    cfg = Config()
    cfg.verify.premise_window_chars = 40
    long_text = ("Unrelated preamble about command injection and shell functions. " * 4
                 + "Use parameterized queries for all database access. "
                 + "Further unrelated material about access control. " * 4)
    chunk = Chunk(chunk_id="S2::c01", doc_id="S2", doc_title="T", text=long_text,
                  char_start=0, char_end=len(long_text), sent_range=(0, 0))

    claim = Claim(text="x", citations=["S2::c01"],
                  quote="Use parameterized queries for all database access.")
    premise = build_premise(claim, [chunk], cfg)
    assert "parameterized queries" in premise
    assert len(premise) < len(long_text), "premise must be narrowed"
    assert len(premise) > len(claim.quote), "bare quote over-accepts; keep context"


def test_premise_falls_back_to_full_text_without_a_usable_quote():
    from veritas.verify import build_premise
    cfg = Config()
    chunk = Chunk(chunk_id="S2::c01", doc_id="S2", doc_title="T", text="Alpha beta gamma.",
                  char_start=0, char_end=17, sent_range=(0, 0))
    assert build_premise(Claim(text="x", citations=["S2::c01"]), [chunk], cfg).endswith(
        "Alpha beta gamma.")
    invented = Claim(text="x", citations=["S2::c01"], quote="not in the chunk at all")
    assert build_premise(invented, [chunk], cfg).endswith("Alpha beta gamma.")


def test_premise_is_headed_by_the_document_title():
    """Without the title the chunk says "The vulnerability ..." and the identifier the
    claim binds to is nowhere in the premise — measured 0.002 vs 0.998 with it."""
    from veritas.verify import build_premise
    cfg = Config()
    body = "The vulnerability was found in March. CVSS v3.1 Base Score: 10.0 (Critical)."
    chunk = Chunk(chunk_id="S1::c03", doc_id="S1",
                  doc_title="Technical Analysis of CVE-2024-3094: XZ Utils Backdoor",
                  text=body, char_start=0, char_end=len(body), sent_range=(0, 0))
    claim = Claim(text="x", citations=["S1::c03"], quote="CVSS v3.1 Base Score: 10.0")
    premise = build_premise(claim, [chunk], cfg)
    assert "CVE-2024-3094" in premise, "the identifier must reach the premise"
    assert "CVSS v3.1 Base Score: 10.0" in premise


def test_premise_names_no_document_when_citations_span_several():
    """Naming one document over a premise assembled from two would attribute text to the
    wrong source."""
    from veritas.verify import build_premise
    cfg = Config()
    a = Chunk(chunk_id="S1::c00", doc_id="S1", doc_title="Doc One", text="Alpha.",
              char_start=0, char_end=6, sent_range=(0, 0))
    b = Chunk(chunk_id="S2::c00", doc_id="S2", doc_title="Doc Two", text="Beta.",
              char_start=0, char_end=5, sent_range=(0, 0))
    premise = build_premise(Claim(text="x", citations=["S1::c00", "S2::c00"]), [a, b], cfg)
    assert "Doc One" not in premise and "Doc Two" not in premise


def test_document_title_comes_from_the_markdown_heading():
    from veritas.ingest import extract_title
    body = "# Technical Analysis of CVE-2024-3094: XZ Utils Backdoor\n\n## Overview\nText."
    assert extract_title(body, "S1_cve_2024_3094.md").startswith("Technical Analysis of CVE-2024-3094")
    assert extract_title("No heading here.", "S1_cve_2024_3094.md") == "S1 Cve 2024 3094"


def test_prompt_requires_declarative_claims():
    """Measured: identical content scores 0.02 as an imperative and 0.99 as a declarative.
    The prompt must carry the hard syntactic rule, not a soft preference."""
    from veritas.generate import build_generation_prompt
    prompt = build_generation_prompt("q?", [(_probe_chunk(), 0.9)])
    assert "declarative" in prompt.lower()
    assert "never begin with a verb" in prompt.lower(), \
        "a soft 'write declaratively' hint was not enough — llama-3.3-70b ignored it"
    assert "begins with a verb" in prompt.lower(), "keep the worked negative examples"


def test_groq_is_a_registered_provider():
    from veritas.generate import _PROVIDERS
    from veritas.config import Config
    assert "groq" in _PROVIDERS
    assert "groq" in Config().llm.providers, "groq must be in the default provider cascade"


def test_prompt_demands_a_verbatim_quote():
    from veritas.generate import build_generation_prompt
    prompt = build_generation_prompt("q?", [(_probe_chunk(), 0.9)])
    assert "quote" in prompt and "verbatim" in prompt.lower()
    assert "insufficient_evidence" in prompt


def test_gate_a_thresholds_on_score_not_rank():
    cfg = Config()
    cfg.abstain.min_rerank_score = 0.35
    chunk = Chunk(chunk_id="S1::c00", doc_id="S1", doc_title="T", text="x" * 200,
                  char_start=0, char_end=200, sent_range=(0, 0))
    assert check_gate_a([(chunk, 0.9)], cfg)[0] is False
    abstain, reason = check_gate_a([(chunk, 0.2)], cfg)
    assert abstain and "0.35" in reason


def test_gate_b_reports_the_real_count():
    cfg = Config()
    cfg.abstain.min_supported_claims = 2
    answer = AgentAnswer(claims=[Claim(text="a"), Claim(text="b")])
    verdicts = [Verdict(claim_text="a", citations=["S1::c00"], supported=True, score=0.9),
                Verdict(claim_text="b", citations=["S1::c00"], supported=False, score=0.1)]
    abstain, reason = check_gate_b(answer, verdicts, [], cfg)
    assert abstain
    assert "1 of 2" in reason, f"gate B must report the real count, got: {reason}"


def test_abstention_metrics_confusion_matrix():
    stats = compute_abstention_metrics([
        {"is_answerable": True, "abstained": False},   # TA
        {"is_answerable": True, "abstained": True},    # FR
        {"is_answerable": False, "abstained": False},  # FA
        {"is_answerable": False, "abstained": True},   # TR
    ])
    assert (stats["ta"], stats["fr"], stats["fa"], stats["tr"]) == (1, 1, 1, 1)
    assert stats["false_answer_rate"] == 0.5 and stats["over_refusal_rate"] == 0.5


def test_citation_recall_counts_refusals_as_misses():
    """The inflation bug: dropping unanswered questions lets an agent buy recall by refusing."""
    records = [
        {"is_answerable": True, "gold_chunk_ids": ["S1::c00"], "predicted_chunk_ids": ["S1::c00"]},
        {"is_answerable": True, "gold_chunk_ids": ["S2::c00"], "predicted_chunk_ids": []},
    ]
    stats = compute_gold_citation_metrics(records)
    assert stats["recall"] == 0.5, f"expected 0.5, got {stats['recall']}"
    assert stats["precision"] == 1.0
    assert stats["questions_scored"] == 2


def test_citation_precision_penalises_extra_citations():
    stats = compute_gold_citation_metrics([{
        "is_answerable": True,
        "gold_chunk_ids": ["S1::c00"],
        "predicted_chunk_ids": ["S1::c00", "S4::c02", "S7::c01"],
    }])
    assert abs(stats["precision"] - 1 / 3) < 1e-3  # metrics are rounded to 4dp
    assert stats["recall"] == 1.0


def test_retrieval_metrics_use_rank_position():
    stats = compute_retrieval_metrics([{
        "is_answerable": True,
        "gold_chunk_ids": ["S1::c01"],
        "context_injected": ["S1::c00", "S1::c01", "S2::c00"],
    }])
    assert stats["recall_at_final_k"] == 1.0
    assert abs(stats["mrr"] - 0.5) < 1e-9


def test_nli_scores_entailment_above_contradiction():
    cfg = load_config("config.yaml") if os.path.exists("config.yaml") else Config()
    from veritas.verify import get_nli_scorer, compute_entailment_score
    try:
        scorer, backend = get_nli_scorer(cfg)
    except RuntimeError as e:
        print(f"  SKIP nli check ({str(e).splitlines()[0]})")
        return

    premise = "CVE-2024-3094 was discovered by Andres Freund while benchmarking SSH latencies."
    entailed = compute_entailment_score(premise, "Andres Freund discovered CVE-2024-3094.", cfg)
    contradicted = compute_entailment_score(premise, "The moon is made of cheese.", cfg)

    assert 0.0 <= entailed <= 1.0 and 0.0 <= contradicted <= 1.0, "scores must be probabilities"
    assert entailed > contradicted, f"entailed {entailed} should beat unrelated {contradicted}"
    assert backend == cfg.verify.model, \
        f"verifier must use the configured model, got {backend!r}"
    # The old bug returned an exact word-overlap fraction; a real NLI score does not.
    assert entailed > 0.5, f"real entailment should clear the support threshold, got {entailed}"


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_gold_dataset_references_real_chunks():
    if not os.path.exists("data/index/chunks.json"):
        print("  SKIP gold check (corpus not ingested)")
        return
    with open("data/index/chunks.json", encoding="utf-8") as f:
        ids = {c["chunk_id"] for c in json.load(f)}
    for path in ("eval/gold.jsonl", "eval/holdout.jsonl"):
        if not os.path.exists(path):
            continue
        for item in _read_jsonl(path):
            for cid in item["gold_chunk_ids"]:
                assert cid in ids, f"{item['qid']} cites nonexistent chunk {cid}"
            if item["is_answerable"]:
                assert item["gold_chunk_ids"], f"{item['qid']} is answerable but has no gold chunk"
                assert item.get("gold_answer"), f"{item['qid']} is answerable but has no gold answer"
            else:
                assert not item["gold_chunk_ids"], f"{item['qid']} is unanswerable but cites chunks"


def test_holdout_is_disjoint_from_the_tuning_set():
    """The holdout only means anything if nothing in it was seen while tuning."""
    if not os.path.exists("eval/holdout.jsonl"):
        print("  SKIP holdout check (no holdout set)")
        return
    gold = _read_jsonl("eval/gold.jsonl")
    holdout = _read_jsonl("eval/holdout.jsonl")

    shared_ids = {i["qid"] for i in gold} & {i["qid"] for i in holdout}
    assert not shared_ids, f"qid collision between the two sets: {sorted(shared_ids)}"
    shared_q = ({i["question"].strip().lower() for i in gold}
                & {i["question"].strip().lower() for i in holdout})
    assert not shared_q, f"question reused across sets: {sorted(shared_q)}"
    assert all(i.get("split") == "holdout" for i in holdout), \
        "every holdout row needs split=holdout — that field is what keeps calibration off it"
    assert not any(i.get("split") == "holdout" for i in gold)


def test_calibration_refuses_to_tune_on_holdout_questions():
    """Tuning a threshold on the out-of-sample set silently destroys what it measures."""
    import tempfile
    from eval.calibrate import collect_signals
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False,
                                     encoding="utf-8") as f:
        f.write(json.dumps({"qid": "h01", "question": "q?", "is_answerable": True,
                            "split": "holdout"}) + "\n")
        path = f.name
    try:
        # Raises before any model loads, so this stays a cheap check.
        collect_signals(Config(), path)
        raise AssertionError("calibration accepted a holdout-only question set")
    except SystemExit as e:
        assert "holdout" in str(e)
    finally:
        os.unlink(path)


def test_verifier_bench_quotes_are_verbatim_in_their_cited_chunks():
    """A mistyped bench quote changes which premise the candidate models see, so the
    comparison would quietly stop measuring what it claims to."""
    if not (os.path.exists("eval/verifier_bench.jsonl")
            and os.path.exists("data/index/chunks.json")):
        print("  SKIP verifier bench check (corpus not ingested)")
        return
    with open("data/index/chunks.json", encoding="utf-8") as f:
        text_by_id = {c["chunk_id"]: " ".join(c["text"].split()).lower() for c in json.load(f)}

    for item in _read_jsonl("eval/verifier_bench.jsonl"):
        haystack = " ".join(text_by_id[cid] for cid in item["citations"])
        needle = " ".join(item["quote"].split()).lower()
        assert needle in haystack, f"{item['vid']}: quote is not verbatim in its cited chunk"


def _http_429(retry_after=None, body=""):
    import requests
    response = requests.Response()
    response.status_code = 429
    response._content = body.encode("utf-8")
    if retry_after is not None:
        response.headers["retry-after"] = str(retry_after)
    return requests.HTTPError("429", response=response)


def test_a_long_burst_limit_is_waited_out_not_mistaken_for_a_daily_quota():
    """The wait length does not tell you which limit fired. A 140s per-minute window cost a
    whole benchmark run when this was decided by magnitude alone."""
    from veritas.generate import _call_with_backoff
    calls = []
    tpm = ("Rate limit reached for model `openai/gpt-oss-120b` ... on tokens per minute "
           "(TPM): Limit 8000, Used 7999. Please try again in 140s.")

    def flaky(prompt, cfg, valid_ids):
        calls.append(1)
        if len(calls) == 1:
            raise _http_429(retry_after=0, body=tpm)
        return AgentAnswer(claims=[Claim(text="ok")], generator="groq:test")

    answer = _call_with_backoff(flaky, "groq", "p", Config(), set())
    assert answer is not None and len(calls) == 2, "a per-minute limit must be waited out"


def test_a_declared_daily_limit_aborts_without_waiting():
    from veritas.generate import _call_with_backoff, QuotaExhausted
    tpd = ("Rate limit reached for model `llama-3.3-70b-versatile` ... on tokens per day "
           "(TPD): Limit 100000, Used 99961. Please try again in 1.7s.")

    def spent(prompt, cfg, valid_ids):
        # Short retry-after: magnitude would call this transient, the body says otherwise.
        raise _http_429(retry_after=2, body=tpd)

    try:
        _call_with_backoff(spent, "groq", "p", Config(), set())
        raise AssertionError("a declared per-day limit must not be waited out")
    except QuotaExhausted as e:
        assert "per-day" in str(e)


def test_spent_quota_aborts_instead_of_degrading_to_extractive():
    """A daily-quota 429 used to fall through to the extractive fallback, quietly filling a
    benchmark with answers no model produced."""
    from veritas.generate import _call_with_backoff, QuotaExhausted

    def always_429(prompt, cfg, valid_ids):
        raise _http_429(retry_after=3600)

    try:
        _call_with_backoff(always_429, "gemini", "p", Config(), set())
        raise AssertionError("a 3600s retry-after must not be waited out")
    except QuotaExhausted as e:
        assert "3600" in str(e) and "--offline" in str(e), f"unhelpful message: {e}"

    # And one that never clears, without a retry-after header to go on.
    def always_429_bare(prompt, cfg, valid_ids):
        raise _http_429()

    try:
        _call_with_backoff(always_429_bare, "groq", "p", Config(), set(), max_waits=0)
        raise AssertionError("an unrecoverable 429 must not return None")
    except QuotaExhausted:
        pass


def test_transient_rate_limit_is_still_waited_out():
    """The fix must not turn a per-minute burst limit into an aborted run."""
    from veritas.generate import _call_with_backoff
    calls = []

    def flaky(prompt, cfg, valid_ids):
        calls.append(1)
        if len(calls) == 1:
            raise _http_429(retry_after=0)
        return AgentAnswer(claims=[Claim(text="ok")], generator="groq:test")

    answer = _call_with_backoff(flaky, "groq", "p", Config(), set())
    assert answer is not None and len(calls) == 2, "a short 429 must be retried, not aborted"


def _patched_providers(**fakes):
    """Swaps provider callables in and restores them, `_SPENT` included."""
    from veritas import generate
    import contextlib

    @contextlib.contextmanager
    def ctx():
        original, spent = generate._PROVIDERS.copy(), generate._SPENT.copy()
        generate._PROVIDERS.update(fakes)
        try:
            yield generate
        finally:
            generate._PROVIDERS.clear()
            generate._PROVIDERS.update(original)
            generate._SPENT.clear()
            generate._SPENT.update(spent)
    return ctx()


def _spent_provider(prompt, cfg_, valid_ids):
    raise _http_429(retry_after=86400)


def test_quota_abort_is_not_swallowed_by_provider_failover():
    from veritas.generate import QuotaExhausted
    cfg = Config()
    cfg.llm.providers = ["groq", "gemini", "offline"]

    with _patched_providers(groq=_spent_provider, gemini=_spent_provider) as generate:
        try:
            generate.generate_answer("q?", [(_probe_chunk(), 0.9)], cfg, offline=False)
            raise AssertionError("failover swallowed the quota error and degraded the answer")
        except QuotaExhausted:
            pass


def test_a_spent_provider_hands_over_to_a_working_one():
    """Aborting while another real model is available would be its own kind of wrong."""
    cfg = Config()
    cfg.llm.providers = ["gemini", "groq"]

    def working(prompt, cfg_, valid_ids):
        return AgentAnswer(claims=[Claim(text="ok")], generator="groq:test")

    with _patched_providers(gemini=_spent_provider, groq=working) as generate:
        answer = generate.generate_answer("q?", [(_probe_chunk(), 0.9)], cfg, offline=False)
        assert answer.generator == "groq:test", "a live provider must still be reached"
        assert "gemini" in generate._SPENT, "a spent quota must not be re-asked every question"


def test_spent_quota_does_not_become_a_cache_replay():
    """A cached answer is a real past generation, but substituting it mid-run still reports
    numbers the current configuration did not produce."""
    from veritas.generate import QuotaExhausted
    cfg = Config()
    cfg.llm.providers = ["gemini"]

    with _patched_providers(gemini=_spent_provider) as generate:
        try:
            generate.generate_answer("q?", [(_probe_chunk(), 0.9)], cfg, offline=False)
            raise AssertionError("a spent quota fell through to cache or extractive output")
        except QuotaExhausted:
            pass


def test_grading_sheet_never_discards_a_human_grade():
    import tempfile
    from eval.grade import emit
    with tempfile.TemporaryDirectory() as tmp:
        results = os.path.join(tmp, "results.json")
        sheet = os.path.join(tmp, "grades.jsonl")
        record = {"qid": "q001", "bucket": "answerable", "is_answerable": True,
                  "abstained": False, "gold_answer": "10.0 Critical",
                  "answer_text": "The score is 10.0.", "gold_chunk_ids": [],
                  "predicted_chunk_ids": [], "context_injected": []}
        with open(results, "w", encoding="utf-8") as f:
            json.dump({"variants": {"veritas_full": {"records": [record]}}}, f)

        emit(results_path=results, out_path=sheet, gold_paths=[])
        rows = _read_jsonl(sheet)
        assert len(rows) == 1 and rows[0]["grade"] is None

        rows[0]["grade"] = "correct"
        with open(sheet, "w", encoding="utf-8") as f:
            f.write(json.dumps(rows[0]) + "\n")

        emit(results_path=results, out_path=sheet, gold_paths=[])  # re-run after a new eval
        assert _read_jsonl(sheet)[0]["grade"] == "correct", "re-emitting wiped a human grade"

        # But a grade belongs to the answer it was given for. A new run that produces a
        # different answer must not inherit the old verdict.
        record["answer_text"] = "The score is 7.5, per the advisory."
        with open(results, "w", encoding="utf-8") as f:
            json.dump({"variants": {"veritas_full": {"records": [record]}}}, f)
        emit(results_path=results, out_path=sheet, gold_paths=[])
        assert _read_jsonl(sheet)[0]["grade"] is None, \
            "a grade rode along onto a different answer"


def test_token_recall_proxy_is_measured_against_grades_not_assumed():
    from eval.metrics import token_recall
    gold = "10.0 Critical"
    assert token_recall(gold, "The CVSS base score is 10.0 (Critical).") == 1.0
    # The failure mode the human grades exist to expose: shared vocabulary, wrong answer.
    assert token_recall("Dormant for up to 14 days",
                        "The malware stays dormant for up to 30 days.") > 0.5


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {test.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
