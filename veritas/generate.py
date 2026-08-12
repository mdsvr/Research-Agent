import os
import json
import hashlib
import re
import time
import requests
from typing import List, Tuple, Optional, Dict, Any

from veritas.config import Config
from veritas.schemas import Chunk, Claim, AgentAnswer

# Repo-root-relative so the cache resolves identically from any working directory.
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "fixtures", "cache.json")


def compute_prompt_hash(model_name: str, prompt: str) -> str:
    """SHA-256 cache key over the exact model and the exact prompt."""
    return hashlib.sha256(f"{model_name}::{prompt}".encode("utf-8")).hexdigest()


def get_cached_response(prompt_hash: str) -> Optional[AgentAnswer]:
    """Replays a previous generation for an identical model+prompt.

    Keyed on the prompt hash only. It is deliberately *not* keyed on question id:
    doing so would let a hand-written answer file stand in for the pipeline and
    silently fake evaluation results.
    """
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"Warning: could not read generation cache: {e}")
        return None

    entry = cache.get(prompt_hash)
    if entry is None:
        return None
    answer = AgentAnswer(**entry)
    answer.generator = "cache"
    return answer


def save_to_cache(prompt_hash: str, answer: AgentAnswer):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache: Dict[str, Any] = {}
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except (OSError, json.JSONDecodeError):
            cache = {}
    cache[prompt_hash] = answer.model_dump()
    tmp_path = CACHE_PATH + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        os.replace(tmp_path, CACHE_PATH)  # atomic; a crashed write cannot truncate the cache
    except OSError as e:
        print(f"Warning: could not write generation cache: {e}")


def build_generation_prompt(query: str, scored_chunks: List[Tuple[Chunk, float]]) -> str:
    """Builds the citation-minimising prompt with explicit source chunk tags.

    The hard part is not answering — it is refusing when the sources are *about* the right
    topic but do not contain the specific fact asked for. That is the dominant failure mode
    measured on the adversarial bucket, so the prompt spends most of its budget on it.
    """
    context_str = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk, _ in scored_chunks)

    return f"""You are a research assistant. Answer the QUESTION using ONLY the SOURCES below.

Before answering, check whether the SOURCES actually state the specific thing asked for.
Being about the right topic is not enough. If the question asks for a number, an
identifier, a version, a score or a name, that exact item must appear in the SOURCES.

Rules:
1. A CLAIM MUST NEVER BEGIN WITH A VERB. Every claim is a declarative sentence with a
   subject, describing what is the case. This holds even when the source is written as
   instructions — restate an instruction as a fact about what is required or recommended.
     good: "Parameterized queries are recommended for all database access."
     good: "Inactive accounts are automatically disabled after 90 days."
     bad:  "Use parameterized queries."              (begins with a verb)
     bad:  "Disable inactive accounts after 90 days." (begins with a verb)
     bad:  "The sources recommend parameterized queries."  (about the sources, not the world)
2. Every claim must be supported by a verbatim span of a cited source.
3. For each claim, put that supporting span in "quote", copied character-for-character
   from the source. The quote may be phrased as an instruction even though the claim is
   not — the claim restates it, the quote reproduces it exactly.
4. Cite using the exact source IDs shown, e.g. ["S3::c12"]. Cite the MINIMUM needed.
5. If the SOURCES do not state what was asked, set "insufficient_evidence" to true and
   return an empty claims list. This is a correct answer, not a failure.
6. Never use outside knowledge. Never infer a value that is not written down. Never
   substitute a related fact for the one requested.

Worked example of rule 5 — sources describe the Heartbleed bug in detail, question asks
"What CVSS score was assigned to Heartbleed?", and no CVSS score appears in the text:
{{"insufficient_evidence": true, "claims": [],
  "reasoning": "Sources cover the Heartbleed mechanism and fix but state no CVSS score."}}

SOURCES:
{context_str}

QUESTION: {query}

Return JSON matching this schema:
{{
  "insufficient_evidence": boolean,
  "claims": [
    {{
      "text": "claim text",
      "citations": ["chunk_id"],
      "quote": "verbatim supporting span copied from the cited source"
    }}
  ],
  "reasoning": "brief rationale"
}}
"""


def normalise_span(text: str) -> str:
    """Whitespace-collapsed, markdown-stripped form of a span. Case preserved.

    Emphasis markers are dropped for the same reason line breaks are: a model copying from
    `- **Zero Raw Pointers in Public API:** All reference passing must use safe wrappers`
    reproduces what the source *says*, not the asterisks around it. Measured on the clean
    benchmark run, that alone caused two of four over-refusals — true claims, real quotes,
    rejected on formatting. Dropping `*` and backticks cannot let a fabrication through:
    the words still have to appear, in order, in a chunk the claim cites.
    """
    return " ".join(text.replace("*", "").replace("`", "").split())


def check_quote_grounding(
    answer: AgentAnswer, chunk_map: Dict[str, Chunk]
) -> Tuple[AgentAnswer, List[str]]:
    """Drops claims whose supporting quote is not literally in one of their cited chunks.

    A deterministic check, and the only one in the pipeline a model cannot argue with: the
    span is either in the corpus or it is not. Claims with no quote are left alone — the
    NLI verifier still has to clear them.
    """
    kept, ungrounded = [], []
    for claim in answer.claims:
        if not claim.quote:
            kept.append(claim)
            continue
        haystack = " ".join(normalise_span(chunk_map[cid].text).lower()
                            for cid in claim.citations if cid in chunk_map)
        if normalise_span(claim.quote).lower() in haystack:
            kept.append(claim)
        else:
            ungrounded.append(claim.quote)

    answer.claims = kept
    return answer, ungrounded


def extract_json_from_response(raw_text: str) -> Optional[dict]:
    """Robust JSON extraction from markdown codeblocks or raw strings."""
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
    if match:
        raw_text = match.group(1)
    else:
        match_raw = re.search(r'(\{.*\})', raw_text, re.DOTALL)
        if match_raw:
            raw_text = match_raw.group(1)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def _parse_answer(text_out: str, valid_chunk_ids, generator: str) -> Optional[AgentAnswer]:
    parsed = extract_json_from_response(text_out)
    if not parsed:
        return None
    answer = AgentAnswer(**parsed)
    answer.generator = generator
    for claim in answer.claims:
        # Drop citations to chunk ids that were never in the context window.
        claim.citations = [cid for cid in claim.citations if cid in valid_chunk_ids]
    return answer


def _call_groq(prompt: str, cfg: Config, valid_ids) -> Optional[AgentAnswer]:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        return None
    res = requests.post("https://api.groq.com/openai/v1/chat/completions", timeout=60,
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": cfg.llm.groq_model,
                              "messages": [{"role": "user", "content": prompt}],
                              "temperature": cfg.llm.temperature,
                              "response_format": {"type": "json_object"}})
    res.raise_for_status()
    text_out = res.json()["choices"][0]["message"]["content"]
    return _parse_answer(text_out, valid_ids, f"groq:{cfg.llm.groq_model}")


def _call_gemini(prompt: str, cfg: Config, valid_ids) -> Optional[AgentAnswer]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    # Key goes in a header, never the query string. requests puts the full URL into every
    # HTTPError message, so `?key=...` leaks the secret verbatim into logs, tracebacks and
    # background-task output on any failure — observed with a real key during a 429.
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{cfg.llm.gemini_model}:generateContent")
    res = requests.post(url, timeout=30, headers={"x-goog-api-key": key}, json={
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": cfg.llm.temperature,
                             "responseMimeType": "application/json"},
    })
    res.raise_for_status()
    text_out = res.json()["candidates"][0]["content"]["parts"][0]["text"]
    return _parse_answer(text_out, valid_ids, f"gemini:{cfg.llm.gemini_model}")


def _call_openrouter(prompt: str, cfg: Config, valid_ids) -> Optional[AgentAnswer]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    res = requests.post("https://openrouter.ai/api/v1/chat/completions", timeout=30,
                        headers={"Authorization": f"Bearer {key}"},
                        json={"model": cfg.llm.openrouter_model,
                              "messages": [{"role": "user", "content": prompt}],
                              "temperature": cfg.llm.temperature,
                              "response_format": {"type": "json_object"}})
    res.raise_for_status()
    text_out = res.json()["choices"][0]["message"]["content"]
    return _parse_answer(text_out, valid_ids, f"openrouter:{cfg.llm.openrouter_model}")


def _call_ollama(prompt: str, cfg: Config, valid_ids) -> Optional[AgentAnswer]:
    res = requests.post("http://localhost:11434/api/generate", timeout=120,
                        json={"model": cfg.llm.ollama_model, "prompt": prompt,
                              "stream": False, "format": "json",
                              "options": {"temperature": cfg.llm.temperature}})
    res.raise_for_status()
    return _parse_answer(res.json().get("response", ""), valid_ids, f"ollama:{cfg.llm.ollama_model}")


_PROVIDERS = {"groq": _call_groq, "gemini": _call_gemini,
              "openrouter": _call_openrouter, "ollama": _call_ollama}

# Providers found to be out of quota, for the lifetime of the process.
_SPENT: set = set()


class QuotaExhausted(RuntimeError):
    """A provider is rate-limited beyond what waiting will fix — a spent daily quota.

    Distinct from a transient 429 because the remedy is different: a per-minute limit is
    waited out, a daily one has to abort the run. Never caught by the provider-failover
    path, so an exhausted quota cannot quietly become a benchmark full of extractive
    fallbacks.
    """


_DAILY_QUOTA = re.compile(r"per\s+day|\bTPD\b|\bRPD\b|daily\s+(?:limit|quota)", re.I)


def _names_a_daily_limit(response) -> bool:
    """True when the provider says the 429 came from a per-day budget.

    The length of the wait does not distinguish daily from burst, which is what this code
    used to assume. Groq's per-day budget is a rolling window and can ask for 2 seconds
    when it is nearly spent, while an ordinary per-minute limit asked for 140 and got a
    whole benchmark run aborted for it. Only the body says which limit was hit.
    """
    try:
        return bool(_DAILY_QUOTA.search(response.text[:2000]))
    except Exception:
        return False  # unreadable body: fall through to the wait-and-see path


def _call_with_backoff(call, name: str, prompt: str, cfg: Config, valid_ids,
                       max_waits: int = 5, max_delay: float = 300.0) -> Optional[AgentAnswer]:
    """Calls a provider, waiting out HTTP 429 rate limits rather than falling through.

    Free tiers rate-limit well within a 120-question evaluation run. Treating a 429 as a
    provider failure would silently substitute extractive-fallback answers into the
    middle of a benchmark, which is exactly the kind of quiet substitution this codebase
    is meant not to do.

    Two 429s that waiting will not fix raise `QuotaExhausted` instead: one whose
    `retry-after` is longer than `max_delay` (the server is telling us the window is a
    daily one), and one that survives the whole backoff.
    """
    for wait_number in range(max_waits + 1):
        try:
            return call(prompt, cfg, valid_ids)
        except requests.HTTPError as e:
            if e.response is None or e.response.status_code != 429:
                raise
            if _names_a_daily_limit(e.response):
                raise QuotaExhausted(
                    f"provider '{name}' reports a per-day limit (HTTP 429). Waiting cannot "
                    f"clear it, so the run aborts rather than filling up with extractive "
                    f"fallbacks. Switch provider or model in config.yaml (llm.providers, "
                    f"llm.{name}_model), or re-run with --offline to replay cache."
                ) from e
            retry_after = e.response.headers.get("retry-after")
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                retry_after, delay = None, 2.0 * (2 ** wait_number)
            if retry_after is not None and delay > max_delay:
                raise QuotaExhausted(
                    f"provider '{name}' asked for a {delay:.0f}s wait (HTTP 429), longer than "
                    f"the {max_delay:.0f}s this run is willing to sleep per question. "
                    f"Aborting rather than filling the run with extractive fallbacks. Wait it "
                    f"out, switch provider in config.yaml (llm.providers), or re-run with "
                    f"--offline to replay cache."
                ) from e
            if wait_number == max_waits:
                raise QuotaExhausted(
                    f"provider '{name}' stayed rate-limited (HTTP 429) across {max_waits} "
                    f"waits. Treating this as an exhausted quota and aborting rather than "
                    f"filling the run with extractive fallbacks. Wait it out, switch "
                    f"provider in config.yaml (llm.providers), or re-run with --offline."
                ) from e
            delay = min(delay + 0.5, max_delay)
            print(f"  '{name}' rate-limited (429); waiting {delay:.1f}s "
                  f"[{wait_number + 1}/{max_waits}]")
            time.sleep(delay)
    return None


def deterministic_fallback(scored_chunks: List[Tuple[Chunk, float]]) -> AgentAnswer:
    """Last-resort extractive answer: quotes the top reranked chunk as its own citation.

    This is a transport, not a reasoner. It cannot judge whether the passage actually
    answers the question, so it can never abstain and its claim trivially entails
    itself. Answers carrying generator="extractive-fallback" are excluded from headline
    evaluation metrics for exactly that reason.
    """
    if not scored_chunks:
        return AgentAnswer(insufficient_evidence=True, claims=[],
                           reasoning="No context chunks available.",
                           generator="extractive-fallback")

    top_chunk, _ = scored_chunks[0]
    return AgentAnswer(
        insufficient_evidence=False,
        claims=[Claim(text=top_chunk.text, citations=[top_chunk.chunk_id])],
        reasoning="Deterministic extractive fallback from top context chunk; not model-generated.",
        generator="extractive-fallback",
    )


def generate_answer(
    query: str,
    scored_chunks: List[Tuple[Chunk, float]],
    cfg: Config,
    offline: bool = False,
) -> AgentAnswer:
    """Structured generation with provider fallback: configured providers -> cache -> extractive."""
    if not scored_chunks:
        return AgentAnswer(insufficient_evidence=True, claims=[],
                           reasoning="No retrieved context chunks.", generator="none")

    prompt = build_generation_prompt(query, scored_chunks)
    valid_chunk_ids = {c.chunk_id for c, _ in scored_chunks}

    quota_error: Optional[QuotaExhausted] = None
    if not offline:
        for name in cfg.llm.providers:
            call = _PROVIDERS.get(name)
            if call is None or name in _SPENT:
                continue
            for attempt in range(cfg.llm.max_retries + 1):
                try:
                    answer = _call_with_backoff(call, name, prompt, cfg, valid_chunk_ids)
                except QuotaExhausted as e:
                    # Remembered for the rest of the process: re-asking a spent quota once
                    # per question burns the run's time budget for 429s we already have.
                    _SPENT.add(name)
                    quota_error = e
                    print(f"Warning: provider '{name}' is out of quota; skipped from here on.")
                    break
                except Exception as e:
                    print(f"Warning: provider '{name}' failed: {type(e).__name__}: {e}")
                    break  # transport/auth failure won't fix itself on a retry
                if answer is not None:
                    save_to_cache(compute_prompt_hash(name, prompt), answer)
                    return answer
                if attempt < cfg.llm.max_retries:
                    print(f"Warning: provider '{name}' returned unparseable JSON; retrying.")

    # Reached only if no provider answered. A spent quota must not become a cache replay or
    # an extractive fallback here: that is how a benchmark fills up with answers no model
    # produced. Providers that merely failed still fall through, as before.
    if quota_error is not None:
        raise quota_error

    # Offline replay: only an identical model+prompt from a previous online run hits.
    for name in cfg.llm.providers:
        cached = get_cached_response(compute_prompt_hash(name, prompt))
        if cached:
            return cached

    return deterministic_fallback(scored_chunks)
