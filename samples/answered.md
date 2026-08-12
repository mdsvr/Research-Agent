# Sample Runs: Answered Queries

Verbatim console output from `py -m veritas ask "..." --offline` on the current corpus.
No live API key was available on this machine, so generation came from the prompt-hash
replay cache where a previous run had covered the exact prompt, and from the deterministic
extractive fallback otherwise. The agent says so on every extractive answer rather than
presenting an extract as a generated response.

Traces for these runs are written to `traces/`.

---

## Sample 1 — Quote grounding refusing a true claim

**Query:** `What functions were targeted in the XZ Utils backdoor?`

```
[ABSTAINED]
Post-verification check failed: 0 of 0 generated claims were verified by the NLI entailment model (minimum required: 1).
Closest matching passages:
  - [S1::c00] "# Technical Analysis of CVE-2024-3094: XZ Utils Backdoor

## Overview
CVE-2024-3094 is a critical supply-chain vulnerabi..." (relevance: 0.99)
  - [S1::c03] "The vulnerability was initially identified by Andres Freund while benchmarking SSH latencies, noticing abnormal CPU util..." (relevance: 0.02)
Reason: Passages discuss the general topic but do not state the specific requested facts.
```

The cached `llama-3.3-70b` answer for this prompt is **substantively right** — it names
`liblzma`, OpenSSH server authentication and `RSA_public_decrypt`. It was refused anyway,
because its supporting quote stitches two separate spans together with an ellipsis:

```
"The compromise alters functions within `liblzma`, specifically targeting OpenSSH server
 authentication mechanisms when linked against `systemd` socket activation libraries. ...
 The extracted object file intercepts the RSA key signature verification function
 `RSA_public_decrypt` in OpenSSH (`sshd`)."
```

Neither half is fabricated, but the string as a whole appears nowhere in the corpus, so the
whitespace-normalised substring check drops it, Gate B sees zero surviving claims, and the
agent abstains. This is the deliberate trade: the grounding check is not allowed to reason
about what the model *meant*, because that is exactly the latitude a fabricated quote would
exploit. The cost is visible in the evaluation table as a real over-refusal.

---

## Sample 2 — The failure mode this pipeline is built to prevent, still visible without an LLM

**Query:** `What is the exact CVSS score of TLS 1.3 RFC 8446?`

```
[ANSWER]
- # RFC 8446 Security Overview: Transport Layer Security (TLS) Protocol Version 1.3
...
2. **Mandatory Perfect Forward Secrecy (PFS):** Removes RSA static key exchange; all
handshakes require Ephemeral Diffie-Hellman (ECDHE or DHE).
[S8::c00] (entailment 0.88 via lytang/MiniCheck-DeBERTa-v3-Large)

Note: no language model was available, so this is a verbatim extract of the top-ranked
passage, not a generated answer.
```

RFC 8446 is a protocol standard and carries no CVSS score. The retrieved passage is
genuinely about TLS 1.3 (reranker score 0.98), so Gate A passes; Gate B cannot fire
because the extractive fallback's claim trivially entails itself. This is the case that
requires a real generator to set `insufficient_evidence`, and it is why the evaluation
harness refuses to report end-to-end abstention numbers as representative in this mode.

Reproduce with a generator configured:

```bash
export GROQ_API_KEY=...         # or GEMINI_API_KEY / OPENROUTER_API_KEY, or run Ollama
py -m veritas ask "What is the exact CVSS score of TLS 1.3 RFC 8446?"
```
