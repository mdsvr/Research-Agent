# Sample Runs: Answered Queries

Verbatim console output from `py -m veritas ask "..."` against `llama-3.1-8b-instant`, the
generator the shipped benchmark was measured with. Traces for these runs are in `traces/`.

---

## Sample 1 — Answered, citation verified

**Query:** `What CVSS v3.1 base score was assigned to CVE-2024-3094?`

```
[ANSWER]
- CVE-2024-3094 has a CVSS v3.1 base score of 10.0 (Critical) [S1::c03] (entailment 0.94 via lytang/MiniCheck-DeBERTa-v3-Large)
```

`S1::c03` is the annotated gold chunk. The 0.94 is a real support probability from the
verifier, and this claim is the one the whole premise-construction design exists for: the
chunk says *"CVSS v3.1 Base Score: 10.0 (Critical)"* and never repeats the CVE identifier,
which lives in the document title. Scored against the bare chunk the same claim gets 0.002 —
the model will not resolve the reference across sentences. The premise is therefore a window
around the quote **headed by the source document's title**, which is part of the source, so
it adds context and never evidence. See `verify.build_premise`.

---

## Sample 2 — The refusal the pipeline is built for

**Query:** `What is the exact CVSS score of TLS 1.3 RFC 8446?`

```
[ABSTAINED]
The generator declared insufficient evidence in context.
Closest passages:
  - [S8::c00] "# RFC 8446 Security Overview: Transport Layer Security (TLS) Protocol Version 1.3

## Core Protocol Enhancements
TLS 1.3..." (relevance: 0.42)
  - [S5::c03] "## Fix and Remediation
- Upgrade OpenSSL to version 1.0.1g or compile OpenSSL with flag `-DOPENSSL_NO_HEARTBEATS`.
- Rev..." (relevance: 0.12)
```

RFC 8446 is a protocol standard and carries no CVSS score. The retrieved passage is
genuinely about TLS 1.3, so Gate A passes — this is exactly the adversarial case Gate A
cannot catch. The refusal comes from the generator setting `insufficient_evidence`, which
the prompt spends most of its budget on, and Gate B honours it. The refusal is informative:
it names the closest passages and their scores, so a reader can tell "not in this corpus"
from "retrieval failed".

---

## Sample 3 — The verifier's ceiling, visible

**Query:** `What functions were targeted in the XZ Utils backdoor?`

```
[ABSTAINED]
Post-verification check failed: 0 of 1 generated claims were verified by the NLI entailment model (minimum required: 1).
Closest matching passages:
  - [S1::c00] "# Technical Analysis of CVE-2024-3094: XZ Utils Backdoor

## Overview
CVE-2024-3094 is a critical supply-chain vulnerabi..." (relevance: 0.99)
  - [S1::c03] "The vulnerability was initially identified by Andres Freund while benchmarking SSH latencies, noticing abnormal CPU util..." (relevance: 0.02)
Reason: Passages discuss the general topic but do not state the specific requested facts.
```

This one is a genuine over-refusal, kept here rather than hidden. The generator produced
*"The functions within `liblzma` were targeted in the XZ Utils backdoor"* — supportable from
`S1::c00` — and the verifier scored it 0.331, below the 0.5 support threshold, so Gate B
refused. It is one of the two over-refusals behind the 13.3% ORR in the evaluation table,
and it is the verifier ceiling documented as Limitation 6, not a retrieval or grounding
failure. Retrieval put the right chunk first at 0.99.
