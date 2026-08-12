# Heartbleed Bug Advisory: OpenSSL Memory Disclosure (CVE-2014-0160)

## Overview
CVE-2014-0160, known as "Heartbleed", is a severe memory leakage flaw in the OpenSSL cryptographic software library (versions 1.0.1 through 1.0.1f). The bug allows remote attackers to read up to 64 kilobytes of host memory per heartbeat payload without authorization.

## Root Cause Analysis
The flaw stems from missing bounds checking on a TLS heartbeat extension request payload. A client sends a heartbeat request containing a payload data buffer and an explicit 16-bit integer length field. The OpenSSL server allocates a response buffer matching the declared length field and uses `memcpy()` to copy payload bytes from memory without validating that the declared length matches the actual received payload byte size.

As a result, the server reads past the allocated request buffer into neighboring process memory, returning secret material including SSL server private keys, session tokens, user passwords, and sensitive memory pages.

## Fix and Remediation
- Upgrade OpenSSL to version 1.0.1g or compile OpenSSL with flag `-DOPENSSL_NO_HEARTBEATS`.
- Revoke and re-issue all SSL/TLS private keys and certificates.
- Force password resets across all affected user authentication databases.

CVSS v2 Base Score: 5.0 (CVSS v3: 7.5 High).
