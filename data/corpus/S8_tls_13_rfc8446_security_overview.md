# RFC 8446 Security Overview: Transport Layer Security (TLS) Protocol Version 1.3

## Core Protocol Enhancements
TLS 1.3 (RFC 8446) redesigns the handshake protocol to improve security and latency over TLS 1.2:
1. **1-RTT Handshake:** Reduces connection handshake latency from 2-RTT to 1-RTT for full handshakes and 0-RTT for resumption (session tickets).
2. **Mandatory Perfect Forward Secrecy (PFS):** Removes RSA static key exchange; all handshakes require Ephemeral Diffie-Hellman (ECDHE or DHE).
3. **Deprecation of Insecure Algorithms:** Removes support for RC4, MD5, SHA-224, 3DES, static RSA key exchange, and CBC mode ciphers.

## Cryptographic Cipher Suites
TLS 1.3 defines simplified cipher suites specifying authenticated encryption with associated data (AEAD):
- `TLS_AES_128_GCM_SHA256`
- `TLS_AES_256_GCM_SHA384`
- `TLS_CHACHA20_POLY1305_SHA256`
