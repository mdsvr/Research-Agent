# Technical Analysis of CVE-2024-3094: XZ Utils Backdoor

## Overview
CVE-2024-3094 is a critical supply-chain vulnerability discovered in XZ Utils versions 5.6.0 and 5.6.1. The backdoor was introduced by an upstream maintainer account named Jia Tan over an extended period. The compromise alters functions within `liblzma`, specifically targeting OpenSSH server authentication mechanisms when linked against `systemd` socket activation libraries.

## Technical Mechanism
The malicious payload is obfuscated within binary test files (`bad-3-corrupt_lzma2.xz` and `good-large_compressed.lzma`) included in the source tarballs. During the build process via `m4/build-to-host.m4`, a complex bash extraction script decodes the payload if specific target conditions are met (e.g., x86-64 Linux environment using GCC and GNU ld).

The extracted object file intercepts the RSA key signature verification function `RSA_public_decrypt` in OpenSSH (`sshd`). By hooking into `systemd` notifications that link `libsystemd` to `liblzma`, the backdoor executes arbitrary commands with root privileges prior to signature validation when a specifically crafted SSH authentication certificate is presented.

## Detection and Mitigation
The vulnerability was initially identified by Andres Freund while benchmarking SSH latencies, noticing abnormal CPU utilization and 500ms delay increments during connection attempts. Affected systems must immediately downgrade `xz-utils` to version 5.4.x or uncompromised 5.6.2 builds.

CVSS v3.1 Base Score: 10.0 (Critical) — Vector: CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:H/A:H.
