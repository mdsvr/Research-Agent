# Cybersecurity Advisory: SolarWinds SUNBURST Supply Chain Attack

## Executive Summary
In late 2020, threat group UNC2452 executed a sophisticated supply chain attack against SolarWinds Orion platform releases 2019.4 HF 5 through 2020.2.1 HF 1. The attackers trojanized the source code build environment to inject a backdoor component dubbed "SUNBURST" (`SolarWinds.Orion.Core.BusinessLayer.dll`).

## Attack Characteristics & Persistence
- **Digitally Signed DLL:** The backdoor payload was signed with a legitimate SolarWinds digital certificate, allowing it to bypass endpoint detection mechanisms.
- **Staged Execution:** The malware remains dormant for up to 14 days before attempting DNS lookups against attacker-controlled C2 domains (`avsvmcloud.com`).
- **Stealth Techniques:** SUNBURST checks for running security software, forensic drivers, and anti-virus services prior to executing command execution payloads.

## Remediation Steps
1. Upgrade SolarWinds Orion platform to version 2020.2.1 HF 2 or 2020.2.5 immediately.
2. Isolate affected Orion servers and rotate all active directory administrative credentials and SAML signing certificates.
