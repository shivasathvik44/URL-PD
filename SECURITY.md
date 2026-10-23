# Security Policy

The **Phishing URL Detector** project takes the security of its detection engine, local FastAPI service, and Firefox WebExtension seriously. This document outlines our security policies, supported versions, threat model, and instructions for reporting vulnerabilities.

---

## Supported Versions

Only the latest active minor release receives security updates.

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5.0 | :x:                |

---

## Reporting a Vulnerability

If you discover a security vulnerability, **do not open a public GitHub issue**.

Please report vulnerabilities through **GitHub Private Vulnerability Reporting** on this repository (under the **Security** tab -> **Report a vulnerability**), or email the maintainers directly.

### What to Include in Your Report

To help us triage and resolve the issue quickly, please provide:
1. A clear description of the vulnerability.
2. Step-by-step reproduction instructions or a minimal Proof of Concept (PoC).
3. The affected component (e.g., core detector, local service, or Firefox extension) and version/commit hash.
4. An assessment of the potential impact.
5. Any proposed remediation or patches, if available.

### Response Timeline

- **Initial Acknowledgment:** Within 48–72 hours of receiving the report.
- **Triage & Assessment:** Within 5 business days, confirming whether the report is accepted or requires additional detail.
- **Fix & Disclosure:** We aim to release a patch within 14–30 days depending on severity and coordinate a public disclosure date once fixes are verified.

---

## Threat Model & Security Boundaries

The project operates under a defense-in-depth security architecture designed around three primary tiers:

### 1. Local FastAPI Service
- **Loopback-Only Binding:** Binds strictly to `127.0.0.1` by default. Binding to `0.0.0.0` or external network interfaces is rejected unless an explicit environment flag is deliberately passed.
- **Host Header Validation:** Mitigates DNS rebinding attacks by validating the HTTP `Host` header against allowed loopback origins (`127.0.0.1:<port>`, `localhost:<port>`, `[::1]:<port>`).
- **Bearer Token Authentication:** Requests to `/api/v1/analyze` require a cryptographically random Bearer token (`secrets.token_urlsafe(32)`). Authentication checks use timing-safe comparison (`hmac.compare_digest`). The token file (`data/processed/service_token`) is created with strict file permissions (`0o600`).
- **Request Validation:** Incoming URLs are strictly validated: length capped at 2048 characters, scheme restricted to HTTP/HTTPS, and malformed inputs rejected early.

### 2. SSRF Protection (Layer 4 Content Fetcher)
- **Pre-Fetch Target Validation:** Any outbound HTTP requests executed by the Layer 4 content fetcher resolve the destination host before connecting and reject private, loopback, link-local, multicast, reserved, and cloud metadata IP addresses (e.g., `169.254.169.254`).
- **Redirect Revalidation:** Each redirect hop is independently re-validated against the SSRF filter before being followed.
- **Resource Constraints:** Timeouts and byte-stream caps prevent slowloris or denial-of-service via oversized page payloads.

### 3. Firefox WebExtension
- **Least Privilege:** Permissions are restricted to `storage`, `tabs`, and `webNavigation`. Host permissions are strictly restricted to loopback service endpoints (`http://127.0.0.1/*`, `http://localhost/*`, `http://[::1]/*`).
- **Navigation Race Protections:** Monotonic navigation sequence counters ensure asynchronous scan responses from earlier, obsolete navigations cannot overwrite the state of a newer top-level page visit.
- **Fail-Closed Warnings:** If the warning interstitial cannot reliably verify the active tab via `browser.tabs.getCurrent()`, navigation is aborted and the continue action is disabled.

---

## Scope: What Is and Isn't a Vulnerability

### In-Scope Vulnerabilities
- Authentication bypass or timing side-channels in the local service token check.
- DNS rebinding or `Host` header validation bypasses.
- Server-Side Request Forgery (SSRF) bypasses allowing Layer 4 fetcher to reach private/loopback/cloud-metadata networks.
- Remote or local code execution, path traversal, or command injection.
- Cross-Site Scripting (XSS) or privilege escalation in the Firefox WebExtension context.
- Insecure storage or exposure of authentication tokens.

### Out-of-Scope
- **Heuristic False Positives / False Negatives:** Phishing detection is probabilistic. An unrecognized phishing URL or a misclassified safe URL is a model tuning issue, not a security vulnerability.
- **Attacks Requiring Full Local Host Compromise:** If an attacker already has local root access or arbitrary code execution under the local user's account, attacks reading user-owned memory or files are outside the extension's threat model.
- **Denial of Service from Local User Action:** Flooding the service locally from an account that already possesses the local authentication token.
