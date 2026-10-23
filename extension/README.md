# Phishing URL Detector — Firefox WebExtension

A lightweight, privacy-preserving Firefox browser extension (Manifest V3) that integrates with the local, offline **Phishing URL Detector** FastAPI service.

As you browse, the extension monitors top-level website navigations, analyzes URLs in real time through your local machine's machine learning and lexical detection engine, and displays clear warning badges and interstitial blocking pages when dangerous phishing sites are detected.

---

## Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Firefox Browser                                  │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │ Top-Level Navigation (http / https)                                  │ │
│   └──────────────────────────────────┬────────────────────────────────────┘ │
│                                      │ webNavigation.onCommitted            │
│                                      ▼ (frameId === 0)                      │
│   ┌───────────────────────────────────────────────────────────────────────┐ │
│   │ Background Service (background.js)                                    │ │
│   │  ├── Ineligible Scheme Filter (ignores about:, file:, moz-extension:)  │ │
│   │  ├── Continue-Once Bypass Check (single-navigation exception)         │ │
│   │  ├── LRU Cache & In-Flight Request Deduplication                      │ │
│   │  └── Local API Client (api_client.js)                                 │ │
│   └──────────────────────────────────┬────────────────────────────────────┘ │
└──────────────────────────────────────┼──────────────────────────────────────┘
                                       │ HTTP POST /api/v1/analyze
                                       │ Headers: Authorization: Bearer <token>
                                       │          X-Detector-Client: firefox-extension
                                       │ Host: 127.0.0.1:8000 (Loopback only)
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Local FastAPI Detection Service                         │
│                           (http://127.0.0.1:8000)                           │
│                                                                             │
│   ├── Host Header Validation & DNS Rebinding Mitigation                     │
│   ├── Constant-time Bearer Token Authentication                             │
│   ├── Concurrency Limiter (asyncio.Semaphore)                               │
│   └── Multi-Layer Analysis Pipeline (Layer 1 -> Layer 2 -> Layer 3 -> L4)   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ JSON Response
                                       │ { classification, risk_score, reasons }
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            User Interface Action                            │
│                                                                             │
│   Safe:        Green 'OK' badge on toolbar icon                             │
│   Uncertain:   Amber '?' badge on toolbar icon                              │
│   Phishing:    Red '!' badge + Automatic redirect to warning interstitial   │
│                (warning/warning.html?target=...)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Privacy & Security Principles

1. **Strictly Local-First**: The extension sends URLs **only** to the loopback interface (`127.0.0.1` or `localhost`). It contains no third-party analytics, cloud fallbacks, or external telemetry.
2. **Loopback Protection**: The extension's client verifies that the configured service endpoint resolves to a loopback address before sending any requests or bearer tokens. Remote addresses are rejected.
3. **Authentication Isolation**: The service access token is stored inside Firefox's secure, origin-isolated `browser.storage.local`. It is never injected into web pages or transmitted externally.
4. **XSS & DOM-Injection Immune**: All dynamic web values (URLs, domains, reason strings, risk scores) are rendered using safe DOM APIs (`textContent`, `document.createTextNode`). No `innerHTML` or `document.write` is used anywhere in the extension codebase.
5. **Recursion & Loop Guards**: Extension internal pages (`moz-extension://`) and browser internals (`about:`, `chrome:`, `javascript:`, `data:`) are strictly ignored, preventing infinite redirect loops.
6. **Fail-Open UX**: If the local Python service is stopped, restarting, or timing out, web browsing continues uninterrupted with a neutral status badge.

---

## Prerequisites

- **Firefox** (v109.0 or later; supports Manifest V3 with background scripts).
- **Python 3.11+** environment with project dependencies and service extras installed.
- **Node.js** (v18+ or later) for running extension unit tests.

---

## Installation & Setup Walkthrough

### Step 1: Start the Local Detector Service

From the root of the repository:

```bash
# Activate your virtual environment
source .venv/bin/activate

# Start the hardened local service
python -m detector_service
```

The service will output startup details:
```
INFO: Starting Phishing URL Detector Service on 127.0.0.1:8000
INFO: Auth token generated: [64-character hex string]
INFO: Token saved with permissions 0600 to data/processed/service_token
```

### Step 2: Retrieve the Authentication Token

The service auto-generates a secure 64-character bearer token saved with restricted user permissions (`0600`):

```bash
cat data/processed/service_token
```

Copy this token to your clipboard.

### Step 3: Load the Temporary Add-on in Firefox

1. Open Firefox and navigate to `about:debugging#/runtime/this-firefox`.
2. Click **Load Temporary Add-on...**
3. In the file picker, select `extension/manifest.json` from this repository.
4. The **Phishing URL Detector** extension will appear under "Temporary Extensions" with ID `phishing-detector@local`.

### Step 4: Configure the Extension

1. In Firefox, click the **Phishing URL Detector** shield icon on your toolbar (or navigate to the Options page via `about:addons` -> **Phishing URL Detector** -> **Preferences** / **Options**).
2. Click the gear icon in the popup or open the settings page directly.
3. Verify the **Service Address** (default: `http://127.0.0.1:8000`).
4. Paste the 64-character token copied from Step 2 into the **Local Bearer Token** field.
5. Click **Test Connection**. You should see a green confirmation:
   > *Connected successfully! Service v0.5.0, ML model loaded.*
6. Click **Save Settings**.

---

## How to Test Safely

> [!CAUTION]
> **NEVER test the extension by browsing to real, live phishing websites.**
> Live phishing pages frequently host exploit kits, drive-by malware, and credential-harvesting scripts. Always test using safe, benign websites or locally simulated test endpoints.

### Test 1: Verifying a Safe Website
1. Open a new tab and navigate to `https://www.wikipedia.org` or `https://github.com`.
2. Observe the extension badge turn green **OK**.
3. Click the toolbar icon to open the popup:
   - Status: **Online (Model Ready)**
   - Verdict: **SAFE**
   - Risk Score: **0.00**
   - Analysis Evidence: *No security issues detected.*

### Test 2: Verifying Phishing Interstitial Warning (Simulated URL)
You can test the interstitial warning safely without exposing your browser to real malware:

1. **Option A: Local Test Server or Host Alias**
   The detector flags known phishing brand misplacements and suspicious TLDs (e.g. PayPal brand keywords on unauthorized domains).
   You can submit a simulated URL directly using the service test client or map a temporary `/etc/hosts` alias to `127.0.0.1` (such as `http://paypal-security-verification.test:8000`).

2. **Option B: Manual Verification in Warning Interstitial**
   Navigate directly to the extension warning page in Firefox to inspect its layout:
   ```
   moz-extension://<UUID>/warning/warning.html?target=http%3A%2F%2Fpaypal-verify-login.tk%2Faccount
   ```
   - Notice the prominent warning banner and risk score.
   - Expand **Advanced Options ▼**.
   - Check the single-use acknowledgment checkbox.
   - Click **Continue to Website (Single Use)**.
   - Notice that the bypass allows that specific navigation once and does not grant a permanent whitelist.

---

## Extension Components & File Structure

```
extension/
├── manifest.json              # Firefox Manifest V3 configuration & permissions
├── icons/                     # Shield icons (16px, 32px, 48px, 96px, 128px)
├── background/
│   ├── cache.js               # LRU cache, InFlight deduplication, ContinueOnce manager
│   ├── api_client.js          # HTTP client with loopback verification & header injection
│   └── background.js          # Navigation observer, tab state & badge manager
├── warning/
│   ├── warning.html           # Dedicated interstitial warning page
│   ├── warning.css            # Security warning styling & dark/light mode
│   └── warning.js             # Safe DOM renderer & continue-once dispatcher
├── popup/
│   ├── popup.html             # Toolbar dropdown interface
│   ├── popup.css              # Toolbar popup styling
│   └── popup.js               # Tab status visualizer & rescan trigger
├── options/
│   ├── options.html           # Settings & configuration interface
│   ├── options.css            # Settings styling
│   └── options.js             # Form validator & connection tester
└── tests/
    ├── test_api_client.test.js     # Unit tests for API client & loopback guards
    ├── test_cache.test.js          # Unit tests for LRU cache & concurrency deduplication
    ├── test_url_rules.test.js      # Unit tests for URL schemes & warning sanitization
    ├── test_navigation_flow.test.js # Regression tests for P1 tab ID, P2 navigation races & message handlers
    └── run_tests.js                # Node test runner script
```

---

## Automated Unit Testing

The extension includes a self-contained automated test suite that executes using Node's built-in test runner without external npm dependencies:

```bash
node extension/tests/run_tests.js
```

Or using Node's test runner directly:

```bash
node --test extension/tests/*.test.js
```

### Test Coverage Highlights (26 Passing Tests):
- **Warning Tab ID Resolution (P1)**: Verifies that `resolveWarningTabId` identifies tabs via `browser.tabs.getCurrent()` in both active and background tabs, and safely returns `null` without falling back to the foreground tab.
- **Navigation Sequence & Race Protection (P2)**: Tests fast subsequent navigations overtaking earlier slow requests, late scan results, and tab closures to ensure older results cannot corrupt newer tab states or badges.
- **IPv6 Loopback Support (P3)**: Validates that `manifest.json` permits `"http://[::1]/*"` and `api_client.js` validates IPv6 loopback.
- **Header Security (P4)**: Asserts that requests do not attempt to set forbidden headers such as `'Host'`.
- **Loopback Enforcement**: Tests that requests to non-loopback IPs (e.g. `192.168.1.1`, `example.com`) throw errors and refuse to send tokens.
- **Cache LRU & TTL**: Verifies LRU eviction and millisecond expiration behavior.
- **In-Flight Request Deduplication**: Verifies that concurrent navigations to identical URLs trigger only one underlying HTTP request.
- **Single-Use Continue Once**: Verifies that bypass tokens can only be consumed once and are scoped to the specific tab and URL.
- **URL Eligibility**: Verifies that `http:` and `https:` are inspected, while `about:`, `file:`, `chrome:`, `moz-extension:`, and `javascript:` are rejected.

---

## Packaging for Release

The extension includes a deterministic packaging script that compiles release archives and computes standard SHA-256 checksums without external dependencies:

```bash
python3 scripts/package_extension.py
```

This validates `extension/manifest.json`, verifies required runtime assets, excludes development files and tests, and generates:
- `dist/phishing-url-detector-extension-<version>.zip`
- `dist/phishing-url-detector-extension-<version>.zip.sha256`

### Verifying Checksum
```bash
cd dist && sha256sum -c phishing-url-detector-extension-*.zip.sha256 && cd ..
```

The resulting zip archive can be loaded directly into Firefox Developer Edition/Nightly or submitted for signing to Mozilla Add-ons (AMO).

---

## Known Limitations

- **Temporary Extension Lifecycle**: In standard Firefox builds, extensions loaded via `about:debugging` remain loaded until Firefox is restarted. For permanent installation, extensions must be signed by Mozilla Add-ons (AMO) or loaded into Firefox Developer Edition / Nightly with signing disabled (`xpinstall.signatures.required = false`).
- **Top-Level Navigation Only**: Only top-level document navigations (`frameId === 0`) are inspected. Subresource requests, web workers, and third-party iframe content are not blocked independently.
- **Single-Use Bypass Scope**: Clicking "Continue to Website" grants access only to the immediate single navigation. Refreshing or navigating back to the warning URL requires re-acknowledgment to prevent accidental persistence of malicious sites.
