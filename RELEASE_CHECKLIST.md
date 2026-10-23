# Release Checklist

This document defines the step-by-step release procedure for **Phishing URL Detector** releases, covering the core Python detection engine, the local FastAPI service, and the Firefox WebExtension.

---

## 1. Pre-Release Verification

- [ ] **Clean Working Tree**: Ensure there are no uncommitted or untracked sensitive files.
  ```bash
  git status
  ```
- [ ] **Version Alignment**: Verify that the release version string is identical across all three components:
  - `pyproject.toml` (`version = "X.Y.Z"`)
  - `src/detector/__init__.py` (`__version__ = "X.Y.Z"`)
  - `extension/manifest.json` (`"version": "X.Y.Z"`)
- [ ] **Changelog / Release Notes**: Ensure release notes document all new features, security hardening, bug fixes, and any deprecations.

---

## 2. Automated Test & Quality Suite

Run all local verification suites:

- [ ] **Python Test Suite**:
  ```bash
  pytest --tb=short
  ```
  *(Must pass 100% of tests with 0 failures).*

- [ ] **Ruff Static Analysis**:
  ```bash
  ruff check src/ tests/ scripts/
  ```
  *(Must report zero errors or warnings).*

- [ ] **Firefox Extension Unit & Regression Tests**:
  ```bash
  node extension/tests/run_tests.js
  ```
  *(Must pass all 26 extension tests).*

- [ ] **Repository Security Audit**:
  ```bash
  python3 scripts/security_audit.py
  ```
  *(Must verify clean Git tracking, no secrets, safe service loopback config, and valid extension permissions).*

- [ ] **Continuous Integration**:
  Push changes or create a release PR to verify that the GitHub Actions matrix workflow passes across all target Python versions (3.10, 3.11, 3.12) and Node.js.

---

## 3. Firefox Extension Deterministic Packaging

- [ ] **Build Release Archive**:
  ```bash
  python3 scripts/package_extension.py
  ```
  This creates:
  - `dist/phishing-url-detector-extension-<version>.zip`
  - `dist/phishing-url-detector-extension-<version>.zip.sha256`

- [ ] **Verify Archive Contents**:
  Confirm that `dist/*.zip` contains only the runtime files (no tests, no docs, no hidden files):
  ```bash
  python3 -c "import zipfile; z = zipfile.ZipFile('dist/phishing-url-detector-extension-0.5.0.zip'); print('\n'.join(z.namelist()))"
  ```

- [ ] **Verify Checksum**:
  ```bash
  cd dist && sha256sum -c phishing-url-detector-extension-<version>.zip.sha256 && cd ..
  ```

- [ ] **Verify Deterministic Reproducibility**:
  Ensure running `package_extension.py` in a separate build directory produces the identical SHA-256 digest.

---

## 4. Manual Firefox Runtime Smoke Test

- [ ] **Start Local Service**:
  ```bash
  uvicorn detector_service.server:app --host 127.0.0.1 --port 8000
  ```
- [ ] **Note Service Token**:
  Retrieve the generated token from `data/processed/service_token` or service startup logs.
- [ ] **Load Extension in Firefox**:
  1. Open Firefox and navigate to `about:debugging#/runtime/this-firefox`.
  2. Click **Load Temporary Add-on...**
  3. Select `extension/manifest.json` (or the packaged zip).
- [ ] **Configure Extension Options**:
  1. Open extension Options page.
  2. Enter the local service address (`http://127.0.0.1:8000`) and paste the service Bearer token.
  3. Click **Save Settings** and verify the "Saved successfully" indicator.
- [ ] **Verify Toolbar Popup**:
  Click the extension toolbar icon; confirm service status displays **Connected** (green dot).
- [ ] **Test Safe Navigation**:
  Navigate to `https://example.com`; verify badge remains green/clear with no interstitial.
- [ ] **Test Phishing Navigation**:
  Navigate to a known phishing test URL or simulated phishing pattern; verify:
  1. The navigation is intercepted immediately.
  2. The warning interstitial (`warning.html`) is displayed.
  3. "Back to Safety" returns to previous page.
  4. "Continue to site (once)" permits a single bypass without repeating the warning indefinitely.

---

## 5. Tagging & Publishing

*Note: Never publish or tag releases without explicit user permission.*

- [ ] **Commit Release Preparation**:
  ```bash
  git commit -m "Release vX.Y.Z"
  ```
- [ ] **Create Cryptographic Git Tag**:
  ```bash
  git tag -s vX.Y.Z -m "Release vX.Y.Z"
  ```
- [ ] **Publish GitHub Release**:
  - Attach the release notes.
  - Attach the deterministic zip: `dist/phishing-url-detector-extension-X.Y.Z.zip`.
  - Attach the checksum file: `dist/phishing-url-detector-extension-X.Y.Z.zip.sha256`.
  - Include the SHA-256 hash in the release description for user verification.
