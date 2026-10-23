#!/usr/bin/env python3
"""Repository Security Audit Script.

Performs offline automated security checks:
1. Verifies no sensitive files (tokens, private keys, database dumps, .env) are tracked in Git.
2. Scans repository source files for accidental secrets and private keys.
3. Checks local service token file permissions (must be 0600 if present).
4. Verifies local service default configuration binds strictly to loopback (127.0.0.1).
5. Verifies extension manifest host permissions are restricted to loopback endpoints.
"""

from __future__ import annotations

import json
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

# Sensitive file patterns that must NEVER be tracked by Git
BLOCKED_TRACKED_PATTERNS = [
    r"^.*service_token.*$",
    r"^.*\.env(\..+)?$",
    r"^.*\.sqlite(-.+)?$",
    r"^.*\.pem$",
    r"^.*\.key$",
    r"^.*id_rsa.*$",
]

# Regex patterns for accidental secrets
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Private Key", re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH) PRIVATE KEY-----")),
    ("GitHub Token", re.compile(r"(?:ghp_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{82})")),
    ("Anthropic API Key", re.compile(r"sk-ant-api[0-9a-zA-Z\-_]{20,}")),
    ("OpenAI API Key", re.compile(r"sk-[a-zA-Z0-9]{32,}")),
    ("Generic Bearer Secret", re.compile(r'(?i)(?:bearer\s+token|secret_key)\s*[:=]\s*["\']([a-zA-Z0-9_\-]{32,})["\']')),
]

# Directories and files to exclude from source secret scanning
SCAN_EXCLUDE_DIRS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    "__pycache__",
    ".idea",
    ".vscode",
}

# Allow-listed test files containing mock tokens/regexes
SCAN_EXCLUDE_FILES = {
    "scripts/security_audit.py",  # Contains regex patterns
    "tests/test_security.py",     # Unit tests verifying API key redaction with dummy keys
}


def get_git_executable() -> str:
    """Return absolute path to git executable."""
    return shutil.which("git") or "/usr/bin/git"


def get_repo_root() -> Path:
    """Find repository root using git or current file location."""
    git_bin = get_git_executable()
    try:
        root = subprocess.check_output(  # noqa: S603
            [git_bin, "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return Path(root)
    except Exception:
        return Path(__file__).resolve().parent.parent


def check_git_tracked_files(repo_root: Path) -> list[str]:
    """Ensure no forbidden/sensitive files are tracked by Git."""
    findings = []
    git_bin = get_git_executable()
    try:
        tracked = subprocess.check_output(  # noqa: S603
            [git_bin, "ls-files"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).splitlines()
    except Exception as exc:
        findings.append(f"Could not list git tracked files: {exc}")
        return findings

    compiled_blocked = [re.compile(p) for p in BLOCKED_TRACKED_PATTERNS]

    for file_path in tracked:
        for pattern in compiled_blocked:
            if pattern.search(file_path):
                findings.append(f"Tracked sensitive file detected: {file_path}")
    return findings


def scan_source_for_secrets(repo_root: Path) -> list[str]:
    """Scan tracked and working tree source files for potential secrets."""
    findings = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue

        rel_path = path.relative_to(repo_root).as_posix()

        # Skip excluded dirs and files
        parts = set(rel_path.split("/"))
        if parts & SCAN_EXCLUDE_DIRS or rel_path in SCAN_EXCLUDE_FILES:
            continue

        # Skip binary files, models, and images
        if path.suffix.lower() in {".joblib", ".pkl", ".png", ".jpg", ".jpeg", ".ico", ".zip", ".tar", ".gz"}:
            continue

        # Skip files inside tests/ that use mock tokens if marked as test files
        # but still scan them if they contain actual private keys
        is_test_file = "tests/" in rel_path or "test_" in path.name

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            content = ""

        if not content:
            continue

        for label, pattern in SECRET_PATTERNS:
            # For mock tokens in tests, ignore Generic Bearer Secret matches
            if is_test_file and label == "Generic Bearer Secret":
                continue
            for line_no, line in enumerate(content.splitlines(), start=1):
                if pattern.search(line):
                    findings.append(
                        f"Potential secret [{label}] found in {rel_path}:{line_no}"
                    )
    return findings


def check_service_token_permissions(repo_root: Path) -> list[str]:
    """Verify data/processed/service_token permissions are 0600 if file exists."""
    findings = []
    token_path = repo_root / "data" / "processed" / "service_token"
    if not token_path.exists():
        return findings

    mode = stat.S_IMODE(token_path.stat().st_mode)
    if mode != 0o600:
        findings.append(
            f"Insecure permissions on {token_path}: octal {oct(mode)} (expected 0600)"
        )
    return findings


def check_service_loopback_config(repo_root: Path) -> list[str]:
    """Verify local service configuration defaults strictly to 127.0.0.1."""
    findings = []
    config_file = repo_root / "src" / "detector_service" / "config.py"
    if not config_file.exists():
        findings.append("Service configuration file missing: src/detector_service/config.py")
        return findings

    content = config_file.read_text(encoding="utf-8")
    if 'host: str = "127.0.0.1"' not in content:
        findings.append("Service config does not default host to 127.0.0.1")
    if "allow_external_binding: bool = False" not in content:
        findings.append("Service config does not default allow_external_binding to False")
    return findings


def check_extension_manifest(repo_root: Path) -> list[str]:
    """Verify extension manifest adheres to least-privilege host permissions."""
    findings = []
    manifest_file = repo_root / "extension" / "manifest.json"
    if not manifest_file.exists():
        findings.append("Extension manifest missing: extension/manifest.json")
        return findings

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        findings.append(f"Failed to parse extension manifest: {exc}")
        return findings

    host_perms = manifest.get("host_permissions", [])
    allowed_prefixes = ("http://127.0.0.1", "http://localhost", "http://[::1]")

    for perm in host_perms:
        if perm in ("<all_urls>", "*://*/*", "http://*/*", "https://*/*"):
            findings.append(f"Overly broad extension host permission: {perm}")
        elif not any(perm.startswith(prefix) for prefix in allowed_prefixes):
            findings.append(f"Unexpected non-loopback host permission: {perm}")

    return findings


def main() -> int:
    repo_root = get_repo_root()
    print(f"[*] Running security audit on repository: {repo_root}")

    all_findings: list[tuple[str, list[str]]] = [
        ("Git Tracked Files Check", check_git_tracked_files(repo_root)),
        ("Secret Scanning Check", scan_source_for_secrets(repo_root)),
        ("Token Permissions Check", check_service_token_permissions(repo_root)),
        ("Service Loopback Config Check", check_service_loopback_config(repo_root)),
        ("Extension Permissions Check", check_extension_manifest(repo_root)),
    ]

    total_issues = 0
    for section_name, findings in all_findings:
        if findings:
            print(f"\n[FAIL] {section_name}:")
            for f in findings:
                print(f"  - {f}")
            total_issues += len(findings)
        else:
            print(f"[PASS] {section_name}")

    if total_issues > 0:
        print(f"\n[!] Security audit failed with {total_issues} issue(s).")
        return 1

    print("\n[✓] All security audit checks passed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
