#!/usr/bin/env python3
"""Reproducible Firefox WebExtension Packaging Script.

Produces deterministic zip archives and SHA-256 checksums of the extension
for releases and distribution.

Features:
- Validates manifest.json (Manifest V3, required metadata, gecko ID).
- Verifies that all referenced runtime assets exist.
- Strictly excludes non-runtime files (tests, temporary files, hidden files, dev docs).
- Normalizes file timestamps (using SOURCE_DATE_EPOCH or 2026-01-01 UTC).
- Normalizes file permissions (0644) and archive file ordering.
- Generates standard GNU-compatible .sha256 checksum files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

# Fixed fallback timestamp (2026-01-01 00:00:00 UTC) for deterministic archives
DEFAULT_SOURCE_DATE_EPOCH = 1767225600

# Subdirectories and files allowed in the release extension archive
ALLOWED_RUNTIME_DIRS = {
    "background",
    "icons",
    "options",
    "popup",
    "warning",
}

EXCLUDED_NAMES = {
    "tests",
    "__pycache__",
    ".DS_Store",
    "Thumbs.db",
    "README.md",
}


class PackagingError(Exception):
    """Raised when extension packaging fails validation or build."""


def get_deterministic_datetime() -> tuple[int, int, int, int, int, int]:
    """Return a fixed (Y, M, D, h, m, s) tuple from SOURCE_DATE_EPOCH or default."""
    try:
        epoch = int(os.environ.get("SOURCE_DATE_EPOCH", DEFAULT_SOURCE_DATE_EPOCH))
    except (ValueError, TypeError):
        epoch = DEFAULT_SOURCE_DATE_EPOCH
    gm = time.gmtime(epoch)
    return (gm.tm_year, gm.tm_mon, gm.tm_mday, gm.tm_hour, gm.tm_min, gm.tm_sec)


def validate_manifest(extension_dir: Path) -> dict:
    """Validate manifest.json syntax, structure, and referenced assets.

    Raises:
        PackagingError: if validation fails.
    """
    manifest_path = extension_dir / "manifest.json"
    if not manifest_path.is_file():
        raise PackagingError(f"manifest.json not found in {extension_dir}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackagingError(f"Failed to parse manifest.json: {exc}") from exc

    # Check manifest version
    if manifest.get("manifest_version") != 3:
        raise PackagingError(
            f"Expected manifest_version 3, found: {manifest.get('manifest_version')}"
        )

    # Check required fields
    for field in ("name", "version", "description"):
        if not manifest.get(field):
            raise PackagingError(f"manifest.json missing required field: '{field}'")

    # Check Firefox-specific gecko id
    gecko = manifest.get("browser_specific_settings", {}).get("gecko", {})
    if not gecko.get("id"):
        raise PackagingError("manifest.json missing browser_specific_settings.gecko.id")

    # Check referenced background scripts
    bg_scripts = manifest.get("background", {}).get("scripts", [])
    if not bg_scripts:
        raise PackagingError("manifest.json has no background.scripts defined")
    for script in bg_scripts:
        script_file = extension_dir / script
        if not script_file.is_file():
            raise PackagingError(f"Referenced background script not found: {script}")

    # Check referenced action popup
    popup = manifest.get("action", {}).get("default_popup")
    if popup and not (extension_dir / popup).is_file():
        raise PackagingError(f"Referenced action default_popup not found: {popup}")

    # Check referenced action icons
    icons = manifest.get("action", {}).get("default_icon", {})
    for size, icon_path in icons.items():
        if not (extension_dir / icon_path).is_file():
            raise PackagingError(f"Referenced action icon ({size}) not found: {icon_path}")

    # Check options page
    options = manifest.get("options_ui", {}).get("page")
    if options and not (extension_dir / options).is_file():
        raise PackagingError(f"Referenced options_ui page not found: {options}")

    return manifest


def collect_runtime_files(extension_dir: Path) -> list[tuple[Path, str]]:
    """Collect runtime files to include in the package, sorted deterministically.

    Returns:
        List of (absolute_path, relative_zip_path) tuples.
    """
    files_to_pack: list[tuple[Path, str]] = []

    for path in extension_dir.rglob("*"):
        if not path.is_file():
            continue

        rel_path = path.relative_to(extension_dir).as_posix()
        parts = rel_path.split("/")

        # Exclude hidden files or dot directories (.git, .vscode, etc.)
        if any(part.startswith(".") for part in parts):
            continue

        # Exclude blacklisted names (e.g. tests, README.md)
        if any(part in EXCLUDED_NAMES for part in parts):
            continue

        # Exclude temporary or swap files
        if path.name.endswith(("~", ".swp", ".bak", ".tmp")):
            continue

        # Root files: only manifest.json is packaged at the top level
        if len(parts) == 1:
            if parts[0] == "manifest.json":
                files_to_pack.append((path, rel_path))
            continue

        # Directory files: must belong to allowed runtime subdirectories
        if parts[0] in ALLOWED_RUNTIME_DIRS:
            files_to_pack.append((path, rel_path))

    # Sort deterministically by relative path
    files_to_pack.sort(key=lambda item: item[1])
    return files_to_pack


def package_extension(
    extension_dir: Path,
    output_dir: Path,
    version_override: str | None = None,
) -> tuple[Path, Path, str]:
    """Create a deterministic zip package and sha256 checksum file.

    Returns:
        Tuple of (zip_path, sha256_path, sha256_hex)
    """
    extension_dir = extension_dir.resolve()
    manifest = validate_manifest(extension_dir)
    version = version_override or manifest["version"]

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_filename = f"phishing-url-detector-extension-{version}.zip"
    zip_path = output_dir / zip_filename
    sha256_path = output_dir / f"{zip_filename}.sha256"

    runtime_files = collect_runtime_files(extension_dir)
    if not runtime_files:
        raise PackagingError(f"No runtime files found to package in {extension_dir}")

    dt = get_deterministic_datetime()

    # Build deterministic zip archive
    # Using strict mode and explicit compression
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in runtime_files:
            file_bytes = file_path.read_bytes()
            zinfo = zipfile.ZipInfo(filename=arcname, date_time=dt)
            # Set standard regular file permissions (rw-r--r--)
            zinfo.external_attr = (0o100644) << 16
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zinfo, file_bytes)

    # Compute SHA-256
    hasher = hashlib.sha256()
    hasher.update(zip_path.read_bytes())
    digest = hasher.hexdigest()

    # Write standard GNU sha256sum format
    sha256_path.write_text(f"{digest}  {zip_filename}\n", encoding="utf-8")

    return zip_path, sha256_path, digest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package Firefox WebExtension deterministically with SHA-256 checksum."
    )
    parser.add_argument(
        "--extension-dir",
        type=Path,
        default=Path("extension"),
        help="Path to extension directory (default: extension)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dist"),
        help="Directory to place the output zip and .sha256 files (default: dist)",
    )
    parser.add_argument(
        "--version-override",
        type=str,
        default=None,
        help="Optional version override for the archive filename",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Suppress informational output",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        zip_path, sha_path, digest = package_extension(
            extension_dir=args.extension_dir,
            output_dir=args.output_dir,
            version_override=args.version_override,
        )
        if not args.quiet:
            print("[✓] Firefox extension packaged successfully:")
            print(f"    Archive:  {zip_path}")
            print(f"    Checksum: {sha_path}")
            print(f"    SHA-256:  {digest}")
        return 0
    except PackagingError as exc:
        print(f"[!] Packaging error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[!] Unexpected error during packaging: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
