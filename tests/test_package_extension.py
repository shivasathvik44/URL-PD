"""Unit and regression tests for scripts/package_extension.py."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from scripts.package_extension import (
    PackagingError,
    collect_runtime_files,
    main,
    package_extension,
    validate_manifest,
)


@pytest.fixture
def repo_extension_dir() -> Path:
    """Return repository extension directory."""
    return Path(__file__).resolve().parent.parent / "extension"


def test_validate_manifest_valid(repo_extension_dir: Path):
    manifest = validate_manifest(repo_extension_dir)
    assert manifest["manifest_version"] == 3
    assert manifest["name"] == "Phishing URL Detector"
    assert manifest["browser_specific_settings"]["gecko"]["id"]


def test_validate_manifest_missing_file(tmp_path: Path):
    with pytest.raises(PackagingError, match="manifest.json not found"):
        validate_manifest(tmp_path)


def test_validate_manifest_invalid_json(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text("{not-valid-json", encoding="utf-8")
    with pytest.raises(PackagingError, match="Failed to parse manifest.json"):
        validate_manifest(tmp_path)


def test_validate_manifest_wrong_version(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps({"manifest_version": 2}), encoding="utf-8")
    with pytest.raises(PackagingError, match="Expected manifest_version 3"):
        validate_manifest(tmp_path)


def test_validate_manifest_missing_gecko_id(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({
            "manifest_version": 3,
            "name": "Test",
            "version": "1.0",
            "description": "Test",
        }),
        encoding="utf-8",
    )
    with pytest.raises(PackagingError, match="browser_specific_settings.gecko.id"):
        validate_manifest(tmp_path)


def test_validate_manifest_missing_referenced_script(tmp_path: Path):
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps({
            "manifest_version": 3,
            "name": "Test",
            "version": "1.0",
            "description": "Test",
            "browser_specific_settings": {"gecko": {"id": "test@example.com"}},
            "background": {"scripts": ["nonexistent_script.js"]},
        }),
        encoding="utf-8",
    )
    with pytest.raises(PackagingError, match="Referenced background script not found"):
        validate_manifest(tmp_path)


def test_collect_runtime_files_excludes_tests_and_docs(repo_extension_dir: Path):
    collected = collect_runtime_files(repo_extension_dir)
    rel_paths = [rel for _, rel in collected]

    # Verify runtime files are included
    assert "manifest.json" in rel_paths
    assert "background/background.js" in rel_paths
    assert "warning/warning.html" in rel_paths

    # Verify tests and docs are excluded
    for path in rel_paths:
        assert not path.startswith("tests/")
        assert "README.md" not in path


def test_package_extension_success(repo_extension_dir: Path, tmp_path: Path):
    zip_path, sha_path, digest = package_extension(
        extension_dir=repo_extension_dir,
        output_dir=tmp_path,
    )

    assert zip_path.is_file()
    assert sha_path.is_file()
    assert len(digest) == 64

    # Verify archive contents
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "manifest.json" in namelist
        assert "background/background.js" in namelist
        assert "background/api_client.js" in namelist
        assert "popup/popup.html" in namelist
        assert "warning/warning.html" in namelist
        assert "options/options.html" in namelist

        # Must not contain tests or README
        assert not any("tests" in name for name in namelist)
        assert not any("README" in name for name in namelist)

    # Verify sha256 checksum file format
    sha_content = sha_path.read_text(encoding="utf-8")
    expected_line = f"{digest}  {zip_path.name}\n"
    assert sha_content == expected_line

    # Verify digest matches actual zip content
    computed = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    assert computed == digest


def test_package_extension_deterministic_reproducibility(repo_extension_dir: Path, tmp_path: Path):
    dir_a = tmp_path / "build_a"
    dir_b = tmp_path / "build_b"

    _, _, digest_a = package_extension(repo_extension_dir, dir_a)
    _, _, digest_b = package_extension(repo_extension_dir, dir_b)

    assert digest_a == digest_b


def test_package_extension_cli(repo_extension_dir: Path, tmp_path: Path):
    ret = main([
        "--extension-dir",
        str(repo_extension_dir),
        "--output-dir",
        str(tmp_path),
        "-q",
    ])
    assert ret == 0
    assert (tmp_path / "phishing-url-detector-extension-0.5.0.zip").is_file()
    assert (tmp_path / "phishing-url-detector-extension-0.5.0.zip.sha256").is_file()
