#!/usr/bin/env python

"""Tests for `hf_lora_loader` helper utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.hf_lora_loader.nodes import HF_SUBDIR, ensure_hf_lora_file


def _mock_downloader(content: bytes):
    def _download(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        target = local_dir / kwargs["filename"]
        target.write_bytes(content)
        return str(target)

    return _download


def test_ensure_hf_lora_file_downloads_when_missing(tmp_path):
    repo_id = "author/repo"
    rel_path = ensure_hf_lora_file(
        repo_id=repo_id,
        filename="model.safetensors",
        lora_root=str(tmp_path),
        downloader=_mock_downloader(b"file-bytes"),
    )

    expected_rel = f"{HF_SUBDIR}/author__repo/model.safetensors"
    assert rel_path == expected_rel

    stored = Path(tmp_path, rel_path)
    assert stored.read_bytes() == b"file-bytes"


def test_ensure_hf_lora_file_skips_when_present(tmp_path):
    base_dir = tmp_path / HF_SUBDIR / "owner__model"
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = base_dir / "lora.safetensors"
    existing.write_bytes(b"cached")

    called = False

    def _tracking_downloader(**kwargs):  # pragma: no cover - guarded by assertion
        nonlocal called
        called = True
        return str(existing)

    rel_path = ensure_hf_lora_file(
        repo_id="owner/model",
        filename="lora.safetensors",
        lora_root=str(tmp_path),
        downloader=_tracking_downloader,
    )

    assert rel_path.endswith("lora.safetensors")
    assert existing.read_bytes() == b"cached"
    assert not called


def test_ensure_hf_lora_file_checksum_mismatch_raises(tmp_path):
    expected_hash = hashlib.sha256(b"expected").hexdigest()

    with pytest.raises(ValueError):
        ensure_hf_lora_file(
            repo_id="org/repo",
            filename="bad.safetensors",
            lora_root=str(tmp_path),
            downloader=_mock_downloader(b"unexpected"),
            expected_sha256=expected_hash,
        )
