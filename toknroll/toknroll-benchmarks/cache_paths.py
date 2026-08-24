#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path


def resolve_under_test_artifacts(*parts: str, override: str | None = None) -> Path:
    root = _resolve_test_artifacts_root(override)
    path = root
    for part in parts:
        path = path / part
    return path


def corpus_dir(override: str | None = None) -> Path:
    """Downloaded corpora: <cache root>/corpus, falling back to the legacy
    ~/.cache/qxotic/tokenizers/corpus location when that is where corpora already live.
    override replaces the cache root (mirrors -Dtoknroll.cache.root / TOKNROLL_CACHE_ROOT)."""
    if (override or "").strip():
        return Path(override.strip()).expanduser().resolve() / "corpus"
    legacy = Path.home() / ".cache" / "qxotic" / "tokenizers" / "corpus"
    root = _resolve_cache_root() / "corpus"
    return legacy if legacy.exists() and not root.exists() else root


def _resolve_test_artifacts_root(override: str | None = None) -> Path:
    configured = (override or "").strip() or os.environ.get("TOKNROLL_TEST_CACHE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _resolve_cache_root() / "test-artifacts"


def _resolve_cache_root() -> Path:
    configured = os.environ.get("TOKNROLL_CACHE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return _os_cache_dir() / "qxotic" / "toknroll"


def _os_cache_dir() -> Path:
    home = Path.home()
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data)
        app_data = os.environ.get("APPDATA", "").strip()
        if app_data:
            return Path(app_data)
        return home / "AppData" / "Local"

    if sys_platform_contains("darwin"):
        return home / "Library" / "Caches"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        return Path(xdg_cache_home)
    return home / ".cache"


def sys_platform_contains(name: str) -> bool:
    import sys

    return name in sys.platform.lower()
