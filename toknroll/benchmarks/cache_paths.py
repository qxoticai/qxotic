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


def _resolve_test_artifacts_root(override: str | None) -> Path:
    configured = (override or "").strip() or os.environ.get("TOKNROLL_TEST_CACHE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    home = Path.home()
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "qxotic" / "toknroll" / "test-artifacts"
        app_data = os.environ.get("APPDATA", "").strip()
        if app_data:
            return Path(app_data) / "qxotic" / "toknroll" / "test-artifacts"
        return home / "AppData" / "Local" / "qxotic" / "toknroll" / "test-artifacts"

    if sys_platform_contains("darwin"):
        return home / "Library" / "Caches" / "qxotic" / "toknroll" / "test-artifacts"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    if xdg_cache_home:
        return Path(xdg_cache_home) / "qxotic" / "toknroll" / "test-artifacts"
    return home / ".cache" / "qxotic" / "toknroll" / "test-artifacts"


def sys_platform_contains(name: str) -> bool:
    import sys

    return name in sys.platform.lower()
