#!/usr/bin/env python3
"""Download the enwik8 corpus into the toknroll corpus cache, where the tests look for it.

The location mirrors TestCachePaths.corpusDir(): TOKNROLL_CACHE_ROOT if set, else the OS cache
directory under qxotic/toknroll, plus "corpus". Present and complete files are left alone.
"""
import argparse
import os
import pathlib
import platform
import urllib.error
import urllib.request
import zipfile

URL = "https://www.mattmahoney.net/dc/enwik8.zip"
SIZE = 100_000_000


def default_corpus_dir() -> pathlib.Path:
    root = os.environ.get("TOKNROLL_CACHE_ROOT")
    if root:
        return pathlib.Path(root) / "corpus"
    home = pathlib.Path.home()
    system = platform.system()
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        base = pathlib.Path(local) if local else home / "AppData" / "Local"
    elif system == "Darwin":
        base = home / "Library" / "Caches"
    else:
        base = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache")
    return base / "qxotic" / "toknroll" / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the enwik8 test corpus")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write enwik8 into (default: the toknroll corpus cache)",
    )
    args = parser.parse_args()
    out = pathlib.Path(args.output_dir) if args.output_dir else default_corpus_dir()
    target = out / "enwik8"
    if target.is_file() and target.stat().st_size == SIZE:
        print(f"present {target}")
        return
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "enwik8.zip.part"
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, zip_path.open("wb") as sink:
            while chunk := response.read(1 << 20):
                sink.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        raise SystemExit(f"ERROR: failed to download {URL}: {exc}") from exc
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("enwik8") as source, (out / "enwik8.part").open("wb") as sink:
            while chunk := source.read(1 << 20):
                sink.write(chunk)
    zip_path.unlink()
    part = out / "enwik8.part"
    if part.stat().st_size != SIZE:
        part.unlink()
        raise SystemExit(f"ERROR: enwik8 is {part.stat().st_size} bytes, expected {SIZE}")
    part.replace(target)
    print(f"downloaded {target} ({SIZE} bytes)")


if __name__ == "__main__":
    main()
