#!/usr/bin/env python3
"""Download the enwik8 corpus into the toknroll corpus cache, where the tests look for it.

The location mirrors TestCachePaths.corpusDir(): TOKNROLL_CACHE_ROOT if set, else the OS cache
directory under qxotic/toknroll, plus "corpus". A present, complete corpus is left alone.
"""
import argparse
import os
import pathlib
import platform
import shutil
import urllib.request
import zipfile

# Sources in order of preference; both serve the same archive. A host may answer some networks
# (GitHub runners, for one) with an HTML page and status 200, so a download is trusted only
# once its zip signature and exact length check out.
URLS = (
    "https://www.mattmahoney.net/dc/enwik8.zip",
    "https://data.deepai.org/enwik8.zip",
)
ZIP_SIZE = 36_445_475
CORPUS_SIZE = 100_000_000


def fetch(url: str, target: pathlib.Path):
    """Download url into target. Returns None on success, else why this source was rejected."""
    request = urllib.request.Request(url, headers={"User-Agent": "qxotic-toknroll-fixtures"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
            head = response.read(4)
            if head != b"PK\x03\x04":
                return f"not a zip (starts with {head!r}, {response.headers.get('Content-Type')})"
            out.write(head)
            shutil.copyfileobj(response, out, 1 << 20)
    except OSError as exc:
        return str(exc)
    size = target.stat().st_size
    return None if size == ZIP_SIZE else f"{size} bytes, expected {ZIP_SIZE}"


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
    corpus = out / "enwik8"
    if corpus.is_file() and corpus.stat().st_size == CORPUS_SIZE:
        print(f"present {corpus}")
        return
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "enwik8.zip.part"
    rejected = []
    for url in URLS:
        problem = fetch(url, archive_path)
        if problem is None:
            break
        rejected.append(f"{url}: {problem}")
    else:
        archive_path.unlink(missing_ok=True)
        raise SystemExit("ERROR: no source delivered enwik8.zip\n  " + "\n  ".join(rejected))
    extract(archive_path, corpus)
    print(f"downloaded {corpus} ({CORPUS_SIZE} bytes)")


def extract(archive_path: pathlib.Path, corpus: pathlib.Path) -> None:
    """Unpack the enwik8 entry next to the archive and move it into place; the archive is removed."""
    part = corpus.with_suffix(".part")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            entry = archive.getinfo("enwik8")
            if entry.file_size != CORPUS_SIZE:
                raise SystemExit(f"ERROR: archive holds {entry.file_size} bytes, expected {CORPUS_SIZE}")
            with archive.open(entry) as source, part.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1 << 20)  # zipfile verifies the CRC at end of entry
    except (zipfile.BadZipFile, KeyError) as exc:
        part.unlink(missing_ok=True)
        raise SystemExit(f"ERROR: {archive_path}: {exc}") from exc
    finally:
        archive_path.unlink(missing_ok=True)
    part.replace(corpus)


if __name__ == "__main__":
    main()
