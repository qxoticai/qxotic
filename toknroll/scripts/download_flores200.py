#!/usr/bin/env python3
"""Download the FLORES-200 devtest sentences into the toknroll corpus cache, as one file, flores200.

The file holds the first 200 sentences of every one of the 204 languages, about 8 MB in every
script the tokenizers see: the multilingual half of libtoknroll's PGO training (PgoTraining).
The location is download_enwik8.py's; a present corpus is left alone. The sentences are
Meta's FLORES-200, CC BY-SA 4.0.
"""
import argparse
import io
import pathlib
import shutil
import sys
import tarfile
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from download_enwik8 import default_corpus_dir  # noqa: E402

URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
ARCHIVE_SIZE = 25_585_843
LANGUAGES = 204
SENTENCES = 200  # of the 1012 per language


def fetch(url: str, target: pathlib.Path):
    """Download url into target. Returns None on success, else why the download was rejected."""
    request = urllib.request.Request(url, headers={"User-Agent": "qxotic-toknroll-fixtures"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
            head = response.read(2)
            if head != b"\x1f\x8b":
                return f"not a gzip (starts with {head!r}, {response.headers.get('Content-Type')})"
            out.write(head)
            shutil.copyfileobj(response, out, 1 << 20)
    except OSError as exc:
        return str(exc)
    size = target.stat().st_size
    return None if size == ARCHIVE_SIZE else f"{size} bytes, expected {ARCHIVE_SIZE}"


def extract(archive_path: pathlib.Path, corpus: pathlib.Path) -> None:
    """The devtest files' first sentences, language by language in name order, into corpus."""
    part = corpus.with_suffix(".part")
    texts = {}
    try:
        with tarfile.open(archive_path) as archive:
            for member in archive:
                name = pathlib.PurePosixPath(member.name)
                if member.isfile() and name.parent.name == "devtest" and name.suffix == ".devtest":
                    with archive.extractfile(member) as source:
                        lines = io.TextIOWrapper(source, encoding="utf-8").readlines()
                    texts[name.stem] = "".join(lines[:SENTENCES])
    except tarfile.TarError as exc:
        raise SystemExit(f"ERROR: {archive_path}: {exc}") from exc
    finally:
        archive_path.unlink(missing_ok=True)
    if len(texts) != LANGUAGES:
        raise SystemExit(f"ERROR: archive holds {len(texts)} languages, expected {LANGUAGES}")
    with part.open("w", encoding="utf-8", newline="\n") as sink:
        for language in sorted(texts):
            sink.write(texts[language])
    part.replace(corpus)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the FLORES-200 devtest sentences")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write flores200 into (default: the toknroll corpus cache)",
    )
    args = parser.parse_args()
    out = pathlib.Path(args.output_dir) if args.output_dir else default_corpus_dir()
    corpus = out / "flores200"
    if corpus.is_file():
        print(f"present {corpus}")
        return
    out.mkdir(parents=True, exist_ok=True)
    archive_path = out / "flores200.tar.gz.part"
    problem = fetch(URL, archive_path)
    if problem is not None:
        archive_path.unlink(missing_ok=True)
        raise SystemExit(f"ERROR: {URL}: {problem}")
    extract(archive_path, corpus)
    print(f"downloaded {corpus} ({corpus.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
