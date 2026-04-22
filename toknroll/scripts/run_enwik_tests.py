#!/usr/bin/env python3
"""Download cached enwik corpora and run tests against them.

Examples:
  python scripts/run_enwik_tests.py
  python scripts/run_enwik_tests.py --datasets enwik8,enwik9 --test Enwik8CorpusCorrectnessTest
  python scripts/run_enwik_tests.py --runner mvn --test Enwik8CorpusCorrectnessTest#familyEncodeParity
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


CORPUS_URLS = {
    "enwik8": "https://www.mattmahoney.net/dc/enwik8.zip",
    "enwik9": "https://www.mattmahoney.net/dc/enwik9.zip",
}

EXPECTED_SIZES = {
    "enwik8": 100_000_000,
    "enwik9": 1_000_000_000,
}

SUITES = {
    "enwik8": "Enwik8CorpusCorrectnessTest",
    "splitters": "FastR50kSplitterTest,FastCl100kSplitterTest,FastO200kSplitterTest,FastQwen35SplitterTest,CommonModelSplitterParityTest,SplitterRegexParityComprehensiveTest,FastSplitterExhaustiveParityTest",
}


def cache_dir() -> Path:
    path = Path.home() / ".cache" / "qxotic" / "tokenizers" / "corpus"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_corpus(name: str, target_dir: Path) -> Path:
    if name not in CORPUS_URLS:
        raise ValueError(f"Unsupported corpus: {name}")
    out = target_dir / name
    expected = EXPECTED_SIZES[name]
    if out.exists() and out.stat().st_size == expected:
        print(f"Using cached {name}: {out}")
        return out

    zip_path = target_dir / f"{name}.zip"
    if not zip_path.exists():
        url = CORPUS_URLS[name]
        print(f"Downloading {name} from {url}")
        urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        if name not in archive.namelist():
            raise RuntimeError(f"Archive {zip_path} does not contain {name}")
        archive.extract(name, target_dir)

    if not out.exists() or out.stat().st_size != expected:
        size = out.stat().st_size if out.exists() else "missing"
        raise RuntimeError(
            f"Invalid {name} size after extract: {size} (expected {expected})"
        )
    return out


def resolve_runner(requested: str) -> str:
    if requested != "auto":
        return requested
    return "mvnd" if shutil.which("mvnd") else "mvn"


def run_tests(
    runner: str,
    test_target: str,
    enwik8_path: Path,
    enwik9_path: Path,
    clean: bool,
    workdir: Path,
) -> int:
    cmd = [
        runner,
        "-pl",
        "toknroll-core",
        "-DfailIfNoTests=false",
        f"-Dtest={test_target}",
        f"-Dtoknroll.enwik8.path={enwik8_path}",
        f"-Dtoknroll.enwik9.path={enwik9_path}",
    ]
    if clean:
        cmd.append("clean")
    cmd.append("test")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(workdir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="enwik8",
        help="Comma-separated corpora to ensure cached (enwik8,enwik9)",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Maven -Dtest target to run",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES.keys()),
        default="enwik8",
        help="Named test suite shortcut",
    )
    parser.add_argument(
        "--runner",
        default="auto",
        choices=["auto", "mvnd", "mvn"],
        help="Maven runner (auto picks mvnd, else mvn)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run clean test instead of test",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Only download/extract corpora, do not run tests",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    runner = resolve_runner(args.runner)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    target = cache_dir()

    paths = {}
    for dataset in datasets:
        paths[dataset] = ensure_corpus(dataset, target)

    enwik8_path = paths.get("enwik8", target / "enwik8")
    enwik9_path = paths.get("enwik9", target / "enwik9")
    if args.no_run:
        print("Done. Cached datasets:")
        for key, value in sorted(paths.items()):
            print(f"  {key}: {value}")
        return 0

    test_target = args.test if args.test else SUITES[args.suite]
    return run_tests(
        runner,
        test_target,
        enwik8_path,
        enwik9_path,
        args.clean,
        repo_root,
    )


if __name__ == "__main__":
    sys.exit(main())
