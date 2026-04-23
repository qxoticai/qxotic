#!/usr/bin/env python3
"""Download enwik corpora and run Wiki JMH benchmarks.

Examples:
  python benchmarks/run_wiki_benchmarks.py --download-only --datasets enwik8
  python benchmarks/run_wiki_benchmarks.py --corpus enwik8 --parallel true --modes encode,decode
  python benchmarks/run_wiki_benchmarks.py --corpus enwik8 --parallel false --encodings r50k_base,cl100k_base,o200k_base
"""

from __future__ import annotations

import argparse
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


def cache_dir() -> Path:
    out = Path.home() / ".cache" / "qxotic" / "tokenizers" / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    return out


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


def ensure_jmh_classpath(core_dir: Path) -> Path:
    classpath_file = core_dir / "target" / "jmh-test-classpath.txt"
    cmd = [
        "mvn",
        "-q",
        "-DskipTests",
        "test-compile",
        "dependency:build-classpath",
        "-Dmdep.includeScope=test",
        "-Dmdep.outputFile=target/jmh-test-classpath.txt",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(core_dir))
    if not classpath_file.exists():
        raise RuntimeError(f"Missing classpath file: {classpath_file}")
    return classpath_file


def build_benchmark_pattern(modes: list[str]) -> str:
    if len(modes) == 1:
        return f"com.qxotic.toknroll.benchmarks.WikiEncodingBenchmark.{modes[0]}"
    joined = "|".join(modes)
    return f"com.qxotic.toknroll.benchmarks.WikiEncodingBenchmark.({joined})"


def run_jmh(
    core_dir: Path,
    classpath_file: Path,
    corpus: str,
    parallel: bool,
    encodings: str,
    modes: list[str],
    warmup: int,
    iterations: int,
    forks: int,
) -> int:
    classpath = classpath_file.read_text(encoding="utf-8").strip()
    full_classpath = f"{classpath}:{core_dir / 'target' / 'test-classes'}:{core_dir / 'target' / 'classes'}"
    benchmark = build_benchmark_pattern(modes)

    cmd = [
        "java",
        "-cp",
        full_classpath,
        "org.openjdk.jmh.Main",
        benchmark,
        "-p",
        f"corpus={corpus}",
        "-p",
        f"parallel={'true' if parallel else 'false'}",
        "-p",
        f"encoding={encodings}",
        "-wi",
        str(warmup),
        "-i",
        str(iterations),
        "-f",
        str(forks),
        "-tu",
        "s",
        "-bm",
        "avgt",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(core_dir))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="enwik8",
        help="Comma-separated corpora to download and unpack (enwik8,enwik9)",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download and unpack corpora",
    )
    parser.add_argument(
        "--corpus",
        default="enwik8",
        choices=["enwik8", "enwik9"],
        help="Corpus used by WikiEncodingBenchmark",
    )
    parser.add_argument(
        "--parallel",
        default="true",
        choices=["true", "false"],
        help="Set WikiEncodingBenchmark parallel parameter",
    )
    parser.add_argument(
        "--encodings",
        default="r50k_base,cl100k_base,o200k_base",
        help="Comma-separated tokenizer encodings",
    )
    parser.add_argument(
        "--modes",
        default="encode,decode",
        help="Comma-separated benchmark modes (encode,decode)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="JMH warmup iterations")
    parser.add_argument(
        "--iterations", type=int, default=5, help="JMH measurement iterations"
    )
    parser.add_argument("--forks", type=int, default=1, help="JMH forks")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    core_dir = repo_root / "toknroll-core"
    target = cache_dir()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for dataset in datasets:
        ensure_corpus(dataset, target)

    if args.download_only:
        print("Done. Cached datasets:")
        for dataset in datasets:
            print(f"  {dataset}: {target / dataset}")
        return 0

    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    allowed_modes = {"encode", "decode"}
    invalid = [m for m in modes if m not in allowed_modes]
    if invalid:
        raise SystemExit(f"Unsupported modes: {invalid}. Allowed: encode,decode")

    ensure_corpus(args.corpus, target)
    classpath_file = ensure_jmh_classpath(core_dir)
    return run_jmh(
        core_dir=core_dir,
        classpath_file=classpath_file,
        corpus=args.corpus,
        parallel=args.parallel == "true",
        encodings=args.encodings,
        modes=modes,
        warmup=args.warmup,
        iterations=args.iterations,
        forks=args.forks,
    )


if __name__ == "__main__":
    sys.exit(main())
