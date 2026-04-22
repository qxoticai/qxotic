#!/usr/bin/env python3
"""Run tokie encode/decode benchmarks on enwik8 corpus.

Examples:
  python scripts/bench_tokie.py
  python scripts/bench_tokie.py --encoding gpt2
  python scripts/bench_tokie.py --encoding tokiers/cl100k --warmup 5 --measure 20
  python scripts/bench_tokie.py --all
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path


def load_enwik8() -> str:
    path = Path.home() / ".cache" / "qxotic" / "tokenizers" / "corpus" / "enwik8"
    if not path.exists():
        raise FileNotFoundError(
            f"enwik8 not found at {path}.\n"
            "Download it first:\n"
            "  python scripts/run_wiki_benchmarks.py --download-only --datasets enwik8"
        )
    return path.read_text(encoding="utf-8")


def benchmark_encode_decode(
    tok, text: str, warmup: int, measure: int
) -> tuple[float, float, int]:
    encode_samples = []
    ids = None
    for i in range(warmup + measure):
        t0 = time.perf_counter()
        ids = tok.encode(text).ids
        dt = time.perf_counter() - t0
        if i >= warmup:
            encode_samples.append(dt)

    assert ids is not None

    decode_samples = []
    for i in range(warmup + measure):
        t0 = time.perf_counter()
        _ = tok.decode(ids) or ""
        dt = time.perf_counter() - t0
        if i >= warmup:
            decode_samples.append(dt)

    enc_avg = statistics.mean(encode_samples)
    dec_avg = statistics.mean(decode_samples)
    return enc_avg, dec_avg, len(ids)


def run_single(repo: str, text: str, warmup: int, measure: int) -> None:
    try:
        import tokie
    except ImportError:
        raise SystemExit("tokie not installed. Run: pip install tokie")

    tok = tokie.Tokenizer.from_pretrained(repo)
    enc_avg, dec_avg, token_count = benchmark_encode_decode(tok, text, warmup, measure)

    print(f"repo={repo} tokens={token_count}")
    print(f"  encode_avg_s={enc_avg:.6f}")
    print(f"  decode_avg_s={dec_avg:.6f}")
    print(f"  encode_ops_per_s={1.0 / enc_avg:.6f}")
    print(f"  decode_ops_per_s={1.0 / dec_avg:.6f}")
    print(f"  encode_tokens_per_s={token_count / enc_avg:.2f}")
    print(f"  decode_tokens_per_s={token_count / dec_avg:.2f}")


def run_all(text: str, warmup: int, measure: int) -> None:
    try:
        import tokie
    except ImportError:
        raise SystemExit("tokie not installed. Run: pip install tokie")

    repos = [
        ("r50k_base", "gpt2"),
        ("cl100k_base", "tokiers/cl100k"),
        ("o200k_base", "tokiers/o200k"),
    ]

    for enc_name, repo in repos:
        tok = tokie.Tokenizer.from_pretrained(repo)
        enc_avg, dec_avg, token_count = benchmark_encode_decode(
            tok, text, warmup, measure
        )

        print(f"encoding={enc_name} repo={repo} tokens={token_count}")
        print(f"  encode_avg_s={enc_avg:.6f}")
        print(f"  decode_avg_s={dec_avg:.6f}")
        print(f"  encode_ops_per_s={1.0 / enc_avg:.6f}")
        print(f"  decode_ops_per_s={1.0 / dec_avg:.6f}")
        print(f"  encode_tokens_per_s={token_count / enc_avg:.2f}")
        print(f"  decode_tokens_per_s={token_count / dec_avg:.2f}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoding",
        default="gpt2",
        help="HuggingFace repo or preset for tokie (default: gpt2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark all three encodings (gpt2, tokiers/cl100k, tokiers/o200k)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--measure",
        type=int,
        default=15,
        help="Number of measured iterations",
    )
    args = parser.parse_args()

    text = load_enwik8()

    if args.all:
        run_all(text, args.warmup, args.measure)
    else:
        run_single(args.encoding, text, args.warmup, args.measure)

    return 0


if __name__ == "__main__":
    sys.exit(main())
