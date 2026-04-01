#!/usr/bin/env python3
"""Benchmark model tokenizers across tiktoken, tokie, HF tokenizers, and mistral-common.

This script targets end-to-end tokenizer workloads similar to the Java/JMH model benchmark:
- encode throughput
- decode throughput

Reported throughput metrics include:
- ops/s
- chars/s
- MB/s and MiB/s

Examples:
  python benchmark_model_tokenizers.py
  python benchmark_model_tokenizers.py --duration 2.0 --warmup 0.5 --size 16k
  python benchmark_model_tokenizers.py --models gpt2,qwen35 --backends tiktoken,hf
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence


try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from tokenizers import Tokenizer as HFTokenizer
except Exception:
    HFTokenizer = None

try:
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
except Exception:
    MistralTokenizer = None

try:
    import tokie  # type: ignore[import-not-found]
except Exception:
    tokie = None


SIZE_TO_CHARS = {
    "512": 512,
    "1k": 1024,
    "2k": 2 * 1024,
    "4k": 4 * 1024,
    "8k": 8 * 1024,
    "16k": 16 * 1024,
    "32k": 32 * 1024,
}

CORPUS_SEEDS = {
    "chat": "<|system|>You are helpful.<|user|>Compare tokenizer throughput and latency.",
    "code": "for (int i = 0; i < n; i++) sum += tokenizer.countTokens(lines[i]);",
    "json": '{"id":123,"items":[{"name":"alpha","value":1},{"name":"beta","value":2}],'
    '"ok":true,"meta":{"source":"bench","tags":["a","b"]}}',
    "prose": "Tokenization quality and throughput both matter for long-context systems. "
    "A practical benchmark should include narrative text, technical text, and "
    "structured data to reflect real workloads.",
    "wiki": "In computer science, tokenization is the process of converting a sequence "
    "of characters into a sequence of tokens, often for parsing or language model "
    "preprocessing.",
}


def resize(seed: str, target: int) -> str:
    parts: List[str] = []
    total = 0
    while total < target:
        parts.append(seed)
        parts.append("\n")
        total += len(seed) + 1
    text = "".join(parts)
    return text[:target]


@dataclass
class Adapter:
    backend: str
    model: str
    model_ref: str
    encode: Callable[[str], List[int]]
    decode: Callable[[Sequence[int]], str]


@dataclass
class Result:
    backend: str
    model: str
    corpus: str
    size: str
    op: str
    text_chars: int
    ops_per_s: float
    chars_per_s: float
    mb_per_s: float
    mib_per_s: float


@dataclass
class Summary:
    backend: str
    model: str
    corpus: str
    size: str
    op: str
    text_chars: int
    samples: int
    ops_per_s_mean: float
    ops_per_s_median: float
    ops_per_s_stdev: float
    chars_per_s_mean: float
    chars_per_s_median: float
    chars_per_s_stdev: float
    mb_per_s_mean: float
    mib_per_s_mean: float


def make_tiktoken_adapter(model: str) -> Optional[Adapter]:
    if tiktoken is None:
        return None
    enc_name = {
        "gpt2": "r50k_base",
        "llama3": "cl100k_base",
        "qwen35": "o200k_base",
        "mistral-tekken": "o200k_base",
    }.get(model)
    if enc_name is None:
        return None
    enc = tiktoken.get_encoding(enc_name)
    return Adapter(
        backend="tiktoken",
        model=model,
        model_ref=enc_name,
        encode=lambda text: enc.encode(text),
        decode=lambda ids: enc.decode(ids),
    )


def make_hf_adapter(model: str, revision: str) -> Optional[Adapter]:
    if HFTokenizer is None:
        return None
    candidates = {
        "gpt2": ["gpt2"],
        "llama3": [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "NousResearch/Llama-3.2-1B",
        ],
        "qwen35": ["Qwen/Qwen3.5-0.6B", "Qwen/Qwen3-0.6B"],
        "mistral-tekken": [
            "mistralai/ministral-8b-instruct-2410",
            "mistralai/open-mistral-nemo-2407",
        ],
    }.get(model, [])

    for repo in candidates:
        try:
            try:
                tok = HFTokenizer.from_pretrained(repo, revision=revision)
            except TypeError:
                tok = HFTokenizer.from_pretrained(repo)
            return Adapter(
                backend="hf-tokenizers",
                model=model,
                model_ref=repo,
                encode=lambda text, t=tok: t.encode(text).ids,
                decode=lambda ids, t=tok: t.decode(ids),
            )
        except Exception:
            continue
    return None


def make_tokie_adapter(model: str) -> Optional[Adapter]:
    if tokie is None:
        return None

    candidates = {
        "gpt2": ["gpt2"],
        # OpenAI-compatible encodings for apples-to-apples with tiktoken.
        "llama3": ["Xenova/text-embedding-ada-002"],
        "qwen35": ["Xenova/gpt-4o"],
        "mistral-tekken": ["Xenova/gpt-4o"],
    }.get(model, [])

    for repo in candidates:
        try:
            tok = tokie.Tokenizer.from_pretrained(repo)
            return Adapter(
                backend="tokie",
                model=model,
                model_ref=repo,
                encode=lambda text, t=tok: list(t.encode(text).ids),
                decode=lambda ids, t=tok: t.decode(list(ids)),
            )
        except Exception:
            continue
    return None


def make_mistral_adapter(model: str, revision: str) -> Optional[Adapter]:
    if MistralTokenizer is None:
        return None
    if model != "mistral-tekken":
        return None
    candidates = [
        "mistralai/ministral-8b-instruct-2410",
        "mistralai/open-mistral-nemo-2407",
    ]
    for repo in candidates:
        try:
            mtok = MistralTokenizer.from_hf_hub(repo_id=repo, revision=revision)
            base = mtok.instruct_tokenizer.tokenizer
            return Adapter(
                backend="mistral-common",
                model=model,
                model_ref=repo,
                encode=lambda text, b=base: b.encode(text, bos=False, eos=False),
                decode=lambda ids, b=base: b.decode(list(ids)),
            )
        except Exception:
            continue
    return None


def run_loop(duration_s: float, fn: Callable[[], int]) -> tuple[int, int, float]:
    start = time.perf_counter()
    ops = 0
    units = 0
    while True:
        units += fn()
        ops += 1
        if time.perf_counter() - start >= duration_s:
            break
    elapsed = time.perf_counter() - start
    return ops, units, elapsed


def benchmark_encode(
    adapter: Adapter,
    text: str,
    corpus: str,
    size: str,
    warmup_s: float,
    run_s: float,
) -> Result:
    run_loop(warmup_s, lambda: len(adapter.encode(text)))
    ops, _units, elapsed = run_loop(run_s, lambda: len(adapter.encode(text)))
    chars_per_s = (ops * len(text)) / elapsed
    return Result(
        backend=adapter.backend,
        model=adapter.model,
        corpus=corpus,
        size=size,
        op="encode",
        text_chars=len(text),
        ops_per_s=ops / elapsed,
        chars_per_s=chars_per_s,
        mb_per_s=chars_per_s / 1_000_000.0,
        mib_per_s=chars_per_s / (1024.0 * 1024.0),
    )


def benchmark_decode(
    adapter: Adapter,
    text: str,
    corpus: str,
    size: str,
    warmup_s: float,
    run_s: float,
) -> Result:
    tokens = adapter.encode(text)
    decoded_once = adapter.decode(tokens)
    decoded_chars = max(1, len(decoded_once))
    run_loop(warmup_s, lambda: len(adapter.decode(tokens)))
    ops, _units, elapsed = run_loop(run_s, lambda: len(adapter.decode(tokens)))
    chars_per_s = (ops * decoded_chars) / elapsed
    return Result(
        backend=adapter.backend,
        model=adapter.model,
        corpus=corpus,
        size=size,
        op="decode",
        text_chars=decoded_chars,
        ops_per_s=ops / elapsed,
        chars_per_s=chars_per_s,
        mb_per_s=chars_per_s / 1_000_000.0,
        mib_per_s=chars_per_s / (1024.0 * 1024.0),
    )


def format_table(rows: List[Result]) -> str:
    headers = [
        "backend",
        "model",
        "corpus",
        "size",
        "op",
        "chars",
        "ops/s",
        "chars/s",
        "MB/s",
        "MiB/s",
    ]
    out = ["  ".join(headers)]
    for r in rows:
        out.append(
            f"{r.backend:14}  {r.model:14}  {r.corpus:10}  {r.size:5}  {r.op:6}  {r.text_chars:6d}  "
            f"{r.ops_per_s:10.1f}  {r.chars_per_s:12.1f}  {r.mb_per_s:8.2f}  {r.mib_per_s:8.2f}"
        )
    return "\n".join(out)


def summarize(rows: List[Result]) -> List[Summary]:
    grouped: Dict[tuple[str, str, str, str, str, int], List[Result]] = {}
    for row in rows:
        grouped.setdefault(
            (row.backend, row.model, row.corpus, row.size, row.op, row.text_chars), []
        ).append(row)

    out: List[Summary] = []
    for (backend, model, corpus, size, op, text_chars), samples in grouped.items():
        ops_values = [x.ops_per_s for x in samples]
        chars_values = [x.chars_per_s for x in samples]
        out.append(
            Summary(
                backend=backend,
                model=model,
                corpus=corpus,
                size=size,
                op=op,
                text_chars=text_chars,
                samples=len(samples),
                ops_per_s_mean=statistics.mean(ops_values),
                ops_per_s_median=statistics.median(ops_values),
                ops_per_s_stdev=statistics.stdev(ops_values)
                if len(ops_values) > 1
                else 0.0,
                chars_per_s_mean=statistics.mean(chars_values),
                chars_per_s_median=statistics.median(chars_values),
                chars_per_s_stdev=statistics.stdev(chars_values)
                if len(chars_values) > 1
                else 0.0,
                mb_per_s_mean=statistics.mean([x.mb_per_s for x in samples]),
                mib_per_s_mean=statistics.mean([x.mib_per_s for x in samples]),
            )
        )
    out.sort(key=lambda s: (s.model, s.corpus, s.size, s.op, s.backend))
    return out


def format_summary_table(rows: List[Summary]) -> str:
    headers = [
        "backend",
        "model",
        "corpus",
        "size",
        "op",
        "n",
        "chars",
        "ops/s mean",
        "ops/s med",
        "ops/s sd",
        "chars/s mean",
        "MB/s",
        "MiB/s",
    ]
    out = ["  ".join(headers)]
    for r in rows:
        out.append(
            f"{r.backend:14}  {r.model:14}  {r.corpus:10}  {r.size:5}  {r.op:6}  {r.samples:2d}  {r.text_chars:6d}  "
            f"{r.ops_per_s_mean:10.1f}  {r.ops_per_s_median:10.1f}  {r.ops_per_s_stdev:9.1f}  "
            f"{r.chars_per_s_mean:12.1f}  {r.mb_per_s_mean:8.2f}  {r.mib_per_s_mean:8.2f}"
        )
    return "\n".join(out)


def write_csv(path: str, rows: List[Result], summaries: List[Summary]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "section",
                "backend",
                "model",
                "corpus",
                "size",
                "op",
                "chars",
                "sample",
                "ops_per_s",
                "chars_per_s",
                "mb_per_s",
                "mib_per_s",
            ]
        )
        grouped: Dict[tuple[str, str, str, str, str, int], List[Result]] = {}
        for row in rows:
            grouped.setdefault(
                (row.backend, row.model, row.corpus, row.size, row.op, row.text_chars),
                [],
            ).append(row)
        for key, values in grouped.items():
            backend, model, corpus, size, op, chars = key
            for index, value in enumerate(values, start=1):
                writer.writerow(
                    [
                        "trial",
                        backend,
                        model,
                        corpus,
                        size,
                        op,
                        chars,
                        index,
                        f"{value.ops_per_s:.6f}",
                        f"{value.chars_per_s:.6f}",
                        f"{value.mb_per_s:.6f}",
                        f"{value.mib_per_s:.6f}",
                    ]
                )
        for summary in summaries:
            writer.writerow(
                [
                    "summary",
                    summary.backend,
                    summary.model,
                    summary.corpus,
                    summary.size,
                    summary.op,
                    summary.text_chars,
                    summary.samples,
                    f"{summary.ops_per_s_mean:.6f}",
                    f"{summary.chars_per_s_mean:.6f}",
                    f"{summary.mb_per_s_mean:.6f}",
                    f"{summary.mib_per_s_mean:.6f}",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark model tokenizers in Python")
    parser.add_argument(
        "--duration", type=float, default=2.0, help="measurement duration seconds"
    )
    parser.add_argument(
        "--warmup", type=float, default=0.5, help="warmup duration seconds"
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="repeated trials per benchmark"
    )
    parser.add_argument("--size", choices=sorted(SIZE_TO_CHARS), default="16k")
    parser.add_argument(
        "--sizes",
        default=None,
        help="optional comma-separated sizes overriding --size",
    )
    parser.add_argument("--corpus", choices=sorted(CORPUS_SEEDS), default="chat")
    parser.add_argument(
        "--corpora",
        default=None,
        help="optional comma-separated corpora overriding --corpus",
    )
    parser.add_argument(
        "--revision", default="main", help="HF revision for model loading"
    )
    parser.add_argument(
        "--models",
        default="gpt2,llama3,qwen35,mistral-tekken",
        help="comma-separated model profiles",
    )
    parser.add_argument(
        "--backends",
        default="tiktoken,tokie,hf,mistral-common",
        help="comma-separated backends: tiktoken,tokie,hf,mistral-common",
    )
    parser.add_argument("--csv", default=None, help="optional CSV output path")
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    backends = {b.strip() for b in args.backends.split(",") if b.strip()}
    sizes = (
        [s.strip() for s in args.sizes.split(",") if s.strip()]
        if args.sizes
        else [args.size]
    )
    corpora = (
        [c.strip() for c in args.corpora.split(",") if c.strip()]
        if args.corpora
        else [args.corpus]
    )
    for size in sizes:
        if size not in SIZE_TO_CHARS:
            raise SystemExit(f"Unsupported size in --sizes: {size}")
    for corpus in corpora:
        if corpus not in CORPUS_SEEDS:
            raise SystemExit(f"Unsupported corpus in --corpora: {corpus}")

    builders: Dict[str, Callable[[str], Optional[Adapter]]] = {}
    if "tiktoken" in backends:
        builders["tiktoken"] = make_tiktoken_adapter
    if "tokie" in backends:
        builders["tokie"] = make_tokie_adapter
    if "hf" in backends:
        builders["hf-tokenizers"] = lambda m: make_hf_adapter(m, args.revision)
    if "mistral-common" in backends:
        builders["mistral-common"] = lambda m: make_mistral_adapter(m, args.revision)

    rows: List[Result] = []
    skipped: List[str] = []

    for model in models:
        for backend_name, builder in builders.items():
            adapter = builder(model)
            if adapter is None:
                skipped.append(f"{backend_name}:{model}")
                continue
            for corpus in corpora:
                for size in sizes:
                    text = resize(CORPUS_SEEDS[corpus], SIZE_TO_CHARS[size])
                    print(
                        f"[bench] {adapter.backend}:{adapter.model} ({adapter.model_ref})"
                        f" corpus={corpus} size={size}"
                    )
                    for trial in range(args.repeats):
                        if args.repeats > 1:
                            print(f"  trial {trial + 1}/{args.repeats}")
                        rows.append(
                            benchmark_encode(
                                adapter,
                                text,
                                corpus,
                                size,
                                args.warmup,
                                args.duration,
                            )
                        )
                        try:
                            rows.append(
                                benchmark_decode(
                                    adapter,
                                    text,
                                    corpus,
                                    size,
                                    args.warmup,
                                    args.duration,
                                )
                            )
                        except BaseException as exc:
                            skipped.append(
                                f"decode-failed:{adapter.backend}:{adapter.model}:{corpus}:{size}:{exc}"
                            )

    if not rows:
        print("No benchmarks ran. Verify optional dependencies and model access.")
        return 1

    summaries = summarize(rows)

    print("\nSummary")
    print("=======")
    print(format_summary_table(summaries))

    if args.repeats == 1:
        print("\nRaw")
        print("===")
        print(format_table(rows))

    if args.csv:
        write_csv(args.csv, rows, summaries)
        print(f"\nCSV written: {args.csv}")

    if skipped:
        print("\nSkipped")
        print("=======")
        for item in skipped:
            print(f"- {item}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
