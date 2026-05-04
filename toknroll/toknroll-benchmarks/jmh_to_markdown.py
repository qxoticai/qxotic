#!/usr/bin/env python3
"""Convert JMH JSON output from ModelTokenizerBenchmark into markdown tables.

Example:
  python toknroll-benchmarks/jmh_to_markdown.py \
    --input target/jmh-model-tokenizers.json \
    --output target/jmh-model-tokenizers.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class RawRow:
    model: str
    corpus: str
    size: str
    implementation: str
    operation: str
    throughput: float
    unit: str


def parse_operation(benchmark_name: str) -> str:
    # e.g. com.qxotic...ModelTokenizerBenchmark.encodeInto -> encodeInto
    return benchmark_name.rsplit(".", 1)[-1]


def load_rows(path: Path) -> List[RawRow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: List[RawRow] = []

    if not isinstance(payload, list):
        raise ValueError("Expected JMH JSON to be a list")

    for item in payload:
        params = item.get("params") or {}
        benchmark = item.get("benchmark", "")
        metric = item.get("primaryMetric") or {}
        score = metric.get("score")

        model = params.get("model")
        corpus = params.get("corpus")
        size = params.get("size")
        implementation = params.get("implementation")

        if (
            not benchmark
            or score is None
            or model is None
            or corpus is None
            or size is None
            or implementation is None
        ):
            continue

        operation = parse_operation(benchmark)
        throughput = float(score)
        unit = "ops/s"
        if operation == "decode":
            secondary = (item.get("secondaryMetrics") or {}).get("decodedTokens") or {}
            decoded_tokens_score = secondary.get("score")
            if decoded_tokens_score is not None:
                throughput = float(decoded_tokens_score)
                unit = "tokens/s"

        rows.append(
            RawRow(
                model=model,
                corpus=corpus,
                size=size,
                implementation=implementation,
                operation=operation,
                throughput=throughput,
                unit=unit,
            )
        )

    if not rows:
        raise ValueError("No compatible ModelTokenizerBenchmark rows found in JSON")

    return rows


def aggregate(rows: Iterable[RawRow]) -> List[RawRow]:
    grouped: Dict[Tuple[str, str, str, str, str, str], List[float]] = defaultdict(list)
    for row in rows:
        key = (
            row.model,
            row.corpus,
            row.size,
            row.implementation,
            row.operation,
            row.unit,
        )
        grouped[key].append(row.throughput)

    out: List[RawRow] = []
    for key, values in grouped.items():
        model, corpus, size, implementation, operation, unit = key
        out.append(
            RawRow(
                model=model,
                corpus=corpus,
                size=size,
                implementation=implementation,
                operation=operation,
                throughput=sum(values) / len(values),
                unit=unit,
            )
        )

    out.sort(
        key=lambda r: (
            r.model,
            r.corpus,
            size_sort_key(r.size),
            r.operation,
            impl_sort_key(r.implementation),
        )
    )
    return out


def impl_sort_key(name: str) -> int:
    order = {
        "reference": 0,
        "jtokkit": 0,
        "jtokkit-adapter": 1,
        "classic": 2,
        "fast": 3,
    }
    return order.get(name, 999)


def size_sort_key(size: str) -> int:
    if size.endswith("k"):
        return int(size[:-1]) * 1024
    return int(size)


def build_markdown(rows: List[RawRow], baseline: str) -> str:
    by_key: Dict[Tuple[str, str, str, str], Dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row.model, row.corpus, row.size, row.operation)
        by_key[key][row.implementation] = row.throughput

    lines: List[str] = []
    lines.append("# JMH Model Tokenizer Benchmark Report")
    lines.append("")
    lines.append("## Raw Throughput")
    lines.append("")
    lines.append(
        "| model | corpus | size | operation | implementation | throughput | unit |"
    )
    lines.append("|---|---|---|---|---|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.model} | {row.corpus} | {row.size} | {row.operation} |"
            f" {row.implementation} | {row.throughput:.2f} | {row.unit} |"
        )

    lines.append("")
    lines.append(f"## Relative Delta vs `{baseline}`")
    lines.append("")
    lines.append("`delta% = ((impl_ops_per_sec / baseline_ops_per_sec) - 1) * 100`")
    lines.append("")
    lines.append("| model | corpus | size | operation | implementation | delta % |")
    lines.append("|---|---|---|---|---|---:|")

    delta_rows: List[Tuple[str, str, str, str, str, float]] = []
    for (model, corpus, size, operation), impl_scores in by_key.items():
        base = impl_scores.get(baseline)
        if base is None or base == 0.0:
            continue
        for impl, score in impl_scores.items():
            if impl == baseline:
                continue
            delta = ((score / base) - 1.0) * 100.0
            delta_rows.append((model, corpus, size, operation, impl, delta))

    delta_rows.sort(
        key=lambda r: (r[0], r[1], size_sort_key(r[2]), r[3], impl_sort_key(r[4]))
    )
    for model, corpus, size, operation, impl, delta in delta_rows:
        lines.append(
            f"| {model} | {corpus} | {size} | {operation} | {impl} | {delta:+.2f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert JMH JSON benchmark output to markdown"
    )
    parser.add_argument("--input", required=True, help="Path to JMH JSON file")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown output file path (default: print to stdout)",
    )
    parser.add_argument(
        "--baseline",
        default="jtokkit",
        help="Implementation name used as baseline for delta calculations",
    )
    args = parser.parse_args()

    rows = aggregate(load_rows(Path(args.input)))
    markdown = build_markdown(rows, baseline=args.baseline)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(markdown, encoding="utf-8")
        print(f"Wrote markdown report: {output_path}")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
