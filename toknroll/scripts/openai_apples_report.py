#!/usr/bin/env python3
"""Build apples-to-apples encode report across Java JMH and Python backends.

Inputs:
- Java JMH JSON from OpenAiEncodingBenchmark.encode
- Python CSV from benchmark_model_tokenizers.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SIZE_TO_CHARS = {
    "1k": 1024,
    "32k": 32 * 1024,
}

MODEL_TO_ENCODING = {
    "gpt2": "r50k_base",
    "llama3": "cl100k_base",
    "qwen35": "o200k_base",
    "mistral-tekken": "o200k_base",
}


def load_java(path: Path) -> dict[tuple[str, str, str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)

    for item in payload:
        benchmark = item.get("benchmark", "")
        if not benchmark.endswith(".encode"):
            continue

        params = item.get("params") or {}
        implementation = params.get("implementation")
        encoding = params.get("encoding")
        corpus = params.get("corpus")
        size = params.get("size")
        if not all((implementation, encoding, corpus, size)):
            continue
        if size not in SIZE_TO_CHARS:
            continue

        score = float((item.get("primaryMetric") or {}).get("score", 0.0))
        if score <= 0.0:
            continue

        chars_per_s = score * SIZE_TO_CHARS[size]
        grouped[(encoding, corpus, size, implementation)].append(chars_per_s)

    out: dict[tuple[str, str, str, str], float] = {}
    for key, values in grouped.items():
        out[key] = sum(values) / len(values)
    return out


def load_python(path: Path) -> dict[tuple[str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("section") != "summary":
                continue
            if row.get("op") != "encode":
                continue

            backend = row.get("backend", "")
            if backend not in {"tiktoken", "tokie"}:
                continue

            model = row.get("model", "")
            encoding = MODEL_TO_ENCODING.get(model)
            if encoding is None:
                continue

            corpus = row.get("corpus", "")
            size = row.get("size", "")
            if size not in SIZE_TO_CHARS:
                continue

            chars_per_s = float(row.get("chars_per_s", "0") or "0")
            if chars_per_s <= 0.0:
                continue
            grouped[(encoding, corpus, size, backend)].append(chars_per_s)

    out: dict[tuple[str, str, str, str], float] = {}
    for key, values in grouped.items():
        out[key] = sum(values) / len(values)
    return out


def fmt_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0.0:
        return "n/a"
    return f"{(numerator / denominator):.3f}x"


def build_report(java_rows: dict, py_rows: dict, title: str) -> str:
    keys = sorted(
        {(encoding, corpus, size) for (encoding, corpus, size, _) in java_rows.keys()}
        | {(encoding, corpus, size) for (encoding, corpus, size, _) in py_rows.keys()}
    )

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Encode chars/s by scenario")
    lines.append("")
    lines.append(
        "| encoding | corpus | size | java fast | java jtokkit | py tokie | py tiktoken | tokie / java fast |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")

    per_encoding: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    overall: list[tuple[float, float, float, float]] = []

    for encoding, corpus, size in keys:
        java_fast = java_rows.get((encoding, corpus, size, "fast"), 0.0)
        java_jtokkit = java_rows.get((encoding, corpus, size, "jtokkit"), 0.0)
        py_tokie = py_rows.get((encoding, corpus, size, "tokie"), 0.0)
        py_tiktoken = py_rows.get((encoding, corpus, size, "tiktoken"), 0.0)

        if not any((java_fast, java_jtokkit, py_tokie, py_tiktoken)):
            continue

        lines.append(
            "| "
            + f"{encoding} | {corpus} | {size} | "
            + f"{java_fast:,.0f} | {java_jtokkit:,.0f} | {py_tokie:,.0f} | {py_tiktoken:,.0f} | "
            + f"{fmt_ratio(py_tokie, java_fast)} |"
        )

        per_encoding[encoding].append((java_fast, java_jtokkit, py_tokie, py_tiktoken))
        overall.append((java_fast, java_jtokkit, py_tokie, py_tiktoken))

    lines.append("")
    lines.append("## Aggregated means")
    lines.append("")
    lines.append(
        "| scope | java fast | java jtokkit | py tokie | py tiktoken | tokie / java fast | fast / jtokkit |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    def emit_mean_row(
        label: str, values: list[tuple[float, float, float, float]]
    ) -> None:
        if not values:
            return
        java_fast = sum(x[0] for x in values) / len(values)
        java_jtokkit = sum(x[1] for x in values) / len(values)
        py_tokie = sum(x[2] for x in values) / len(values)
        py_tiktoken = sum(x[3] for x in values) / len(values)
        lines.append(
            "| "
            + f"{label} | {java_fast:,.0f} | {java_jtokkit:,.0f} | {py_tokie:,.0f} | {py_tiktoken:,.0f} | "
            + f"{fmt_ratio(py_tokie, java_fast)} | {fmt_ratio(java_fast, java_jtokkit)} |"
        )

    for encoding in sorted(per_encoding):
        emit_mean_row(encoding, per_encoding[encoding])
    emit_mean_row("overall", overall)

    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Java run can be non-forked and should be treated as indicative, not publication-grade."
    )
    lines.append(
        "- Python and Java use different runtimes; cross-runtime ratios are directional only."
    )
    lines.append("- This report compares encode throughput only.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OpenAI apples-to-apples report")
    parser.add_argument("--java-json", required=True, help="JMH JSON path")
    parser.add_argument("--python-csv", required=True, help="Python benchmark CSV path")
    parser.add_argument("--output", required=True, help="Output markdown path")
    parser.add_argument(
        "--title",
        default="OpenAI Encode Apples-to-Apples Report",
        help="Markdown title",
    )
    args = parser.parse_args()

    java_rows = load_java(Path(args.java_json))
    py_rows = load_python(Path(args.python_csv))
    markdown = build_report(java_rows, py_rows, args.title)
    Path(args.output).write_text(markdown, encoding="utf-8")
    print(f"Wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
