#!/usr/bin/env python3
"""Build apples-to-apples OpenAI report across Java JMH and Python backends.

Inputs:
- Java JMH JSON from OpenAiEncodingBenchmark
- Python CSV from benchmarks/benchmark_model_tokenizers.py
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
}


def load_java(
    path: Path, operations: set[str]
) -> dict[tuple[str, str, str, str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)

    for item in payload:
        benchmark = item.get("benchmark", "")
        operation = benchmark.rsplit(".", 1)[-1]
        if operation not in operations:
            continue

        params = item.get("params") or {}
        implementation = params.get("implementation")
        encoding = params.get("encoding")
        corpus = params.get("corpus")
        size = params.get("size")
        if (
            not isinstance(implementation, str)
            or not isinstance(encoding, str)
            or not isinstance(corpus, str)
            or not isinstance(size, str)
        ):
            continue
        if size not in SIZE_TO_CHARS:
            continue

        score = float((item.get("primaryMetric") or {}).get("score", 0.0))
        if score <= 0.0:
            continue

        if operation == "decode":
            secondary = (item.get("secondaryMetrics") or {}).get("decodedTokens") or {}
            tokens_per_s = float(secondary.get("score", 0.0) or 0.0)
            if tokens_per_s <= 0.0:
                continue
            grouped[(encoding, corpus, size, operation, implementation)].append(
                tokens_per_s
            )
        else:
            chars_per_s = score * SIZE_TO_CHARS[size]
            grouped[(encoding, corpus, size, operation, implementation)].append(
                chars_per_s
            )

    out: dict[tuple[str, str, str, str, str], float] = {}
    for key, values in grouped.items():
        out[key] = sum(values) / len(values)
    return out


def load_python(
    path: Path, operations: set[str], backends: set[str]
) -> dict[tuple[str, str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("section") != "summary":
                continue
            operation = row.get("op", "")
            if operation not in operations:
                continue

            backend = row.get("backend", "")
            if backend not in backends:
                continue

            model = row.get("model", "")
            encoding = MODEL_TO_ENCODING.get(model)
            if encoding is None:
                continue

            corpus = row.get("corpus", "")
            size = row.get("size", "")
            if size not in SIZE_TO_CHARS:
                continue

            if operation == "decode":
                tokens_per_s = float(row.get("tokens_per_s", "0") or "0")
                if tokens_per_s <= 0.0:
                    continue
                grouped[(encoding, corpus, size, operation, backend)].append(
                    tokens_per_s
                )
            else:
                chars_per_s = float(row.get("chars_per_s", "0") or "0")
                if chars_per_s <= 0.0:
                    continue
                grouped[(encoding, corpus, size, operation, backend)].append(
                    chars_per_s
                )

    out: dict[tuple[str, str, str, str, str], float] = {}
    for key, values in grouped.items():
        out[key] = sum(values) / len(values)
    return out


def fmt_ratio(numerator: float, denominator: float) -> str:
    if denominator <= 0.0:
        return "n/a"
    return f"{(numerator / denominator):.3f}x"


def build_report(
    java_rows: dict,
    py_rows: dict,
    title: str,
    operations: list[str],
    py_backends: list[str],
) -> str:
    keys = sorted(
        {
            (encoding, corpus, size)
            for (encoding, corpus, size, _, _) in java_rows.keys()
        }
        | {
            (encoding, corpus, size)
            for (encoding, corpus, size, _, _) in py_rows.keys()
        }
    )

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    for operation in operations:
        unit = "tokens/s" if operation == "decode" else "chars/s"
        lines.append(f"## {operation.capitalize()} {unit} by scenario")
        lines.append("")

        py_columns = " | ".join(f"py {backend}" for backend in py_backends)
        lines.append(
            "| encoding | corpus | size | java fast | java jtokkit | "
            + py_columns
            + " |"
        )
        lines.append("|---|---|---|---:|---:|" + "---:|" * len(py_backends))

        per_encoding: dict[str, list[tuple[float, float, list[float]]]] = defaultdict(
            list
        )
        overall: list[tuple[float, float, list[float]]] = []

        for encoding, corpus, size in keys:
            java_fast = java_rows.get((encoding, corpus, size, operation, "fast"), 0.0)
            java_jtokkit = java_rows.get(
                (encoding, corpus, size, operation, "jtokkit"), 0.0
            )
            py_values = [
                py_rows.get((encoding, corpus, size, operation, backend), 0.0)
                for backend in py_backends
            ]

            if not any([java_fast, java_jtokkit] + py_values):
                continue

            lines.append(
                "| "
                + f"{encoding} | {corpus} | {size} | {java_fast:,.0f} | {java_jtokkit:,.0f} | "
                + " | ".join(f"{value:,.0f}" for value in py_values)
                + " |"
            )

            per_encoding[encoding].append((java_fast, java_jtokkit, py_values))
            overall.append((java_fast, java_jtokkit, py_values))

        lines.append("")
        lines.append(f"## Aggregated means ({operation})")
        lines.append("")
        lines.append(
            "| scope | java fast | java jtokkit | " + py_columns + " | fast / jtokkit |"
        )
        lines.append("|---|---:|---:|" + "---:|" * len(py_backends) + "---:|")

        def emit_mean_row(
            label: str, values: list[tuple[float, float, list[float]]]
        ) -> None:
            if not values:
                return
            java_fast = sum(x[0] for x in values) / len(values)
            java_jtokkit = sum(x[1] for x in values) / len(values)
            py_means = [
                sum(x[2][idx] for x in values) / len(values)
                for idx in range(len(py_backends))
            ]
            lines.append(
                "| "
                + f"{label} | {java_fast:,.0f} | {java_jtokkit:,.0f} | "
                + " | ".join(f"{value:,.0f}" for value in py_means)
                + f" | {fmt_ratio(java_fast, java_jtokkit)} |"
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
    lines.append("- This report compares encode/decode throughput only.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build OpenAI apples-to-apples report")
    parser.add_argument("--java-json", required=True, help="JMH JSON path")
    parser.add_argument("--python-csv", required=True, help="Python benchmark CSV path")
    parser.add_argument("--output", required=True, help="Output markdown path")
    parser.add_argument(
        "--title",
        default="OpenAI Encode/Decode Apples-to-Apples Report",
        help="Markdown title",
    )
    parser.add_argument(
        "--ops",
        default="encode,decode",
        help="comma-separated operations from JMH/Python rows (e.g. encode,decode)",
    )
    parser.add_argument(
        "--python-backends",
        default="tiktoken,tokie,hf-tokenizers,mistral-common",
        help="comma-separated Python backends to include",
    )
    args = parser.parse_args()

    operations = [x.strip() for x in args.ops.split(",") if x.strip()]
    if not operations:
        raise SystemExit("No operations specified via --ops")
    py_backends = [x.strip() for x in args.python_backends.split(",") if x.strip()]
    if not py_backends:
        raise SystemExit("No Python backends specified via --python-backends")

    java_rows = load_java(Path(args.java_json), set(operations))
    py_rows = load_python(Path(args.python_csv), set(operations), set(py_backends))
    markdown = build_report(java_rows, py_rows, args.title, operations, py_backends)
    Path(args.output).write_text(markdown, encoding="utf-8")
    print(f"Wrote report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
