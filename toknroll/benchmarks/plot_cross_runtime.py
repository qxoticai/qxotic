#!/usr/bin/env python3
"""Generate polished cross-runtime benchmark charts.

Inputs:
- Java JMH JSON from ModelTokenizerBenchmark
- Python CSV from benchmarks/benchmark_model_tokenizers.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


SIZE_TO_CHARS = {
    "1k": 1024,
    "2k": 2 * 1024,
    "4k": 4 * 1024,
    "8k": 8 * 1024,
    "16k": 16 * 1024,
    "32k": 32 * 1024,
}

SIZE_ORDER = ["1k", "2k", "4k", "8k", "16k", "32k"]
IMPL_ORDER = ["fast", "classic"]

PREFERRED_COMPARISONS = [
    ("gpt2", "tiktoken", "Tok'n'Roll vs TikToken"),
    ("gpt2", "tokie", "Tok'n'Roll vs Tokie"),
    ("gpt2", "hf-tokenizers", "Tok'n'Roll vs HuggingFace"),
    ("llama3", "hf-tokenizers", "Tok'n'Roll vs HuggingFace"),
    ("qwen35", "hf-tokenizers", "Tok'n'Roll vs HuggingFace"),
    ("mistral-tekken", "hf-tokenizers", "Tok'n'Roll vs HuggingFace"),
    ("mistral-tekken", "mistral-common", "Tok'n'Roll vs Mistral"),
]

COLORS = {
    "fast": "#0f766e",
    "classic": "#0891b2",
    "python": "#334155",
}


def metric_for_operation(operation: str) -> str:
    return "ops_per_s" if operation == "decode" else "chars_per_s"


def load_java(path: Path) -> dict[tuple[str, str, str, str, str], float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    for item in payload:
        params = item.get("params") or {}
        implementation = params.get("implementation")
        model = params.get("model")
        corpus = params.get("corpus")
        size = params.get("size")
        if (
            implementation not in {"fast", "classic"}
            or not isinstance(model, str)
            or not isinstance(corpus, str)
            or not isinstance(size, str)
            or size not in SIZE_TO_CHARS
        ):
            continue
        operation = str(item.get("benchmark", "")).rsplit(".", 1)[-1]
        if operation not in {"encode", "decode"}:
            continue
        primary = item.get("primaryMetric") or {}
        score = float(primary.get("score", 0.0) or 0.0)
        if operation == "encode":
            score *= SIZE_TO_CHARS[size]
        if score <= 0.0:
            continue
        grouped[(implementation, model, corpus, size, operation)].append(score)
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def load_python(path: Path) -> dict[tuple[str, str, str, str, str], float]:
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("section") != "summary":
                continue
            backend = row.get("backend", "")
            model = row.get("model", "")
            corpus = row.get("corpus", "")
            size = row.get("size", "")
            operation = row.get("op", "")
            if size not in SIZE_TO_CHARS or operation not in {"encode", "decode"}:
                continue
            value = float(row.get(metric_for_operation(operation), "0") or "0")
            if value <= 0.0:
                continue
            grouped[(backend, model, corpus, size, operation)].append(value)
    return {k: sum(v) / len(v) for k, v in grouped.items()}


def discover_comparisons(
    java_rows: dict[tuple[str, str, str, str, str], float],
    python_rows: dict[tuple[str, str, str, str, str], float],
) -> list[tuple[str, str, str]]:
    java_models = {model for _, model, _, _, _ in java_rows.keys()}
    py_pairs = {(model, backend) for backend, model, _, _, _ in python_rows.keys()}
    out: list[tuple[str, str, str]] = []
    for model, backend, title in PREFERRED_COMPARISONS:
        if model in java_models and (model, backend) in py_pairs:
            out.append((model, backend, title))
    return out


def aggregate_by_size(
    java_rows: dict[tuple[str, str, str, str, str], float],
    python_rows: dict[tuple[str, str, str, str, str], float],
    model: str,
    backend: str,
    operation: str,
) -> list[tuple[str, dict[str, float], float]]:
    rows: list[tuple[str, dict[str, float], float]] = []
    for size in SIZE_ORDER:
        py_by_corpus = {
            corpus: value
            for (
                p_backend,
                p_model,
                corpus,
                p_size,
                p_operation,
            ), value in python_rows.items()
            if p_backend == backend
            and p_model == model
            and p_size == size
            and p_operation == operation
        }
        if not py_by_corpus:
            continue
        impl_values: dict[str, float] = {}
        matched_corpora: set[str] = set()
        for implementation in IMPL_ORDER:
            java_by_corpus = {
                corpus: value
                for (
                    j_implementation,
                    j_model,
                    corpus,
                    j_size,
                    j_operation,
                ), value in java_rows.items()
                if j_implementation == implementation
                and j_model == model
                and j_size == size
                and j_operation == operation
            }
            common = sorted(set(java_by_corpus.keys()) & set(py_by_corpus.keys()))
            if not common:
                continue
            impl_values[implementation] = sum(java_by_corpus[c] for c in common) / len(
                common
            )
            matched_corpora.update(common)
        if not impl_values or not matched_corpora:
            continue
        py_value = sum(py_by_corpus[c] for c in sorted(matched_corpora)) / len(
            matched_corpora
        )
        rows.append((size, impl_values, py_value))
    return rows


def aggregate_ratio_matrix(
    java_rows: dict[tuple[str, str, str, str, str], float],
    python_rows: dict[tuple[str, str, str, str, str], float],
    model: str,
    backend: str,
    operation: str,
    implementation: str,
) -> tuple[list[str], list[str], list[list[float]]]:
    corpora = sorted(
        {
            corpus
            for (
                _impl,
                j_model,
                corpus,
                _size,
                j_operation,
            ) in java_rows.keys()
            if j_model == model and j_operation == operation and _impl == implementation
        }
        & {
            corpus
            for (
                p_backend,
                p_model,
                corpus,
                _size,
                p_operation,
            ) in python_rows.keys()
            if p_backend == backend and p_model == model and p_operation == operation
        }
    )
    matrix: list[list[float]] = []
    for corpus in corpora:
        row: list[float] = []
        for size in SIZE_ORDER:
            j = java_rows.get((implementation, model, corpus, size, operation), 0.0)
            p = python_rows.get((backend, model, corpus, size, operation), 0.0)
            row.append((j / p) if p > 0.0 else 0.0)
        matrix.append(row)
    return corpora, SIZE_ORDER[:], matrix


def slugify(name: str) -> str:
    return (
        name.lower()
        .replace("tok'n'roll", "toknroll")
        .replace(" ", "-")
        .replace("/", "-")
    )


def plot_pair_bars(
    output_dir: Path,
    model: str,
    backend: str,
    title: str,
    operation: str,
    points: list[tuple[str, dict[str, float], float]],
) -> list[tuple[str, str, str, str, float]]:
    labels = [size for size, _, _ in points]
    py_values = [py_value for _, _, py_value in points]
    impls = [i for i in IMPL_ORDER if any(i in impl_map for _, impl_map, _ in points)]
    ratio_rows: list[tuple[str, str, str, str, float]] = []

    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=170)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f1f5f9")

    x_positions = list(range(len(labels)))
    n_series = len(impls) + 1
    bar_width = 0.82 / n_series
    start = -0.41 + (bar_width / 2)

    for idx, implementation in enumerate(impls):
        values = [impl_map.get(implementation, 0.0) for _, impl_map, _ in points]
        offsets = [x + start + idx * bar_width for x in x_positions]
        ax.bar(
            offsets,
            values,
            width=bar_width,
            label=f"Tok'n'Roll {implementation}",
            color=COLORS[implementation],
            edgecolor="#0f172a",
            linewidth=0.8,
        )
        ratios = [j / p for j, p in zip(values, py_values) if p > 0.0 and j > 0.0]
        if ratios:
            ratio_rows.append(
                (model, backend, operation, implementation, sum(ratios) / len(ratios))
            )

    py_offsets = [x + start + len(impls) * bar_width for x in x_positions]
    ax.bar(
        py_offsets,
        py_values,
        width=bar_width,
        label=f"{backend} (Python)",
        color=COLORS["python"],
        edgecolor="#0f172a",
        linewidth=0.8,
    )

    metric_label = "ops/s" if operation == "decode" else "chars/s"
    subtitle = ", ".join(f"{impl}: {ratio:.2f}x" for _, _, _, impl, ratio in ratio_rows)
    ax.set_title(f"{title} | {model} | {operation}", fontsize=15, weight="bold", pad=14)
    if subtitle:
        ax.text(
            0.995,
            1.01,
            subtitle,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#0f172a",
        )
    ax.set_xlabel("Input length", fontsize=11)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)

    output_path = output_dir / f"{slugify(title)}-{model}-{backend}-{operation}.png"
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return ratio_rows


def plot_ratio_heatmap(
    output_dir: Path,
    model: str,
    backend: str,
    operation: str,
    implementation: str,
    corpora: list[str],
    sizes: list[str],
    matrix: list[list[float]],
) -> None:
    if not corpora or not matrix:
        return
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=170)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#f8fafc")
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=2.4)

    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            if value <= 0.0:
                continue
            ax.text(
                c,
                r,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#0f172a",
            )

    ax.set_title(
        f"Speedup Matrix | {model} | {operation} | Tok'n'Roll {implementation} / {backend}",
        fontsize=12,
        weight="bold",
    )
    ax.set_xticks(list(range(len(sizes))))
    ax.set_xticklabels(sizes)
    ax.set_yticks(list(range(len(corpora))))
    ax.set_yticklabels(corpora)
    ax.set_xlabel("Input length")
    ax.set_ylabel("Corpus")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("speedup ratio")
    fig.tight_layout()
    fig.savefig(
        output_dir
        / f"heatmap-{model}-{backend}-{operation}-toknroll-{implementation}.png"
    )
    plt.close(fig)


def plot_summary_speedups(
    output_dir: Path,
    ratio_rows: list[tuple[str, str, str, str, float]],
) -> None:
    if not ratio_rows:
        return
    sorted_rows = sorted(ratio_rows, key=lambda row: row[4], reverse=True)
    labels = [f"{m}:{b}:{op}:{impl}" for m, b, op, impl, _ in sorted_rows]
    values = [v for _, _, _, _, v in sorted_rows]

    fig, ax = plt.subplots(figsize=(12, max(4.6, 0.36 * len(values))), dpi=170)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ecfeff")
    bars = ax.barh(labels, values, color="#0f766e", edgecolor="#134e4a", linewidth=0.9)
    ax.axvline(1.0, color="#991b1b", linestyle="--", linewidth=1)
    for bar, value in zip(bars, values):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.2f}x",
            va="center",
            ha="left",
            fontsize=8,
        )
    ax.set_title("Tok'n'Roll Speedup Overview", fontsize=14, weight="bold")
    ax.set_xlabel("mean speedup ratio (Tok'n'Roll / Python backend)")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_dir / "summary-speedups-modern.png")
    plt.close(fig)


def write_merged_csv(
    output_path: Path,
    java_rows: dict[tuple[str, str, str, str, str], float],
    python_rows: dict[tuple[str, str, str, str, str], float],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["source", "backend", "model", "corpus", "size", "op", "throughput", "unit"]
        )
        for (implementation, model, corpus, size, operation), value in sorted(
            java_rows.items()
        ):
            writer.writerow(
                [
                    "java",
                    f"toknroll-{implementation}",
                    model,
                    corpus,
                    size,
                    operation,
                    f"{value:.6f}",
                    "ops/s" if operation == "decode" else "chars/s",
                ]
            )
        for (backend, model, corpus, size, operation), value in sorted(
            python_rows.items()
        ):
            writer.writerow(
                [
                    "python",
                    backend,
                    model,
                    corpus,
                    size,
                    operation,
                    f"{value:.6f}",
                    "ops/s" if operation == "decode" else "chars/s",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate cross-runtime benchmark charts"
    )
    parser.add_argument(
        "--java-json",
        default="toknroll-core/target/jmh-model-tokenizers.json",
        help="Path to ModelTokenizerBenchmark JMH JSON",
    )
    parser.add_argument(
        "--python-csv",
        default="target/python-tokenizers.csv",
        help="Path to Python benchmark CSV",
    )
    parser.add_argument(
        "--output-dir",
        default="target/benchmarks/charts",
        help="Directory where PNG charts are written",
    )
    parser.add_argument(
        "--merged-csv",
        default="target/benchmarks/merged-cross-runtime.csv",
        help="Path to merged CSV output",
    )
    args = parser.parse_args()

    java_rows = load_java(Path(args.java_json))
    python_rows = load_python(Path(args.python_csv))
    comparisons = discover_comparisons(java_rows, python_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_merged_csv(Path(args.merged_csv), java_rows, python_rows)

    ratio_rows: list[tuple[str, str, str, str, float]] = []
    chart_count = 0
    for model, backend, title in comparisons:
        for operation in ("encode", "decode"):
            points = aggregate_by_size(
                java_rows, python_rows, model, backend, operation
            )
            if not points:
                continue
            ratio_rows.extend(
                plot_pair_bars(output_dir, model, backend, title, operation, points)
            )
            chart_count += 1
            for implementation in IMPL_ORDER:
                corpora, sizes, matrix = aggregate_ratio_matrix(
                    java_rows,
                    python_rows,
                    model,
                    backend,
                    operation,
                    implementation,
                )
                if corpora and matrix:
                    plot_ratio_heatmap(
                        output_dir,
                        model,
                        backend,
                        operation,
                        implementation,
                        corpora,
                        sizes,
                        matrix,
                    )
                    chart_count += 1

    plot_summary_speedups(output_dir, ratio_rows)
    if ratio_rows:
        chart_count += 1

    print(f"Wrote merged CSV: {args.merged_csv}")
    print(f"Wrote charts under: {output_dir} ({chart_count} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
