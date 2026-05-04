#!/usr/bin/env python3
"""Generate static horizontal bar charts comparing multi-threaded tokenizer throughput.

Generates two separate charts: one for encode, one for decode.

Example:
  python toknroll-benchmarks/chart_benchmarks_mt.py
  python toknroll-benchmarks/chart_benchmarks_mt.py --output-encode mt_encode_chart.png --output-decode mt_decode_chart.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt


# Colors matching the request
COLORS = {
    "Java": "#00758F",  # Blue for Tok'n'Roll (from GraalVM badge)
    "Tokie": "#8B4513",  # Brown
}

# Raw data: time in seconds for 100 MB (enwik8)
# Using the multi-threaded benchmark data
DATA = {
    "r50k_base": {
        "encode": {"Java": 0.892, "Tokie": 1.625},
        "decode": {"Java": 0.115, "Tokie": 0.276},
    },
    "cl100k_base": {
        "encode": {"Java": 0.952, "Tokie": 1.467},
        "decode": {"Java": 0.144, "Tokie": 0.236},
    },
    "o200k_base": {
        "encode": {"Java": 0.562, "Tokie": 1.462},
        "decode": {"Java": 0.117, "Tokie": 0.228},
    },
}

# enwik8 is exactly 100 million bytes = 100 MB
CORPUS_MB = 100.0


def compute_throughput(time_s: float) -> float:
    """Convert time in seconds to throughput in MB/s."""
    return CORPUS_MB / time_s


def create_single_chart(operation: str, output_path: Path | None = None) -> None:
    """Create a single chart for encode or decode."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 3.4))
    ax.set_title(
        f"Multi-threaded {operation.capitalize()} on enwik8",
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=12,
    )

    encodings = ["r50k_base", "cl100k_base", "o200k_base"]
    implementations = ["Java", "Tokie"]

    # Nice display names
    encoding_labels = {
        "r50k_base": "r50k",
        "cl100k_base": "cl100k",
        "o200k_base": "o200k",
    }

    impl_labels = {
        "Java": "toknroll (Java)",
        "Tokie": "tokie",
    }

    # For each implementation, we'll have a group of 3 bars (one per encoding)
    n_impls = len(implementations)
    n_encodings = len(encodings)

    # Calculate positions
    group_height = 2.4  # Height of each implementation group
    bar_height = 0.75
    group_spacing = 0.4  # Space between groups

    # Calculate total height needed
    total_height = n_impls * group_height + (n_impls - 1) * group_spacing

    # Create y positions for each bar
    all_y_positions = []
    all_values = []
    all_colors = []
    all_labels = []

    for impl_idx, impl in enumerate(implementations):
        # Base position for this implementation group (top to bottom)
        group_base = (
            total_height
            - (impl_idx * (group_height + group_spacing))
            - group_height / 2
        )

        # For each encoding (r50k at bottom, o200k at top within group)
        for enc_idx, encoding in enumerate(encodings):
            time_s = DATA[encoding][operation][impl]
            throughput = compute_throughput(time_s)

            # Position within group (bottom to top: r50k, cl100k, o200k)
            y_pos = (
                group_base
                - group_height / 2
                + (enc_idx + 0.5) * (group_height / n_encodings)
            )

            all_y_positions.append(y_pos)
            all_values.append(throughput)
            all_colors.append(COLORS[impl])
            all_labels.append(f"{encoding_labels[encoding]}")

    # Create horizontal bars
    bars = ax.barh(
        all_y_positions,
        all_values,
        color=all_colors,
        edgecolor="white",
        linewidth=1.0,
        height=bar_height,
        alpha=0.9,
    )

    # Add value labels and encoding labels on bars
    for bar, value, label in zip(bars, all_values, all_labels):
        width = bar.get_width()
        # Value label at the end of the bar
        ax.text(
            width + max(all_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#333333",
        )
        # Encoding label inside the bar (left side)
        ax.text(
            max(all_values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white",
            alpha=0.9,
        )

    # Add implementation group labels on the left side
    for impl_idx, impl in enumerate(implementations):
        group_base = (
            total_height
            - (impl_idx * (group_height + group_spacing))
            - group_height / 2
        )
        label_y = group_base
        ax.text(
            -max(all_values) * 0.02,
            label_y,
            impl_labels[impl],
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )

    # Set labels
    ax.set_xlabel("Throughput (MB/s)", fontsize=11, fontweight="bold")

    # Remove y-axis ticks and labels (we have custom labels)
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Set y-axis limits to show all groups
    ax.set_ylim(-0.5, total_height + 0.5)

    # Add grid
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color="#CCCCCC")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Chart saved to: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-encode",
        type=Path,
        default=None,
        help="Output file path for encode chart (e.g., mt_encode.png).",
    )
    parser.add_argument(
        "--output-decode",
        type=Path,
        default=None,
        help="Output file path for decode chart (e.g., mt_decode.png).",
    )
    args = parser.parse_args()

    create_single_chart("encode", args.output_encode)
    create_single_chart("decode", args.output_decode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
