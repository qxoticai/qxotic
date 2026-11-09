"""Draws encode.png from run.sh's results: the median of each tokenizer's rounds per model.

    python chart.py results.jsonl encode.png

A tokenizer whose ids differ from toknroll's on a model is an error, never a bar: the chart compares
speed at equal output only.
"""

import json
import statistics
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

MODELS = {  # repository: the name on the chart, in the chart's order
    "Qwen/Qwen3-0.6B": "Qwen3",
    "unsloth/Llama-3.2-1B-Instruct": "Llama 3.2",
    "openai/gpt-oss-20b": "gpt-oss",
    "deepseek-ai/DeepSeek-V3.2": "DeepSeek V3.2",
    "google/gemma-4-e2b-it": "Gemma 4",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B",
}
# series in a fixed order, each a fixed color: a categorical palette validated for a light surface
SERIES = [
    ("toknroll", "toknroll", "#2a78d6"),
    ("tokenizers 1.", "HF tokenizers", "#eb6834"),  # the version measured joins the name
    ("tokenizers 0.", "HF tokenizers", "#1baf7a"),
    ("tiktoken", "tiktoken", "#eda100"),
]
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
BAR, GAP, GROUP_GAP = 0.2, 0.03, 0.3  # in bar-row units: a bar, the gap between neighbours, between models
RADIUS_PX = 4 * 2  # 4 px at about the 2x resolution saved


def version(library: str) -> str:
    """The version as people say it: "1.0.0-rc.4" is 1.0 rc4, a release keeps its three numbers."""
    v = library.split()[-1]
    if "-rc." in v:
        base, rc = v.split("-rc.")
        return f"{base.removesuffix('.0')} rc{rc}"
    return v


def legend_name(s: int, versions: dict) -> str:
    """A series' legend name; the two HF tokenizers series say which version each is."""
    return f"{SERIES[s][1]} {version(versions[s])}" if SERIES[s][1] == "HF tokenizers" else SERIES[s][1]


def series_of(library: str) -> int:
    return next(i for i, (prefix, _, _) in enumerate(SERIES) if library.startswith(prefix))


def load(results: str):
    runs, ids, versions = defaultdict(list), {}, {}
    for line in open(results):
        r = json.loads(line)
        key = (r["model"], series_of(r["library"]))
        runs[key].append(r["mb_per_s"])
        ids.setdefault(key, r["ids_sha256"])
        versions[key[1]] = r["library"]
    for (model, s), digest in ids.items():
        if digest != ids[(model, 0)]:
            raise SystemExit(f"{versions[s]} gives other ids than toknroll on {model}: no bar for it")
    return runs, versions


def bar(value: float, y: float, px_per_x: float, px_per_y: float) -> Path:
    """A bar from the baseline to value: square at the baseline, its two data-end corners rounded
    by RADIUS_PX, the same in pixels on both axes."""
    rx = min(RADIUS_PX / px_per_x, value / 2)
    ry = min(RADIUS_PX / px_per_y, BAR / 2)
    k = 0.5523  # a quarter circle as one cubic Bezier
    top, bottom, end = y - BAR / 2, y + BAR / 2, value
    vertices = [
        (0, top), (end - rx, top),
        (end - rx + k * rx, top), (end, top + ry - k * ry), (end, top + ry),
        (end, bottom - ry),
        (end, bottom - ry + k * ry), (end - rx + k * rx, bottom), (end - rx, bottom),
        (0, bottom), (0, top),
    ]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY]
    return Path(vertices, codes)


def main(results: str, out: str) -> None:
    runs, versions = load(results)
    bars, ticks, y = [], [], 0.0  # bars: (y, MB/s, series)
    for model, name in MODELS.items():
        first = y
        for s in range(len(SERIES)):
            if (model, s) in runs:
                bars.append((y, statistics.median(runs[(model, s)]), s))
                y += BAR + GAP
        ticks.append(((first + y - BAR - GAP) / 2, name))
        y += GROUP_GAP
    top = max(v for _, v, _ in bars)
    present = sorted({s for _, _, s in bars})

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7, 5.2), dpi=220)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, top * 1.1)
    ax.set_ylim(y - GROUP_GAP - GAP + BAR / 2 + 0.06, -BAR / 2 - 0.06)
    ax.set_yticks([t for t, _ in ticks], [name for _, name in ticks], fontsize=12, color=INK)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors=INK_2, labelsize=10, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("MB/s of text never seen before, one CPU core (higher is better)",
                  color=INK_2, fontsize=10.5, labelpad=8)
    fig.legend([Rectangle((0, 0), 1, 1, color=SERIES[s][2]) for s in present], [legend_name(s, versions) for s in present],
               loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=len(present), frameon=False,
              fontsize=10.5, labelcolor=INK, handlelength=1.0, handleheight=0.9, columnspacing=1.2, handletextpad=0.5)
    fig.text(0.012, 0.012,
             "Wikipedia articles from enwik8, 8 MB, one encode call each; median of 3 runs; every tokenizer returns the same ids.\n"
             + ", ".join(versions[s] for s in present) + ".",
             fontsize=8, color=INK_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))

    # the layout is final: a radius in pixels becomes data units on each axis
    (x0, y0), (x1, y1) = ax.transData.transform([(0, 0), (1, 1)])
    px_per_x, px_per_y = abs(x1 - x0), abs(y1 - y0)
    for yy, value, s in bars:
        color = SERIES[s][2]
        ax.add_patch(PathPatch(bar(value, yy, px_per_x, px_per_y), color=color, lw=0))
        label = f"{value:.0f}" if value >= 10 else f"{value:.1f}"
        ax.text(value + top * 0.012, yy, label, va="center", ha="left", fontsize=10, color=INK_2)
    fig.savefig(out, facecolor=SURFACE)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
