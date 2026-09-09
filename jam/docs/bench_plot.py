#!/usr/bin/env python3
"""Regenerate jam's README benchmark plots from the measured pp512 sweep (docs/bench_sweep.sh).
Data: jinfer (native jam) and llama-bench, matched instruction set per tier, Gemma 4 E2B (pure quants from
BF16), 16 threads, Ryzen 9 9950X3D (Zen 5). Run:  python3 docs/bench_plot.py   (writes docs/bench-*.png,
a light and a dark variant of each, for a <picture> element)."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

matplotlib.rcParams["font.family"] = ["Noto Sans", "DejaVu Sans", "sans-serif"]

quants = ["Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"]
tiers = ["sse3", "avx2", "avx_vnni", "avx512_vnni"]

# pp512 t/s, Gemma 4 E2B, 16 threads, Ryzen 9 9950X3D (Zen 5), 2026-09-09
jam = {
 "sse3":        [178, 175, 119, 109, 102],
 "avx2":        [649, 647, 653, 647, 533],
 "avx_vnni":    [954, 791, 638, 660, 533],
 "avx512_vnni": [1358, 1241, 1368, 1097, 987],
}
llama = {
 "sse3":        [176, 136, 49, 45, 48],
 "avx2":        [514, 477, 527, 291, 371],
 "avx_vnni":    [647, 509, 520, 289, 367],
 "avx512_vnni": [947, 605, 835, 313, 421],
}

# Two themes, matching GitHub's README surfaces. Series colors were checked with the dataviz palette
# validator (CVD separation, chroma, contrast) against each surface; the tier ramp is ordinal teal.
THEMES = {
    "light": dict(surface="#ffffff", ink="#1f2328", ink2="#59636e", muted="#818b98", grid="#e6e8eb", base="#c9ced4",
                  jam="#0a8fa0", llama="#eb6834", ramp=["#6cc3cd", "#3aa7b5", "#0a8fa0", "#075f6a"]),
    "dark":  dict(surface="#0d1117", ink="#e6edf3", ink2="#b1bac4", muted="#8b949e", grid="#21262d", base="#3d444d",
                  jam="#0a8fa0", llama="#d95926", ramp=["#1a5f68", "#0a8fa0", "#3fb6c4", "#8fd8e0"]),
}
LABEL_JAM, LABEL_LLAMA = "jinfer (native jam)", "llama.cpp"
SUBTITLE = "Gemma 4 E2B  ·  16 threads  ·  Ryzen 9 9950X3D (Zen 5)"


def style(ax, th):
    ax.set_facecolor(th["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(th["base"])
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", colors=th["muted"], labelsize=8.5, length=0, pad=6)
    ax.yaxis.grid(True, color=th["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))


def rounded_bars(ax, x, heights, width, color, radius_px, gap_px, fig):
    """Columns with a rounded cap and a square base (rounded box clipped at the baseline)."""
    fig.canvas.draw()
    px_per_x = ax.get_window_extent().width / np.diff(ax.get_xlim())[0]
    px_per_y = ax.get_window_extent().height / np.diff(ax.get_ylim())[0]
    gap = gap_px / px_per_x / 2
    r = radius_px / px_per_x
    aspect = px_per_x / px_per_y
    for xi, h in zip(x, heights):
        w = width - 2 * gap
        box = FancyBboxPatch((xi - w / 2, -r * aspect), w, h + r * aspect,
                             boxstyle=f"round,pad=0,rounding_size={r}", mutation_aspect=aspect,
                             facecolor=color, edgecolor="none", linewidth=0)
        box.set_clip_path(Rectangle((xi - w / 2, 0), w, h, transform=ax.transData))
        ax.add_patch(box)


def value_labels(ax, x, heights, th, fontsize=8):
    for xi, h in zip(x, heights):
        ax.annotate(f"{h:,}", (xi, h), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color=th["ink2"])


def legend(fig, th, y):
    handles = [Rectangle((0, 0), 1, 1, color=th["jam"]), Rectangle((0, 0), 1, 1, color=th["llama"])]
    fig.legend(handles, [LABEL_JAM, LABEL_LLAMA], frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, y), fontsize=9, labelcolor=th["ink2"], handlelength=1.2, handleheight=0.9,
               columnspacing=1.6)


def titles(fig, th, title, y=0.985):
    fig.text(0.5, y, title, ha="center", va="top", fontsize=13, color=th["ink"], fontweight="semibold")
    fig.text(0.5, y - 0.048, SUBTITLE, ha="center", va="top", fontsize=9, color=th["muted"])


def save(fig, name, th):
    fig.savefig(f"docs/{name}", dpi=200, facecolor=th["surface"])
    plt.close(fig)
    print(f"wrote docs/{name}")


for mode, th in THEMES.items():
    suffix = "" if mode == "light" else "-dark"

    # ---- Plot 1: flagship tier, jinfer (native jam) and llama.cpp side by side ----
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=th["surface"])
    style(ax, th)
    x = np.arange(len(quants)); w = 0.3
    ax.set_xlim(-0.6, len(quants) - 0.4); ax.set_ylim(0, 1600)
    ax.set_xticks(x); ax.set_xticklabels(quants, color=th["ink2"], fontsize=9.5)
    fig.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.98)
    rounded_bars(ax, x - w / 2, jam["avx512_vnni"], w, th["jam"], 4, 2, fig)
    rounded_bars(ax, x + w / 2, llama["avx512_vnni"], w, th["llama"], 4, 2, fig)
    value_labels(ax, x - w / 2, jam["avx512_vnni"], th)
    value_labels(ax, x + w / 2, llama["avx512_vnni"], th)
    ax.set_ylabel("prefill, pp512 (tok/s)", color=th["muted"], fontsize=9)
    titles(fig, th, "jinfer (native jam) vs llama.cpp on AVX-512-VNNI")
    legend(fig, th, 0.885)
    save(fig, f"bench-avx512{suffix}.png", th)

    # ---- Plot 2: jinfer (native jam) across instruction sets ----
    fig, ax = plt.subplots(figsize=(8, 4.6), facecolor=th["surface"])
    style(ax, th)
    x = np.arange(len(quants)); w = 0.19
    ax.set_xlim(-0.6, len(quants) - 0.4); ax.set_ylim(0, 1600)
    ax.set_xticks(x); ax.set_xticklabels(quants, color=th["ink2"], fontsize=9.5)
    fig.subplots_adjust(top=0.78, bottom=0.1, left=0.1, right=0.98)
    for i, t in enumerate(tiers):
        rounded_bars(ax, x + (i - 1.5) * w, jam[t], w, th["ramp"][i], 3, 2, fig)
    ax.set_ylabel("prefill, pp512 (tok/s)", color=th["muted"], fontsize=9)
    titles(fig, th, "jinfer (native jam) prefill by instruction set")
    handles = [Rectangle((0, 0), 1, 1, color=c) for c in th["ramp"]]
    fig.legend(handles, tiers, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.885),
               fontsize=9, labelcolor=th["ink2"], handlelength=1.2, handleheight=0.9, columnspacing=1.6)
    save(fig, f"bench-isa{suffix}.png", th)

    # ---- Plot 3: the whole sweep, one panel per instruction set ----
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), facecolor=th["surface"])
    x = np.arange(len(quants)); w = 0.3
    for ax, t in zip(axes.flat, tiers):
        style(ax, th)
        top = max(jam[t] + llama[t]) * 1.2
        ax.set_xlim(-0.6, len(quants) - 0.4); ax.set_ylim(0, top)
        ax.set_xticks(x); ax.set_xticklabels(quants, color=th["ink2"], fontsize=9)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))
    fig.subplots_adjust(top=0.82, bottom=0.06, left=0.075, right=0.985, hspace=0.4, wspace=0.18)
    for ax, t in zip(axes.flat, tiers):
        rounded_bars(ax, x - w / 2, jam[t], w, th["jam"], 4, 2, fig)
        rounded_bars(ax, x + w / 2, llama[t], w, th["llama"], 4, 2, fig)
        value_labels(ax, x - w / 2, jam[t], th, fontsize=7.5)
        value_labels(ax, x + w / 2, llama[t], th, fontsize=7.5)
        ax.set_title(t, loc="left", fontsize=10, color=th["ink"], fontweight="semibold", pad=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("pp512 (tok/s)", color=th["muted"], fontsize=9)
    titles(fig, th, "jinfer (native jam) vs llama.cpp, prefill by instruction set", y=0.985)
    legend(fig, th, 0.92)
    save(fig, f"bench-tiers{suffix}.png", th)
