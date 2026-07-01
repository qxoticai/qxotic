#!/usr/bin/env python3
"""Regenerate jam's README benchmark plots from the measured pp512 sweep.
Data: jinfer (native jam) vs llama.cpp, matched ISA per tier, Llama-3.2-1B, 16 threads,
Ryzen 9 9950X3D (Zen 5). Run:  python3 docs/bench_plot.py   (writes docs/bench-*.png)."""
import matplotlib.pyplot as plt
import numpy as np

quants = ["Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K"]
JAM, LLAMA = "#0a7d8c", "#c2c7cc"   # jam teal, llama grey

# pp512 t/s, Llama-3.2-1B, 16 threads, Ryzen 9 9950X3D (Zen5)
jam = {
 "sse3":        [169, 179, 114, 108, 83],
 "avx2":        [784, 644, 816, 793, 675],
 "avx_vnni":    [1187, 928, 881, 820, 691],
 "avx512_vnni": [1840, 1562, 1831, 1457, 1248],
}
llama = {
 "sse3":        [390, 293, 102, 93, 100],
 "avx2":        [1071, 1003, 1066, 611, 785],
 "avx_vnni":    [1372, 1077, 1068, 610, 785],
 "avx512_vnni": [2189, 1271, 1794, 661, 891],
}

# ---- Plot 1: flagship tier, jam vs llama absolute ----
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(quants)); w = 0.38
b1 = ax.bar(x - w/2, jam["avx512_vnni"], w, label="jam", color=JAM)
b2 = ax.bar(x + w/2, llama["avx512_vnni"], w, label="llama.cpp", color=LLAMA, edgecolor="#9aa0a6")
for b in (b1, b2):
    ax.bar_label(b, padding=2, fontsize=8, color="#444")
ax.set_ylabel("prefill  pp512  (tok/s)")
ax.set_title("jam vs llama.cpp — AVX-512-VNNI  ·  Llama-3.2-1B  ·  16 threads (Zen 5)", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(quants)
ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0, 2450); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig("docs/bench-avx512.png", dpi=130)
print("wrote docs/bench-avx512.png")

# ---- Plot 2: ratio across ISA tiers ----
tiers = ["sse3", "avx2", "avx_vnni", "avx512_vnni"]
shades = ["#bfe3e8", "#7fc6d1", "#3fa9b9", "#0a7d8c"]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
x = np.arange(len(quants)); w = 0.2
for i, t in enumerate(tiers):
    r = [j/l for j, l in zip(jam[t], llama[t])]
    bars = ax.bar(x + (i-1.5)*w, r, w, label=t, color=shades[i])
ax.axhline(1.0, color="#d62728", lw=1.1, ls="--", label="parity (1.0×)")
ax.set_ylabel("jam ÷ llama.cpp   (pp512, higher = jam faster)")
ax.set_title("jam ÷ llama.cpp prefill, by weight type and ISA tier  ·  Llama-3.2-1B (Zen 5)", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels(quants)
ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0, 2.45); ax.grid(axis="y", alpha=0.25)
fig.tight_layout(); fig.savefig("docs/bench-ratio.png", dpi=130)
print("wrote docs/bench-ratio.png")
