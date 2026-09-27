# jam benchmarks

Prefill throughput (`pp512`) of jinfer on the native jam backend and of llama.cpp, at matched
instruction set per tier. Gemma 4 E2B, 16 threads, Ryzen 9 9950X3D (Zen 5). llama.cpp is the reference: its CPU
kernels are the baseline jam is measured against, and its block formats are what jam consumes
unchanged.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-tiers-dark.png">
  <img alt="jinfer (native jam) vs llama.cpp, prefill by instruction set" src="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-tiers.png">
</picture>

| pp512 t/s, jinfer (native jam) / llama.cpp | Q4_0 | Q8_0 | Q4_K | Q5_K | Q6_K |
|---|---|---|---|---|---|
| sse3 | 178 / 176 | 175 / 136 | 119 / 49 | 109 / 45 | 102 / 48 |
| avx2 | 649 / 514 | 647 / 477 | 653 / 527 | 647 / 291 | 533 / 371 |
| avx_vnni | 954 / 647 | 791 / 509 | 638 / 520 | 660 / 289 | 533 / 367 |
| avx512_vnni | 1358 / 947 | 1241 / 605 | 1368 / 835 | 1097 / 313 | 987 / 421 |

The flagship tier on its own:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-avx512-dark.png">
  <img alt="jinfer (native jam) vs llama.cpp on AVX-512-VNNI" src="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-avx512.png">
</picture>

The same int8 kernels span the whole x86 ladder, from the pre-AVX2 floor up to AVX-512:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-isa-dark.png">
  <img alt="jinfer (native jam) prefill by instruction set" src="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-isa.png">
</picture>

## 2026-09-27: the 32x4 VNNI band vs llama.cpp's tiled mul_mat

llama.cpp PR #27851 (merged 2026-09-26) replaced its K-quant `vec_dot` prefill with a tiled VNNI GEMM, and jam's AVX-512-VNNI bands were rewritten in response ([design.md](design.md), "Prefill bands").
Everything below is one session: the same box, the same pure quants, the `avx512_vnni` tier, qxotic `ea22debad`, llama.cpp master `86a24a182` built with `GGML_NATIVE=ON`.
llama.cpp runs in three modes: its defaults, `--repack 0` (no x86 repack layouts, so the K-quants take the new tiled path and Q4_0 loses its repack kernel), and `GGML_CPU_TILED_MM=0` (the tiled path off, the kernels before #27851).

At 16 threads:

| pp512 t/s | Q4_0 | Q8_0 | Q4_K | Q5_K | Q6_K |
|---|---|---|---|---|---|
| jinfer (native jam) | 1563 | 1487 | 1502 | 1499 | 1507 |
| llama.cpp, default | 982 | 617 | 879 | 1228 | 1157 |
| llama.cpp, `--repack 0` (tiled) | 534 | 613 | 1241 | 1228 | 1143 |
| llama.cpp, `GGML_CPU_TILED_MM=0` (before #27851) | 986 | 618 | 883 | 318 | 428 |

By thread count, jinfer / llama.cpp's best mode for the format (the repack kernel for Q4_0, the tiled path for the K-quants):

| threads | Q4_0 | Q8_0 | Q4_K | Q5_K | Q6_K |
|---|---|---|---|---|---|
| 1 | 176 / 91 | 175 / 49 | 179 / 133 | 178 / 132 | 185 / 127 |
| 2 | 323 / 176 | 321 / 96 | 325 / 259 | 330 / 258 | 339 / 248 |
| 4 | 606 / 339 | 614 / 186 | 607 / 480 | 586 / 477 | 643 / 456 |
| 8 | 1078 / 639 | 1063 / 357 | 1047 / 881 | 1049 / 875 | 1073 / 836 |
| 16 | 1563 / 986 | 1487 / 618 | 1502 / 1241 | 1499 / 1228 | 1507 / 1157 |

jam leads at every thread count and format: by 1.2x to 1.45x on the K-quants, 1.6x to 1.9x on Q4_0 and 2.4x to 3.6x on Q8_0.
The lead is widest per core and narrows with threads: from 1 to 16 threads jinfer scales 8.4x on Q4_K where llama.cpp's tiled path scales 9.3x.
llama.cpp's numbers repeat within 1%, jinfer's within 2% to 5%.

By prompt length, a separate session, jinfer / llama.cpp's best mode (`-r 10` for pp16 and pp64).

16 threads:

| t/s | pp16 | pp64 | pp512 |
|---|---|---|---|
| Q4_0 | 543 / 655 | 1098 / 862 | 1575 / 978 |
| Q8_0 | 342 / 399 | 877 / 554 | 1441 / 620 |
| Q4_K | 511 / 656 | 1034 / 842 | 1480 / 1205 |
| Q5_K | 502 / 476 | 1032 / 836 | 1416 / 1226 |
| Q6_K | 443 / 394 | 990 / 745 | 1468 / 1153 |

1 thread:

| t/s | pp16 | pp64 | pp512 |
|---|---|---|---|
| Q4_0 | 102 / 91 | 164 / 93 | 176 / 91 |
| Q8_0 | 91 / 43 | 157 / 49 | 177 / 49 |
| Q4_K | 101 / 85 | 166 / 127 | 178 / 132 |
| Q5_K | 96 / 77 | 160 / 124 | 178 / 132 |
| Q6_K | 102 / 55 | 170 / 108 | 186 / 127 |

A 16-token prompt on 16 threads is where jam loses: llama.cpp is ahead by 17% to 28% on Q4_0, Q8_0 and Q4_K.
jam repacks every weight band on every call, a fixed cost worth about 10 columns of dots, and at 16 columns it is no longer amortized; llama.cpp's repack kernels for Q4_0 and Q4_K pay theirs once, at load.
On one thread the same prompt is ahead on every format, so the rest of the gap is the per-call fan-out across 16 workers.
From 64 tokens up jam leads everywhere.

What #27851 changed inside llama.cpp, the tiled path against the kernels before it:

| threads | Q4_K | Q5_K | Q6_K |
|---|---|---|---|
| 1 | 84 -> 133 | 24 -> 132 | 33 -> 126 |
| 8 | 581 -> 881 | 173 -> 874 | 239 -> 836 |
| 16 | 883 -> 1241 | 318 -> 1228 | 428 -> 1143 |

Its default mode keeps the repack kernel for Q4_K, so a Q4_K run only sees the tiled path with `--repack 0`.

At the matmul level (`jam_bench`, m = 4096, k = 4096, GMAC/s), the new band against jam's previous 16-row bands:

| | Q4_0 | Q8_0 | Q4_K | Q5_K | Q6_K |
|---|---|---|---|---|---|
| n = 512, 8 threads, before | 2059 | 1764 | 2027 | 1433 | 1268 |
| n = 512, 8 threads, after | 3052 | 3155 | 3098 | 3155 | 3297 |

These two rows are from 2026-09-26 and were measured a few hours apart, so read them as the size of the change, not to the percent.
This CPU drifts by about 20% between sessions (thread placement, the V-cache CCD, sustained clocks); the pp512 tables above are one session each.
Logs: `bench-results/2026-09-27-vnni-band-polished`, the prompt-length runs under `by-prompt` (gitignored, local).

## Method

- Weights are pure quants of Gemma 4 E2B, each made from the BF16 checkpoint with
  `llama-quantize --pure <bf16.gguf> <out.gguf> <TYPE>`; Q8_0 is the published file.
- jinfer runs `JinferBench -p 512 -n 0 -r 5 -w 2 -t 16` (see
  [jinfer-bench](../../jinfer/jinfer-bench/README.md)) with jam capped per tier through `JAM_ISA`;
  `JAM_DEBUG=1` records the bound tier in the log.
- llama.cpp runs `llama-bench -p 512 -n 0 -r 5 -t 16` from one build per tier, configured with
  `GGML_NATIVE=OFF` and only that tier's `GGML_*` flags.
- `bench_sweep.sh` runs the whole grid; `bench_plot.py` holds the numbers and draws the charts. The PNGs
  are not checked in here: they live under `jam/` in [qxoticai/assets](https://github.com/qxoticai/assets),
  which the `<picture>` sources here and in the README point at. After regenerating, copy them there.

These numbers cover one machine and one model. Run `jam_bench` and a local `pp512` to measure other
hardware. The [tinyBLAS harness](../jam-native/bench/README.md) compares the raw matmul, outside any
inference engine.
