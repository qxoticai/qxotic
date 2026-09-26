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

## 2026-09-26: the 32x4 VNNI band vs llama.cpp's tiled mul_mat

llama.cpp PR #27851 (merged 2026-09-26) replaced its K-quant `vec_dot` prefill with a tiled VNNI GEMM (2.9-3.7x on the K-quants above), and jam's AVX-512-VNNI bands were rewritten the same day ([design.md](design.md), "Prefill bands").
Same box, same pure quants, `avx512_vnni` tier, 16 threads, llama.cpp master `86a24a182` built with `GGML_NATIVE=ON`;
`--repack 0` disables llama.cpp's x86 repack layouts so the K-quants take the new tiled path (Q4_0 loses its repack kernel with it):

| pp512 t/s | Q4_0 | Q8_0 | Q4_K | Q5_K | Q6_K |
|---|---|---|---|---|---|
| jinfer (native jam) | 1497 | 1457 | 1525 | 1467 | 1510 |
| llama.cpp, default | 991 | 622 | 891 | 1209 | 1157 |
| llama.cpp, `--repack 0` (tiled) | 531 | 621 | 1241 | 1211 | 1150 |
| llama.cpp, `GGML_CPU_TILED_MM=0` (the pre-#27851 `vec_dot` path, best of both repack modes) | 984 | 621 | 882 | 317 | 428 |

At the matmul level (`jam_bench 4096 512 4096`, 8 threads, GMAC/s) the bands went Q4_K 2027 -> 3098, Q5_K 1433 -> 3155, Q6_K 1268 -> 3297, Q8_0 1764 -> 3155, Q4_0 2059 -> 3052 (session-to-session variance on this CPU is about 20%, so compare rows measured together).
Logs: `bench-results/2026-09-26-vnni-band-vs-llama-tiled` (gitignored, local).

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
