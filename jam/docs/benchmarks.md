# jam benchmarks

Prefill throughput (`pp512`) of jinfer on the native jam backend and of llama.cpp, at matched ISA
per tier. Gemma 4 E2B, 16 threads, Ryzen 9 9950X3D (Zen 5). llama.cpp is the reference: its CPU
kernels are the baseline jam is measured against, and its block formats are what jam consumes
unchanged.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-tiers-dark.png">
  <img alt="jinfer (native jam) vs llama.cpp, prefill by ISA tier" src="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-tiers.png">
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
  <img alt="jinfer (native jam) prefill across ISA tiers" src="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-isa.png">
</picture>

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
