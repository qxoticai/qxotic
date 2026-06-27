# jam vs tinyBLAS

Side-by-side benchmark of jam's quantized matmul against llama.cpp's tinyBLAS
(`llamafile_sgemm`), on the **Q8_0** and **Q4_0** weight formats.

## Prerequisites

- jam built: `qxotic/jam/build/libjam.so` (`cmake --build jam/build`, or `mvn -pl jam package`)
- llama.cpp built with the CPU backend: `llama.cpp/build/bin/lib{ggml-cpu,ggml-base}.so`
  (`cmake -B build -DGGML_NATIVE=ON && cmake --build build --target ggml-cpu -j`)

If your llama.cpp lives elsewhere, set `LLAMA=/path/to/llama.cpp` before `./build.sh`.

## Build & run

```sh
./build.sh
JAM_NUM_THREADS=<physical cores> ./jam_vs_tinyblas 4096 512 4096
```

Args: `[M N K] [iters]`, or `[size] [iters]` for a square `m=n=k`.

## What it measures

The same quantized **weight bytes** feed both engines — `jam_ref`'s quantizers emit the exact
GGML block layout (`block_q8_0` = 34 B, `block_q4_0` = 18 B). The F32→Q8_0 **activation requant**
is timed on both sides (jam does it inside `jam_mm`; tinyBLAS via `quantize_row_q8_0`). Both use the
same thread count, with a `>LLC` cache scrub between iters so each call streams cold from DRAM.

Per quant it prints jam and tinyBLAS **GMAC/s** + **GB/s**, the **jam/tinyBLAS** speed ratio, and
**max|Δ|** between the two outputs — a free correctness / byte-compatibility cross-check (expect ~0.01–0.2,
i.e. requant rounding only).

## Sample (16-core AVX-512-VNNI)

```
jam vs tinyBLAS   m=2048 n=2048 k=2048   threads=16   jam isa=avx512_vnni
  quant |  jam GMAC/s     GB/s |   tb GMAC/s     GB/s |   jam/tb   max|Δ|
  Q8_0  |      1777.2      7.9 |       741.6      3.3 |    2.40x    0.0312
  Q4_0  |      2369.8      9.9 |       648.8      2.7 |    3.65x      0.16
```

## Notes

- tinyBLAS requires `n >= 2`; for `n == 1` (decode/gemv) it falls back to generic ggml in real
  llama.cpp, so this harness targets the **prefill** regime.
- **GMAC/s** is the exact apples-to-apples metric. **GB/s** is nominal — jam streams a *repacked*
  weight that may differ in size from the original block layout, so read it as indicative.
- **tinyBLAS threading.** `llamafile_sgemm`'s own multithreading goes through ggml's threadpool
  (chunk-stealing), which isn't available standalone — so the harness drives a *single-threaded*
  `llamafile_sgemm` per worker over a disjoint slice of output rows. With `JAM_NUM_THREADS=1` this *is*
  tinyBLAS's native single-call path, and there jam is already ~3× faster (the repacked-weight kernel);
  the gap actually **narrows** to ~2.4× at 16 threads as both approach memory bandwidth. So the
  row-partition isn't handicapping tinyBLAS — if anything it helps it scale.
- **jam's weight repack is real.** jam repacks each weight once and caches it; the warm-up pays the
  repack, timed calls reuse it — the realistic per-token inference case. tinyBLAS reads the raw blocks
  every call. That's a genuine jam design advantage, not a measurement artifact.
- The `k` argument to `llamafile_sgemm` is in **blocks** (`k/32`), not elements — and the CPU backend
  must be initialized (`ggml_cpu_init()`) before the first call. Both are handled; noted here because
  neither is obvious and either one silently corrupts results.
- Set `JAM_NUM_THREADS` to your **physical** core count for the best numbers (it sizes both pools).
- This is a cross-project harness (it links the sibling llama.cpp), so it is **not** wired into
  jam's CMake build — just `./build.sh`.
