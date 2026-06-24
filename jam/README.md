# jam

**Java Accelerated Math** — the fastest multithreaded matmul for LLM inference, in one function:
**`C = A @ Bᵀ`**, with quantized weights.

`jam` is a tiny, dependency-free C library (a focused subset of BLAS) whose only job is a fast,
quant-aware matrix multiply across every CPU SIMD level — and the Apple GPU. It exists to make LLM
**prefill** fast: the weight `A` may be quantized (and selects the kernel), `B` and `C` are float.

- **One op, one job.** `jam_mm` computes `C = A @ Bᵀ`. Matrix-vector (decode/gemv) is just `n == 1`.
- **GGUF-native.** dtype tags are numerically identical to `ggml_type`; the layout is the `ggml
  mul_mat` convention, so weights are byte-compatible with llama.cpp — no conversion.
- **Multiplies, never converts.** Block decode and activation requant happen *inside* the kernel;
  quantizing the weights is the host's job.
- **Picks the best kernel once.** At context creation it detects the CPU and binds a kernel pointer;
  the per-call path is a table lookup plus a few branches (the tinyBLAS discipline).
- **Always multithreaded.** Every kernel is a row-range worker the thread pool fans out automatically.

## Quick start

```c
#include <jam.h>

// A = Q8_0 weight [out × in], B = F32 activations [tokens × in], C = F32 output [out × tokens].
jam_status st = jam_mm(NULL,                  // NULL -> global, env-configured context
                       W, JAM_Q8_0, in,       // A: ptr, dtype, row stride (lda)
                       X, JAM_F32,  in,        // B: ptr, dtype, row stride (ldb)
                       Y, JAM_F32,  tokens,    // C: ptr, dtype, row stride (ldc)
                       out, tokens, in);       // m, n, k
```

```c
jam_status jam_mm(jam_ctx* ctx,
                  const void* a, jam_dtype at, int lda,    // weight  [m × k]
                  const void* b, jam_dtype bt, int ldb,    // activation [n × k]
                  void*       c, jam_dtype ct, int ldc,    // output  [m × n]
                  int m, int n, int k);
```

`W [m×k]` (weights) and `A [n×k]` (activations) are both **row-major with the reduction dim `k`
contiguous**, so every dot is two contiguous reads. The output is **token-major** (ggml/llama.cpp
layout): `C[j*ldc + i] = dot(W[i,:], A[j,:])`, i.e. each token `j`'s `m`-feature vector is contiguous
(`ldc >= m`) — so it drops straight into an inference engine with no transpose.
Returns `JAM_OK`, `JAM_EINVAL` (bad args), `JAM_EUNSUPPORTED` (dtype combo not built), or `JAM_EBUSY`
(another `mm` is in flight on this context — it's a serial stream; retry or fall back).

### Supported dtype combinations

| A (weight) | B | C | path |
|---|---|---|---|
| `JAM_F32`  | `JAM_F32` | `JAM_F32` | float GEMM |
| `JAM_Q8_0` | `JAM_F32` | `JAM_F32` | int8 weight · float activation (requantized internally) |

Other combinations return `JAM_EUNSUPPORTED` (Q4_K / Q6_K / MXFP4 are planned). `k` (and `lda` for
Q8_0) must be a multiple of 32.

## Configuration

### Environment variables (the global context)

Read **once**, lazily, on the first `jam_mm(NULL, …)` call. An *explicit* context ignores the env.

| env var | meaning | default |
|---|---|---|
| `JAM_NUM_THREADS` | thread-pool size | `0` = number of cores |
| `JAM_ISA` | cap / select the kernel (capability name below, or `metal`) | best detected |
| `JAM_DEBUG` | print CPU features + the kernels bound at context creation (to stderr) | off |

```sh
JAM_NUM_THREADS=16 JAM_ISA=avx2  ./app    # 16 threads, capped at AVX2 (no AVX-512 / AVX-VNNI)
JAM_ISA=metal                    ./app    # run on the Apple GPU
JAM_DEBUG=1                      ./app    # diagnose which kernel got selected and why
```

With `JAM_DEBUG=1`, each context creation logs which kernels were bound — handy for confirming the fast
path is active or diagnosing a fallback:

```
[jam] cpu=x86_64 features: sse2 avx2 fma f16c avx512f avx512bw avx512dq avx512vl avx512vnni avxvnni
[jam] cap=auto  active=avx512_vnni  threads=32  metal=no
[jam]   F32  kernel: avx512 (mnpack, 16-wide)
[jam]   Q8_0 kernel: avx512_vnni (512-bit vpdpbusd)
```

### Explicit context (for fine control)

```c
jam_config cfg = {0};
cfg.nthreads = 8;                 // 0 = cores
cfg.max_isa  = JAM_ISA_AVX2;      // cap; JAM_ISA_AUTO = best
cfg.name     = "decode-pool";     // optional label, shown in JAM_DEBUG logs (jam_ctx_name(ctx))
jam_ctx* ctx = jam_ctx_create(&cfg);
jam_mm(ctx, /* … */);
jam_ctx_destroy(ctx);
```

You can also plug in your **own** thread pool by setting `cfg.parallel_for` / `cfg.pool` (jam then never
creates its own pool — useful for sharing one pool with the rest of your engine).

## Backends & capabilities

`jam` detects the CPU once and binds the best kernel for each path. `jam_active_isa(ctx)` /
`jam_isa_name(...)` report what's live; `JAM_ISA` caps it.

| arch | capability names (ascending) |
|---|---|
| x86 | `generic` · `avx2` · `avx_vnni` · `avx512` · `avx512_vnni` |
| arm | `generic` · `neon` · `dotprod` · `i8mm` |
| gpu | `metal` (Apple; opt-in, not auto-selected) |

`auto` (unset `JAM_ISA`) = best detected. Capping disables every level above — `JAM_ISA=avx2` avoids
AVX-512 *and* AVX-VNNI.

**Q8_0 int8 dot, per ISA:**

| ISA | instruction | notes |
|---|---|---|
| `avx2` | `maddubs` + `madd` | every x86 since Haswell (2013) |
| `avx_vnni` | 256-bit `vpdpbusd` | client CPUs *without* AVX-512 (Alder/Raptor Lake) |
| `avx512` | 512-bit `maddubs` | AVX-512 CPUs that lack VNNI (Skylake-SP/X) |
| `avx512_vnni` | 512-bit `vpdpbusd` | the x86 fast path (Zen4/5, Ice/Sapphire Rapids) |
| `neon` | `vmull` + `vpadal` | ARMv8 floor |
| `dotprod` | `sdot` | the ARM workhorse (all Apple M-series, Graviton2+) |
| `i8mm` | `smmla` (2×2 tile) | highest ARM throughput (Graviton3/4, Apple M-series) |
| `metal` | MSL compute | Apple GPU; dequantizes the weight, dots exact F32 `B` |

F32 GEMM uses register-tiled `mnpack` kernels (`avx2`/`avx512` on x86; ARM F32 currently uses the
portable kernel). `metal` is a GPU **backend**, not a CPU ISA: `auto` never picks it — request it
explicitly; it runs Q8_0/F32 on the GPU and falls back to CPU for other dtypes or non-contiguous
strides. `sve` / `amx` / SME are reserved names, not yet implemented.

## Threading & concurrency

`jam_mm` is multithreaded **within** a call. A `jam_ctx` is a **serial execution stream**: `mm` calls on
the **same** context run one at a time and must **not** be called concurrently (they share the pool and
the requant scratch).

- **One stream (an inference forward pass):** use the global context from a single thread. Each call is
  internally parallel. This is the intended, optimal case.
- **Parallel independent matmuls:** create **several contexts**, one per thread — each owns its pool +
  scratch. The cuBLAS-handle / oneDNN-stream model.

## Build

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build   # -> build/libjam.so(.1)
```

Per-ISA kernels live in their own translation units (`src/jam_kernels_<isa>.c`) compiled with their own
`-m` flags and bound at create. The dispatcher and the scalar floor (`jam_kernels_generic.c`) stay at
baseline flags, so the library **runs on any CPU** and can't emit a SIMD instruction before checking for
it. CMake builds the x86 TUs on x86, the ARM TUs on aarch64, and the Metal backend on Apple.

## Tests & benchmarks

```sh
cd build && ctest --output-on-failure        # comprehensive correctness
./jam_bench [size] [iters]                    # square (compute-bound)
./jam_bench M N K [iters]                     # explicit shape; e.g. 4096 1 4096 = gemv
```

`jam_test` exercises **every kernel the machine supports** — one context per capability (via `max_isa`),
at 1 and 3 threads — checking each against a double-precision reference (F32 vs exact dot; Q8_0 vs both
exact-B and requant-B, so it tolerates only the int8 quantization error).

`jam_bench` reports two units, because FLOP/s isn't meaningful across quantizations:

- **`GMAC/s`** = `m·n·k / t` — the arithmetic rate (int8 vs f32). The metric in the **compute-bound**
  regime (prefill / large `n`).
- **`GB/s`** = `(weights + activations + output) / t` — DRAM traffic, with Q8_0 weights at their real
  `34/32 = 1.0625` B/value. The metric in the **bandwidth-bound** regime (gemv / decode), where it
  should approach peak DRAM.

It scrubs the caches between timed calls so each matmul streams the weight from **DRAM**, not cache —
`JAM_BENCH_SCRUB_MB` (default 256) sizes the scrub buffer; raise it on CPUs with > 256 MB of L3.

```sh
./build-and-test.sh              # build + ctest + bench both regimes (square and a DRAM-bound gemv)
```

## Java (JNI)

`com.qxotic.jam.JAM` exposes a **single native method** — `mm` — on the env-configured global context
(call from one thread; internally multithreaded). Pointers are `long` off-heap addresses
(`MemorySegment.address()`); dtypes and dims are `int`:

```java
int st = JAM.mm(wAddr, JAM.Q8_0, in,      // W weights     [m × k]
                xAddr, JAM.F32,  in,      // A activations [n × k]
                yAddr, JAM.F32,  tokens,  // C output      [m × n]
                out, tokens, in);         // m, n, k
```

JUnit tests (`src/test/java`) cover loading, F32/gemv/Q8_0 correctness, and the `EUNSUPPORTED`/`EINVAL`
status paths. Build the native lib first — CMake stages it under `dist/native/` (a test resource), so
`NativeLoader` finds and loads it exactly as it would from the fat jar (no override, no per-OS config):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
mvn test
```

## Distribution (`jam.jar`)

The Java artifact is a **fat jar that carries the native library for every platform** and loads the
right one automatically — nothing to install. The combinatorics stay small because **ISA selection is at
runtime, not build time**: each `libjam` contains *all* the ISA kernels and picks one by CPUID, so you
ship **one library per `OS × CPU-arch`**, not per ISA:

```
jam.jar
└── com/qxotic/jam/native/
    ├── linux-x86-64/libjam.so       (generic · avx2 · avx_vnni · avx512 · avx512_vnni, runtime-picked)
    ├── linux-aarch64/libjam.so      (neon · dotprod · i8mm)
    ├── darwin-x86-64/libjam.dylib
    ├── darwin-aarch64/libjam.dylib  (+ Metal)
    └── windows-x86-64/jam.dll
```

`NativeLoader` detects `os.name`/`os.arch`, extracts the matching library to a temp file, and
`System.load`s it — then `JAM.mm` works. Override with `-Djam.library.path=/abs/libjam.so` (or
`JAM_LIBRARY_PATH`); a dev fallback tries `java.library.path`.

**Build it.** Each platform stages its lib (CMake `POST_BUILD` → `dist/native/…`); `scripts/build-jar.sh`
compiles the Java and bundles whatever's staged. Because one machine builds one platform, the *full* fat
jar is assembled by CI: `.github/workflows/build.yml` runs the `OS × arch` matrix (GitHub Linux/macOS/
Windows + native ARM runners), uploads each lib, and a final job merges them and runs `build-jar.sh`.

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build   # builds + stages this platform's lib
./scripts/build-jar.sh                                             # -> dist/jam.jar (fat for built platforms)
```

(Windows uses **clang**, not MSVC — the per-ISA `-m` flags are gcc/clang; MSVC `/arch` translation is a TODO.)

## Status

Working and tested: the engine, the env-configured global context, explicit contexts, the JNI binding +
fat-jar loader, **F32** GEMM, and **Q8_0** and **MXFP4** `@ F32 → F32` across the full x86 ladder
(`generic` → `avx2` → `avx_vnni` → `avx512` → `avx512_vnni`). ARM (`neon` / `dotprod` / `i8mm`) and the
Metal backend are implemented and cross-compile-/integration-checked but await validation on Apple
hardware. Kernels are organized as **per-quant decoders × per-ISA int8 dots** over a shared GEMM engine,
so a new quant is one decoder and a new ISA is one dot — not a kernel per combination.

Roadmap: more weight dtypes (Q4_K, Q6_K), AMX (x86) and SVE/SME (ARM), a NEON F32 kernel, a tiled Metal
kernel (`simdgroup_matrix` + zero-copy), and 512-bit MXFP4.
