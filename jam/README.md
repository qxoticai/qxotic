# jam

**Just a matmul.** The fastest one on the JVM.

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

JVM Accelerated Math. Fast quantized matrix multiplication for CPUs, from Java or C. One operation,
every instruction set: SSE3 through AVX-512-VNNI on x86, NEON, DotProd and I8MM on ARM, and Metal
on Apple GPUs. Linux, Windows and macOS.

On its native paths jam is competitive with llama.cpp's CPU kernels at matched instruction set.

## Quick start

`JAM.providers()` discovers the installed backends in priority order. Matmul is a bounds-checked
call on native `MemorySegment`s.

```java
JAM jam = JAM.providers().getFirst().create();
int st = jam.mm(w, a, r, JAM.Q8_0, m, n, k);                 // contiguous: F32 activations + result

// strided, with byte offsets (zero allocation over one large mmap'd buffer):
int s2 = jam.mm(w, wOff, JAM.Q8_0, k,   // weight: segment, byte offset, dtype, row stride
                a, aOff, JAM.F32,  k,   // activations
                r, rOff, JAM.F32,  m,   // result   ->  R = W @ Aᵀ
                m, n, k);
```

From C:

```c
#include <jam.h>

jam_status st = jam_mm(NULL,             // NULL = the global context
                       W, JAM_Q8_0, k,   // weights     [m x k]  (row stride k)
                       X, JAM_F32,  k,   // activations [n x k]
                       Y, JAM_F32,  m,   // result      [m x n]  (token-major, stride m)
                       m, n, k);         // R = W @ Aᵀ
```

Supported quantizations include `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `MXFP4` and `NVFP4`, plus
dense `F32`, `F16` and `BF16`. Activations and result are always `F32`. The operands must be
**native** segments, not heap arrays.

## Why jam

- **A single op.** `jam_mm` computes `R = W @ Aᵀ`. Matrix-vector products (gemv) are supported
  implicitly at `n == 1`.
- **Picks the fastest kernel.** jam detects the supported CPU features once and selects the best
  kernels, with no further per-call dispatch.
- **Parallel.** Every call runs across multiple threads.
- **No conversions.** Weights stay in their quantized format, byte-compatible with llama.cpp's
  `mul_mat`, so a `.gguf` tensor can be passed directly.
- **No third-party runtime dependencies.** `jam-native` bundles and loads the native library for
  the current OS and arch. Override its location with `-Djam.native.library.path` or
  `JAM_NATIVE_LIBRARY_PATH`. The available native toolchains determine which builds ship.

## Performance

Prefill throughput (`pp512`) of jinfer on the native jam backend and of llama.cpp, at matched instruction set:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-tiers-dark.png">
  <img alt="jinfer (native jam) vs llama.cpp, prefill by instruction set" src="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-tiers.png">
</picture>

The same int8 kernels span the whole x86 ladder, from the pre-AVX2 floor up to AVX-512:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-isa-dark.png">
  <img alt="jinfer (native jam) prefill by instruction set" src="https://raw.githubusercontent.com/qxoticai/assets/main/jam/bench-isa.png">
</picture>

One machine, one model; the numbers, the method and the sweep script are in
[docs/benchmarks.md](docs/benchmarks.md).

## Modules and launch flags

`JAM` is a minimal interface in the `com.qxotic.jam` module. Its providers ship separately:
`com.qxotic.jam.libjam` (`jam-native`, the default), `com.qxotic.jam.vector` (`jam-vector`), and
`com.qxotic.jam.scalar` (`jam-scalar`). Implement `JAM.Provider` to add another backend.

On the classpath, the Vector API backend requires:

```sh
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED ...
```

On the module path, require only the API. Providers are discovered from their module descriptors:

```java
module app {
    requires com.qxotic.jam;
}
```

Grant native access to the providers that use it:

```sh
java --enable-native-access=com.qxotic.jam.libjam,com.qxotic.jam.vector \
  --module-path ... --module app/com.example.Main
```

The scalar provider requires no launch flags.

## Backends

jam detects the CPU and uses the best available kernel. Cap it with `JAM_ISA` or `cfg.max_isa`.

| arch | instruction sets | Q8_0 dot |
|---|---|---|
| x86 | `sse3` → `ssse3` → `avx2` → `avx_vnni` → `avx512` → `avx512_vnni` | `vpdpbusd` (256/512-bit) |
| ARM | `neon` → `dotprod` → `i8mm` | `sdot` / `smmla` |
| GPU | `metal` (Apple Silicon, on by default) | MSL compute |

`JAM_ISA=auto` (the default) picks the best available; on Apple Silicon that includes the Metal
backend. Name a CPU rung (`JAM_ISA=i8mm`) to stay CPU-only. Backend routing, the packed weight
layouts and the threading contract are described in [docs/design.md](docs/design.md).

## Configuration

```sh
JAM_ISA=avx2                         ./app   # cap every provider at AVX2
JAM_ISA=i8mm                         ./app   # CPU-only on Apple Silicon (Metal is on by default)
JAM_DEBUG=1                          ./app   # print detected features + bound kernels
```

Threads are not a jam setting: a provider runs on the host's `JAM.Parallel` (in jinfer,
`-Djinfer.threads`); a C host without an executor gets a pool sized by `jam_config.nthreads`.

For per-pool control, create a context explicitly:

```c
jam_config cfg = {.nthreads = 8, .max_isa = JAM_ISA_AVX2};
jam_ctx* ctx = jam_ctx_create(&cfg);
jam_mm(ctx, /* ... */);
jam_ctx_destroy(ctx);
```

## Build

```sh
mvn -pl jam/jam-vector -am package -DskipTests   # from the repository root
```

Toolchains, the cmake-only build, the multi-platform release set and the test suites are in
[BUILDING.md](BUILDING.md).

Part of [Quixotic](../README.md), an open stack for local AI on the JVM.

## License

Apache 2.0
