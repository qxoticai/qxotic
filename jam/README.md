<h1 align="center"><strong>jam</strong></h1> 

<p align="center"><strong>JVM Accelerated Math.</strong></p>

<p align="center">
  <a href="https://openjdk.org/projects/jdk/25/"><img src="https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white" alt="Java 25+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License"></a>
  <a href="https://www.graalvm.org/latest/reference-manual/native-image/"><img src="https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F" alt="GraalVM Native Image"></a>
  <a href="https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey">
</p>
    
A.k.a jokingly as _"Just A Matmul"_. JAM implements fast quantized matrix multiplication routines for CPUs, with APIs for Java and C.  
A single, safe, entry point: `mm` dispatch to specialized kernels for a several instruction sets:  
SSE3 through AVX-512-VNNI on x86, NEON, DotProd and I8MM on ARM, and Metal on Apple GPUs.

The JAM native kernels are competitive with llama.cpp's CPU kernels across several instruction sets.

## Quick start

```java
JAM jam = JAM.providers().getFirst().create();
int st = jam.mm(w, a, r, JAM.Q8_0, m, n, k);                 // contiguous: F32 activations + result

// Strided, with byte offsets:
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

Supported quantizations: `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `MXFP4` and `NVFP4`, dense `F32`, `F16` and `BF16`.  
Activations and result are always `F32`. The operands must be **native** segments, not heap arrays.

`JAM.providers()` discovers the available backends from the classpath. The `mm` operation is bounds-checked `MemorySegment`s.

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

jam automatically detects the CPU features, the number of cores, discards low-power cores and selects the best available kernels.  
Manually control the target instruction set  with `JAM_ISA` or `cfg.max_isa`.

| Arch | Instruction Sets | Q8_0 dot |
|---|---|---|
| x86 | `sse3` → `ssse3` → `avx2` → `avx_vnni` → `avx512` → `avx512_vnni` | `vpdpbusd` (256/512-bit) |
| ARM | `neon` → `dotprod` → `i8mm` | `sdot` / `smmla` |
| GPU | `metal` (Apple Silicon, on by default) | MSL compute |

`JAM_ISA=auto` (the default) selects the best available kernels; on Apple Silicon that includes the Metal 
backend. Set a CPU instruction-set (`JAM_ISA=i8mm`) to stay CPU-only. Backend dispatch, the packed weight
layouts and the threading contract are described in [docs/design.md](docs/design.md).

## Configuration

```sh
JAM_ISA=avx2                         ./app   # pin every provider at AVX2
JAM_ISA=i8mm                         ./app   # CPU-only on Apple Silicon (Metal is on by default)
JAM_DEBUG=1                          ./app   # print detected features + bound kernels
```

Thread pools are configurable: a JAM backend can use the host's provided `JAM.Parallel` pool (in jinfer,
`-Djinfer.threads`); a C host without an executor gets a native thread pool sized by `jam_config.nthreads`.

For fine-grained thread-pool control, create a JAM native context explicitly:

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
