---
sidebar_position: 3
---

# jam

**JVM Accelerated Math** (*just a matmul*). Fast quantized matrix multiplication for CPUs, from Java or C.

Linux, Windows, and macOS. x86: SSE3 through AVX-512-VNNI. ARM: NEON / DotProd / I8MM. Apple GPU: Metal.

## Why jam?

- **One op.** `jam_mm` computes `R = W @ Aᵀ`. Gemv is implicit at `n == 1`.
- **Picks the fastest kernel.** Detects CPU features once, selects the best kernels, no per-call dispatch.
- **Parallel.** Every call runs across multiple threads.
- **No conversions.** Weights stay quantized, byte-compatible with llama.cpp's `mul_mat`; pass a `.gguf` tensor directly.
- **No third-party runtime dependencies.** `jam-native` bundles and loads the native library for the current OS/arch. Override its location with `-Djam.native.library.path` or `JAM_NATIVE_LIBRARY_PATH`.

## Quick start

### C

```c
#include <jam.h>

jam_status st = jam_mm(NULL,             // NULL = the global context
                       W, JAM_Q8_0, k,   // weights     [m x k]  (row stride k)
                       X, JAM_F32,  k,   // activations [n x k]
                       Y, JAM_F32,  m,   // result      [m x n]  (token-major, stride m)
                       m, n, k);         // R = W @ Aᵀ
```

### Java

`JAM.providers()` discovers the installed backends in priority order. Matmul is a bounds-checked call on native `MemorySegment`s.

```java
JAM jam = JAM.providers().getFirst().create();
int st = jam.mm(w, a, r, JAM.Q8_0, m, n, k);                 // contiguous: F32 activations + result

// strided, with byte offsets (zero allocation over one large mmap'd buffer):
int s2 = jam.mm(w, wOff, JAM.Q8_0, k,   // weight: segment, byte offset, dtype, row stride
                a, aOff, JAM.F32,  k,   // activations
                r, rOff, JAM.F32,  m,   // result   ->  R = W @ Aᵀ
                m, n, k);
```

`JAM` is the API in `com.qxotic.jam`. Its providers ship as separate modules:

- `com.qxotic.jam.libjam` (`jam-native`), the default
- `com.qxotic.jam.vector` (`jam-vector`), the Java Vector API backend
- `com.qxotic.jam.scalar` (`jam-scalar`), the portable Java reference

Implement `JAM.Provider` to add your own.

Classpath applications using `jam-vector` need:

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

Quantizations: `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `MXFP4`, `NVFP4`, plus dense `F32`/`F16`/`BF16`. Activations and result are always `F32`. Operands must be **native** segments, not heap arrays.

## Backends

jam detects the CPU and uses the best available kernel. Cap it with `JAM_ISA` or `cfg.max_isa`.

| arch | ISA ladder | Q8_0 dot |
|---|---|---|
| x86 | `sse3` → `ssse3` → `avx2` → `avx_vnni` → `avx512` → `avx512_vnni` | `vpdpbusd` (256/512-bit) |
| ARM | `neon` → `dotprod` → `i8mm` | `sdot` / `smmla` |
| GPU | `metal` (Apple Silicon, on by default) | MSL compute |

`JAM_ISA=auto` (default) picks the best. On Apple Silicon that includes the Metal backend (prefill on the GPU, one-column decode stays on the CPU SDOT kernels). Name a CPU rung (`JAM_ISA=i8mm`) to stay CPU-only. SVE, AMX, and SME are not implemented.

## Performance

On its native AVX-512-VNNI path, jam beats llama.cpp's CPU kernels at matched ISA for four of five weight types (Q5_K by 2.2×, Q6_K by 1.4×). The same int8 kernels span the whole x86 ladder, pre-AVX2 through AVX-512.

These are one machine / one model. Run `jam_bench` and llama.cpp's `pp512` to measure your hardware.

## Configuration

```sh
JAM_ISA=avx2                         ./app   # cap every provider at AVX2
JAM_ISA=i8mm                         ./app   # CPU-only on Apple Silicon (Metal is on by default)
JAM_DEBUG=1                          ./app   # print detected features + bound kernels
```

Threads are not a jam setting.
A provider is created with the host's `JAM.Parallel` (`Provider.create(parallel)`) and runs every task on it, its width being the thread budget; in jinfer that is `-Djinfer.threads`.
A backend may run its own threads instead, under one rule: while `mm` runs it uses at most `width()` cores and the host's workers are idle, and when `mm` returns its own workers are idle.
The native library creates no thread inside a JVM; a C host without an executor of its own gets a small pool sized by `jam_config.nthreads` (0 = every online CPU).

For per-pool control, create a context explicitly:

```c
jam_config cfg = {.nthreads = 8, .max_isa = JAM_ISA_AVX2};
jam_ctx* ctx = jam_ctx_create(&cfg);
jam_mm(ctx, /* ... */);
jam_ctx_destroy(ctx);
```

A `jam_ctx` is a serial stream: one `mm` at a time. For concurrent matmuls, use one context per thread.

## Build

Requirements: **CMake ≥ 3.16**, a **C11 compiler** (clang preferred), **JDK ≥ 25**. On macOS, `xcode-select --install` covers clang, cmake, and Metal. On Windows, clang is required (MSVC can't build the SIMD kernels).

**Maven** runs cmake, javac, and tests in one step:

```sh
mvn package      # -> dist/jam.jar  (native lib built for you)
mvn test         # configure + build + JUnit
```

**Or build just the native library with cmake** (for the C API, or to pre-stage `dist/native/`):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build           # -> build/libjam.so, staged into dist/native/
```

Flags: `-DJAM_METAL=OFF` (no Metal), `-DJAM_JNI=OFF` (C only), `-DJAM_TESTS=OFF`, `-DJAM_STRIP=ON`. `mvn package -Djam.native.skip=true` reuses a pre-staged `dist/native/`.

Each host builds only the kernels it can run; the library picks the best at runtime. CI builds each platform natively and merges them into one fat `jam.jar`.

## Tests

```sh
cd build && ctest --output-on-failure   # every kernel, 1 & 3 threads, vs a double-precision reference
./jam_bench [M N K] [iters]             # GMAC/s (compute) and GB/s (bandwidth)
```
