---
sidebar_position: 3
---

# jam

**JVM Accelerated Math** (*just a matmul*). Fast quantized matrix multiplication for CPUs, from Java or C.

Linux, Windows, and macOS. x86: SSE3 through AVX-512-VNNI. ARM: NEON / DotProd / I8MM. Apple GPU: Metal.

## Why jam?

- **One op.** `jam_mm` computes `R = W @ Aᵀ`. Gemv is implicit at `n == 1`.
- **Picks the fastest kernel.** Detects CPU features once, selects the best kernels, no per-call dispatch.
- **Parallel.** Calls fan out over the host's `JAM.Parallel` (or the C context's pool).
- **No conversions.** Weights stay quantized, byte-compatible with llama.cpp's `mul_mat`; pass a `.gguf` tensor directly.
- **No third-party runtime dependencies.** `jam-native` bundles and loads the native library for the current OS/arch. Override its location with `-Djam.native.library.path` or `JAM_NATIVE_LIBRARY_PATH`.

## Quick start

### C

```c
#include <jam.h>

jam_context* context = jam_context_new(NULL);   // the best kernels on every CPU; NULL on failure
int32_t status = jam_mm(context,
                        W, JAM_Q8_0, k,   // weights     [m x k]  (row stride k)
                        X, JAM_F32,  k,   // activations [n x k]
                        Y, JAM_F32,  m,   // result      [n x m]  (token-major, row stride m)
                        m, n, k);         // R = W @ A^T
if (status < 0) fprintf(stderr, "jam: %s\n", jam_last_error(NULL));
jam_context_free(context);
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

Quantizations: `Q1_0`, `Q2_0`, `PQ2_0`, `PTQ1_0`, `Q4_0`, `Q5_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `MXFP4`, `NVFP4`, plus dense `F32`/`F16`/`BF16`. Activations and result are always `F32`. Operands must be **native** segments, not heap arrays.

## Backends

jam detects the CPU and uses the best available kernel.
Cap it with `-Djam.native.isa` (or `JAM_NATIVE_ISA`) in Java, `max_isa` in C.

| arch | instruction sets | Q8_0 dot |
|---|---|---|
| x86 | `sse3` → `ssse3` → `avx2` → `avx_vnni` → `avx512` → `avx512_vnni` | `vpdpbusd` (256/512-bit) |
| ARM | `neon` → `dotprod` → `i8mm` | `sdot` / `smmla` |
| GPU | `metal` (Apple Silicon, on by default) | MSL compute |

Unset (or `auto`), it picks the best.
On Apple Silicon that includes the Metal backend (prefill on the GPU, one-column decode stays on the CPU SDOT kernels).
Cap it at a CPU level (`i8mm`) to stay CPU-only.
SVE, AMX, and SME are not implemented.

## Performance

On its native paths jam is competitive with llama.cpp's CPU kernels at matched instruction set. The same int8 kernels span the whole x86 ladder, pre-AVX2 through AVX-512.

Run `jam_bench` and llama.cpp's `pp512` to measure your hardware.

## Configuration

```sh
JAM_NATIVE_ISA=avx2                  ./app   # cap native jam at AVX2 (or -Djam.native.isa=avx2)
JAM_NATIVE_ISA=i8mm                  ./app   # CPU-only on Apple Silicon (Metal is on by default)
JAM_DEBUG=1                          ./app   # print detected features + bound kernels
```

Threads are not a jam setting.
A provider is created with the host's `JAM.Parallel` (`Provider.create(parallel)`) and runs every task on it, its width being the thread budget; in jinfer that is `-Djinfer.threads`.
A backend may run its own threads instead, under one rule: while `mm` runs it uses at most `width()` cores and the host's workers are idle, and when `mm` returns its own workers are idle.
The native library creates no thread inside a JVM; a C host without an executor of its own gets a small pool sized by `jam_context_options.threads` (0 = every online CPU).

For per-pool control, create a context explicitly:

```c
jam_context_options options = {.threads = 8, .max_isa = JAM_ISA_AVX2};
jam_context* context = jam_context_new(&options);   /* NULL on failure: jam_last_error says why */
jam_mm(context, /* ... */);
jam_context_free(context);
```

A `jam_context` is a serial stream: one `mm` at a time.
For concurrent matmuls, use one context per thread.

## Build

Requirements: **CMake ≥ 3.16**, a **C11 compiler** (GCC or Clang), **JDK ≥ 25**. On macOS, `xcode-select --install` covers clang and Metal; install CMake with `brew install cmake`. On Windows the toolchain is clang: zig's, cross-compiling from Linux, or clang on the MSVC ABI; MSVC is refused because the kernel flags are GCC/Clang spellings.

**Maven** runs cmake, javac, and tests in one step, from the repository root once a root `mvn install` has built `jam-core` and `jota-core`:

```sh
mvn -f jam/jam-native/pom.xml package   # -> jam/jam-native/target/jam-native-0.3.1.jar with this host's library
mvn -f jam/jam-native/pom.xml test      # configure + build + JUnit
```

**Or build just the native library with cmake** (for the C API, or to pre-stage `dist/native/`):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build           # -> build/libjam.so, staged into dist/native/
```

Flags: `-DJAM_METAL=OFF` (no Metal), `-DJAM_JNI=OFF` (C only), `-DJAM_TESTS=OFF`, `-DJAM_STRIP=ON`. `mvn package -Djam.native.skip=true` reuses a pre-staged `dist/native/`.

Every build carries all kernel tiers of its architecture; the library binds the best one at runtime. The shipped libraries (linux-x86-64, linux-aarch64, windows-x86-64, darwin-aarch64) are built by `jam-native/scripts/natives.sh` at one commit on two hosts: a Linux host builds the Linux x86-64 one with zig against glibc 2.17 and cross-compiles the Windows one with zig, an Apple-silicon Mac builds the macOS one and, in its manylinux aarch64 container, the Linux aarch64 one with zig, and the script imports the Mac's libraries into the Linux host's release set. There is no CI.

## Tests

```sh
mvn -f jam/jam-native/pom.xml test     # JNI + FFM on AUTO; host-pool bridge on generic CPU
cd build && ctest --output-on-failure   # every kernel, 1 & 3 threads, vs a double-precision reference
./jam_bench [M N K] [iters]             # GMAC/s (compute) and GB/s (bandwidth)
```

On Apple Silicon, the AUTO Java passes exercise Metal. Maven runs `HostPoolTest` separately with `JAM_NATIVE_ISA=generic` because a GPU matmul has no CPU work and therefore never calls the host pool.
