# jam

**Just a matmul.** The fastest one on the JVM.

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

JVM Accelerated Math. Fast quantized matrix multiplication for CPUs, from Java or C. One operation,
every instruction set: SSE3 through AVX-512-VNNI on x86, NEON, DotProd and I8MM on ARM, and Metal
on Apple GPUs. Linux, Windows and macOS.

On its native AVX-512-VNNI path, jam often beats llama.cpp's hand-tuned CPU kernels at matched ISA.

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

Prefill throughput (`pp512`, `R = W @ Aᵀ`), Llama-3.2-1B, 16 threads, Ryzen 9 9950X3D (Zen 5):

![jam vs llama.cpp prefill on AVX-512-VNNI](docs/bench-avx512.png)

On its flagship VNNI tier, jam wins four of five weight types. Q5_K is **2.2x** faster and Q6_K is
**1.4x** faster. The same int8 kernels span the whole x86 ladder, from the pre-AVX2 floor up to
AVX-512:

![jam ÷ llama.cpp across ISA tiers](docs/bench-ratio.png)

The sub-parity bars are the pre-VNNI Q4_0 and Q8_0, where the int8 dot has no `vpdpbusd` to lean
on. On the k-quants jam is at or above parity at every tier. These numbers cover one machine and
one model. Run `jam_bench` and a local `pp512` to measure other hardware.

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

Supported quantizations include `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `MXFP4` and `NVFP4`, plus
dense `F32`, `F16` and `BF16`. Activations and result are always `F32`. The operands must be
**native** segments, not heap arrays.

## Backends

jam detects the CPU and uses the best available kernel. Cap it with `JAM_ISA` or `cfg.max_isa`.

| arch | ISA ladder | Q8_0 dot |
|---|---|---|
| x86 | `sse3` → `ssse3` → `avx2` → `avx_vnni` → `avx512` → `avx512_vnni` | `vpdpbusd` (256/512-bit) |
| ARM | `neon` → `dotprod` → `i8mm` | `sdot` / `smmla` |
| GPU | `metal` (Apple Silicon, on by default) | MSL compute |

`JAM_ISA=auto` is the default and picks the best available. On Apple Silicon that includes the
Metal backend. Name a CPU rung (`JAM_ISA=i8mm`) to stay CPU-only, or `JAM_ISA=metal` to insist on
it. With Metal active the ctx keeps both executors and routes by measured shape. Dense weights go
to the GPU at every n, since its wider DRAM path wins even at n==1. Quantized weights go to the GPU
only for n>=16, that is prefill, where the block quants run simdgroup-matrix MMA kernels (half
operands, float accumulation, 64x32 tiles), while one-column and small-n decode stays on the CPU
SDOT and I8MM kernels. All Metal calls are zero-copy: W, A and C are borrowed through page-rounded
`newBufferWithBytesNoCopy` views over the caller's unified-memory pages and released after the
synchronous wait, so there are no uploads and no result copies, and strided views are consumed
directly. `JAM_METAL_PROFILE=1` prints per-call encode, submit, wait and GPU averages at context
destroy. SVE, AMX and SME are not yet implemented.

### Packed weights

Decode streams every weight byte per token, and the GGUF block layouts waste bandwidth (unaligned
fp16 scales) or instructions (k-quant bit unpacking) there. jam therefore defines packed in-memory
layouts for Q4_0, Q4_K, Q5_K and Q6_K (per-4-row-group sections, specified next to the dtype tags
in `jam.h`, never a wire format). The contract is caller-packs, jam-reads, one copy:
`jam_pack_size(ctx, dt, m, k)` says whether this ctx's kernels want the layout for a `[m x k]`
weight, and how many bytes it is. The caller produces the bytes once at load, drops the canonical
copy, and passes `wt | JAM_PACKED` to `jam_mm`. Every engine reads that same copy: the 4x1 decode
GEMVs, the 4x4 sdot prefill kernels, and, on Metal and zero-copy via unified memory, the packed MMA
kernels. Values are exactly the canonical dequant. `jam_pack_abi()` guards packers against layout
drift between jam versions.

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

A `jam_ctx` is a serial stream: one `mm` at a time. For concurrent matmuls, use one context per
thread.

## Build

Everything is built, tested and packaged locally, inside VMs. There is no CI. A build needs
**CMake ≥ 3.16**, a **C11 compiler** (GCC or Clang; MSVC is rejected at configure), **Maven**, and
a **JDK ≥ 25**. On macOS, Apple silicon only, `xcode-select --install` covers clang, cmake and the
Metal frameworks. Windows builds use **MinGW-w64**, normally cross-compiled from Linux (below).

**Maven** runs cmake, javac and the tests in one step, from the **repository root**. jam is not
dependency-closed (`jam-core` builds on `jota-core`), so `mvn` inside this directory only works
after a root `mvn install`:

```sh
mvn -pl jam/jam-vector -am package -DskipTests   # jars under jam/*/target (the native lib builds first)
mvn -pl jam/jam-vector -am verify                # configure + build + JUnit incl. the parity suite
```

**Or build just the native library with cmake**, with no JVM, for the C API or to pre-stage
`dist/native/`:

```sh
cmake -B build                # Release by default
cmake --build build           # -> build/libjam.so, staged into dist/native/<os>-<arch>/
```

Flags: `-DJAM_METAL=OFF` (no Metal), `-DJAM_JNI=OFF` (C only, drops the JDK requirement),
`-DJAM_TESTS=OFF`, `-DJAM_STRIP=ON`, `-DJAM_SANITIZE=ON` (ASan+UBSan test build).

A build carries **every kernel tier of its target architecture**, since any x86-64 compiler emits
the AVX-512 TUs, and the library binds the best tier the CPU supports at runtime, so one artifact
runs on any CPU of its arch. Compilers older than GCC 11 or Clang 12 skip the AVX-VNNI tier
(probed, never fatal).

### The fat jar (every platform)

`jam.jar` bundles one native library per platform under `com/qxotic/jam/native/<os>-<arch>/`.
`NativeLoader` extracts and loads the matching one at first use, falling back to the pure-Java
backends when none matches. The shipped set, and where each library is built:

| artifact | built on | toolchain |
|---|---|---|
| `linux-x86-64/libjam.so` | the Linux release host | `zig cc` targeting glibc 2.17 + `cmake/toolchains/linux-x86-64-glibc2.17.cmake` |
| `windows-x86-64/jam.dll` | the Linux release host | MinGW-w64 (`x86_64-w64-mingw32-gcc`) + `cmake/toolchains/windows-x86-64.cmake` |
| `darwin-aarch64/libjam.dylib` | an Apple-silicon Mac, over ssh | Xcode CLT (clang, ObjC++, Metal); Intel Macs unsupported |

Other platforms (linux-aarch64 among them) get the Java backends; `cmake/toolchains/` keeps a
linux-aarch64 cross file for local builds, but nothing ships it.

One script builds the whole set from one source tree into `dist/release` (apart from `dist/native`,
where every ordinary build stages this host's library) and stamps each library with the digest of
the native sources it came from and its own checksum:

```sh
JAM_MAC=user@mac jam-native/scripts/natives.sh build   # or: make jam-natives (from the repo root)
```

The Linux host builds its own library and the Windows one. The Linux library is compiled with
[zig](https://ziglang.org) as the C compiler, targeting `x86_64-linux-gnu.2.17`: zig bundles the
glibc stubs for every version, so the artifact runs on RHEL 7 or Ubuntu 14.04 class systems without
a container or sysroot, and the script fails if the link ever needs a newer glibc symbol. The
Windows library comes from MinGW-w64 (the checked-in `cmake/cross-jni/win32/` shim supplies the
target `jni_md.h`; any JDK's `jni.h` works). The Mac leg rsyncs the source tree to `JAM_MAC`,
builds it there with Metal on, runs the C test suite on the real GPU, and fetches the dylib back.
`JAM_TARGETS="linux-x86-64"` builds a subset while iterating on one leg.

The release build (`-Prelease`) never runs cmake: it packages the staged set after
`scripts/natives.sh verify` has checked that every shipped library is present, unchanged since its
build, built from the current native sources, and exports every symbol the Java side binds.
Libraries used to accumulate across builds until a pre-rename `jam.dll` shipped and crashed on
load; the stamps make that impossible. `NativeJAMProvider` also binds every symbol before it
reports the backend available, so an unusable bundle degrades to the Java backends with a warning
instead of failing on the first matmul.

Host requirements: cmake, a JDK, zig 0.16, MinGW-w64 and rsync on the Linux host; Xcode Command
Line Tools, a Homebrew cmake and any JDK on the Mac, reachable by key-based ssh. The script checks
all of that first and only then wipes and rebuilds the release set, so a misconfigured host fails
in seconds with the previous set intact.

## Tests

```sh
cd build && ctest --output-on-failure   # every kernel, 1 & 3 threads, vs a double-precision reference
./jam_bench [M N K] [iters]             # GMAC/s (compute) and GB/s (bandwidth)
```

Part of [Quixotic](../README.md), an open stack for local AI on the JVM.

## License

Apache 2.0
