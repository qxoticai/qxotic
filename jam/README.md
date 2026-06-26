# jam

[![Java 22+](https://img.shields.io/badge/Java-22%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/22/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)

**Java Accelerated Math** -- FAST quantized matrix multiplication for CPUs.

**x86 SSE3/AVX2/AVX-VNNI/AVX-512 · ARM NEON/dotprod/i8mm · Apple Metal · GGUF-native dtypes**

---

## Why jam?

- **One op, one job.** `jam_mm` computes `R = W @ A^T`. Decode is just `n == 1`.
- **Picks the fastest kernel.** Detects CPU features once, binds a kernel pointer. No dispatch overhead.
- **Always parallel.** Every kernel fans out across threads within a single call.
- **No conversion.** Block decode and activation requant happen inside the kernel. Weights are
  byte-compatible with llama.cpp (GGUF `mul_mat` layout).
- **Fat JAR, zero install.** `NativeLoader` extracts the right `libjam` by `os.arch` at runtime.
  Override with `-Djam.library.path` or `JAM_LIBRARY_PATH`.

---

## Quick start

### C

```c
#include <jam.h>

jam_status st = jam_mm(NULL,                // NULL -> global context
                       W, JAM_Q8_0, in,     // weights      [m x k]
                       X, JAM_F32,  in,     // activations  [n x k]
                       Y, JAM_F32,  tokens, // result       [m x n]
                       out, tokens, in);    // m, n, k
```

### Java

```java
int st = JAM.mm(wAddr, JAM.Q8_0, in,        // single native method
                xAddr, JAM.F32,  in,
                yAddr, JAM.F32,  tokens,
                out, tokens, in);            // R = W @ A^T
```

Supported weight dtypes: `F32`, `F16`, `BF16`, `Q4_0`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`, `MXFP4`, `NVFP4`.
Activations and output are always `F32`.

---

## Backends

`jam` detects the CPU at context creation and binds the best kernel per path.
Cap with `JAM_ISA` or `cfg.max_isa`.

| arch | ISA ladder | Q8_0 dot instruction |
|---|---|---|
| x86 | `sse3` -> `avx2` -> `avx_vnni` -> `avx512` -> `avx512_vnni` | `vpdpbusd` (256/512-bit) |
| ARM | `neon` -> `dotprod` -> `i8mm` | `sdot` / `smmla` |
| GPU | `metal` (Apple, opt-in) | MSL compute |

`JAM_ISA=auto` (default) picks the best. `JAM_ISA=metal` runs on Apple GPU.
`sve` / `amx` / SME reserved.

---

## Configuration

```sh
JAM_NUM_THREADS=16 JAM_ISA=avx2  ./app    # 16 threads, cap at AVX2
JAM_ISA=metal                    ./app    # Apple GPU
JAM_DEBUG=1                      ./app    # print CPU features + bound kernels
```

Explicit context for per-pool control:

```c
jam_config cfg = {.nthreads = 8, .max_isa = JAM_ISA_AVX2};
jam_ctx* ctx = jam_ctx_create(&cfg);
jam_mm(ctx, /* ... */);
jam_ctx_destroy(ctx);
```

A `jam_ctx` is a serial stream -- one `mm` at a time. For concurrent matmuls, create multiple contexts.

---

## Build

### Dependencies

| Dep | Debian/Ubuntu | Fedora | Arch | macOS | Windows |
|---|---|---|---|---|---|---|
| CMake >= 3.16 | `sudo apt install cmake` | `sudo dnf install cmake` | `sudo pacman -S cmake` | `xcode-select --install` | `choco install cmake` |
| C11 compiler (clang) | `sudo apt install clang` | `sudo dnf install clang` | `sudo pacman -S clang` | `xcode-select --install` | `choco install llvm` |
| JDK >= 22 | `sudo apt install openjdk-22-jdk` | `sudo dnf install java-22-openjdk-devel` | `sudo pacman -S jdk-openjdk` | `brew install openjdk@22` | `choco install temurin22` |
| Ninja | `sudo apt install ninja-build` | `sudo dnf install ninja-build` | `sudo pacman -S ninja` | `brew install ninja` | `choco install ninja` |

macOS: `xcode-select --install` installs clang + cmake + Metal/Foundation frameworks in one command.
Ninja is optional; install separately with `brew install ninja`.
Windows: clang is mandatory -- MSVC does not support GCC-style `-m` flags (`-mavx2`, `-mavx512f`, etc.).
Linux: gcc also works, but clang is recommended for consistency across platforms.

### Build commands

**Maven** (recommended) -- runs cmake configure + build + javac + tests in one shot:

```sh
mvn package             # -> dist/jam.jar (native lib auto-built via exec-maven-plugin)
mvn test                # configure + build + JUnit tests
```

**Manual** cmake:

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build                     # -> build/libjam.so, staged to dist/native/
./scripts/build-jar.sh                  # -> dist/jam.jar
```

**Build options:**

```sh
cmake -B build -DJAM_METAL=OFF    # disable Metal GPU backend (macOS only)
cmake -B build -DJAM_JNI=OFF      # C-only library, no JNI binding
cmake -B build -DJAM_TESTS=OFF    # skip test/benchmark executables
cmake -B build -DJAM_STRIP=ON     # strip debug symbols
mvn package -Djam.native.skip=true       # skip cmake (reuse pre-staged dist/native/)
```

### Build resilience

Per-ISA kernels live in their own translation units and compile with their own `-m` flags.
The dispatcher and scalar floor stay at baseline, so the library runs on any CPU.

ISA-specific TUs are guarded by `CMAKE_SYSTEM_PROCESSOR` -- x86 TUs only build on x86-64 hosts,
ARM TUs only build on aarch64 hosts, Metal only builds on Apple (skip with `-DJAM_METAL=OFF`).
The build never attempts to cross-compile kernels the host cannot produce.

**Cross-platform JAR assembly** is done by CI (`.github/workflows/build.yml`): 5 OS/arch matrix
jobs each build natively, upload `dist/native/`, and a final merge job runs `build-jar.sh` to
produce a single `jam.jar` with libraries for every platform.

---

## Tests

```sh
cd build && ctest --output-on-failure    # every kernel, 1 and 3 threads, vs double-precision ref
./jam_bench [M N K] [iters]              # GMAC/s (compute-bound) and GB/s (bandwidth-bound)
```

---

## What jam does NOT do

- **No tensor allocation.** You provide the buffers.
- **No quantization.** You quantize weights; jam multiplies them.
- **No model I/O.** Raw pointers only. GGUF loading is [`gguf`](../gguf)'s job.

---

## License

Apache 2.0
