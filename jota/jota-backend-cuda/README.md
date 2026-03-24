# Jota Backend CUDA

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

CUDA backend for Jota on NVIDIA GPUs.

CUDA backend is unsupported on macOS.

## Compile-time requirements

- Java 25+
- Maven (`mvnd` recommended)
- CMake 3.21+
- C compiler toolchain (`gcc` or `clang`)
- NVIDIA CUDA Toolkit development files
- JNI headers available from your JDK (`JAVA_HOME` must point to the JDK)

## Compile dependencies

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jota</artifactId>
  <version>${qxotic.version}</version>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jota-backend-cuda</artifactId>
  <version>${qxotic.version}</version>
</dependency>
```

For GraalVM Native Image applications:

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jota-graal</artifactId>
  <version>${qxotic.version}</version>
</dependency>
```

## Runtime dependencies

- Java 25+
- NVIDIA CUDA runtime + driver
- NVIDIA GPU with CUDA support
- `nvcc` available on `PATH`, unless overridden via config

## Bundled native library

The backend ships a minimal JNI shared library `jota_cuda`.

- The file name is platform-specific (for example, Linux: `libjota_cuda.so`).
- It is packaged in the backend JAR and auto-extracted at runtime.
- Manual library path overrides are still supported.

Supported packaged targets:

- Linux `x86_64`
- Linux `aarch64`
- Windows `x86_64`

## Configuration properties

Build-time:

- `native.cmake.executable`: CMake executable (default: `cmake`)
- `cuda.configure.extraArgs`: extra CMake configure arguments
- `cuda.build.extraArgs`: extra CMake build arguments

Runtime:

- `jota.cuda.compiler`: CUDA compiler executable (`NVCC` fallback, default `nvcc`)
- `jota.cuda.arch`: target GPU architecture (for example `sm_90`)
- `jota.cuda.compile.timeout.seconds`: kernel compile timeout in seconds (default: `10`)
- `jota.cuda.compile.opt`: optimization level (default: `2`)
- `jota.cuda.compile.flags`: extra CUDA compile flags (space-separated)
- `jota.cuda.pch.enabled`: precompiled header usage (`true`/`false`, default `false`)
- `jota.cuda.pch.log`: precompiled header logging (`true`/`false`)
- `jota.cuda.timing.log`: CUDA backend timing logs (`true`/`false`)
- `jota.verifyScratch`: scratch-buffer verification (`true`/`false`)

## Environment variables

- `JAVA_HOME`: JDK path used to resolve JNI headers for native build
- `NVCC`: compiler fallback when `jota.cuda.compiler` is not set
- `JOTA_CUDA_ARCH`: architecture fallback when `jota.cuda.arch` is not set
- `PATH`: must contain `nvcc`
- `LD_LIBRARY_PATH` (Linux), `PATH` (Windows): optional fallback locations for loading `jota_cuda` and CUDA runtime libraries
