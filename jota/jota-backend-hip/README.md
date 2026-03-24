# Jota Backend HIP

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

HIP/ROCm backend for Jota on AMD GPUs.

## Compile-time requirements

- Java 25+
- Maven (`mvnd` recommended)
- CMake 3.16+
- C compiler toolchain (`gcc` or `clang`)
- ROCm development files (HIP headers and `amdhip64` library)
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
  <artifactId>jota-backend-hip</artifactId>
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
- AMD ROCm runtime
- AMD GPU with HIP/ROCm support
- `hipcc` available on `PATH`, unless overridden via config

## Bundled native library

The backend ships a minimal JNI shared library `jota_hip`.

- The file name is platform-specific (for example, Linux: `libjota_hip.so`).
- It is packaged in the backend JAR and auto-extracted at runtime.
- Manual library path overrides are still supported.

Supported packaged targets:

- Linux `x86_64`
- Linux `aarch64`
- Windows `x86_64`

## Configuration properties

Build-time:

- `native.cmake.executable`: CMake executable (default: `cmake`)
- `hip.configure.extraArgs`: extra CMake configure arguments
- `hip.build.extraArgs`: extra CMake build arguments

Runtime:

- `jota.hip.compiler`: HIP compiler executable (`HIPCC` fallback, default `hipcc`)
- `jota.hip.arch`: target offload architecture (for example `gfx1100`)
- `jota.hip.compile.timeout.seconds`: kernel compile timeout in seconds (default: `10`)
- `jota.hip.compile.opt`: optimization level (default: `2`)
- `jota.hip.compile.flags`: extra HIP compile flags (space-separated)
- `jota.hip.pch.enabled`: precompiled header usage (`true`/`false`, default `true`)
- `jota.hip.pch.log`: precompiled header logging (`true`/`false`)
- `jota.hip.timing.log`: HIP backend timing logs (`true`/`false`)
- `jota.verifyScratch`: scratch-buffer verification (`true`/`false`)

## Environment variables

- `ROCM_PATH`: ROCm installation path hint for native build configuration
- `HIPCC`: HIP compiler executable fallback when `jota.hip.compiler` is not set
- `JOTA_HIP_ARCH`: architecture fallback when `jota.hip.arch` is not set
- `PATH`: must contain `hipcc`
- `LD_LIBRARY_PATH` (Linux), `PATH` (Windows): optional fallback locations for loading `jota_hip` and ROCm runtime libraries
