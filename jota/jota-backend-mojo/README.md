# Jota Backend Mojo

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

Mojo backend for Jota on AMD GPUs (experimental, via HIP execution runtime).

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
  <artifactId>jota-backend-mojo</artifactId>
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
- Mojo toolchain (`mojo` CLI)
- HIP/ROCm runtime and compatible AMD GPU
- HIP toolchain (`hipcc`) available and functional (required by backend probe)

## Bundled native library

The backend ships a minimal JNI shared library `jota_mojo`.

- The file name is platform-specific (for example, Linux: `libjota_mojo.so`).
- It is packaged in the backend JAR and auto-extracted at runtime.
- Manual library path overrides are still supported.

Supported packaged targets:

- Linux `x86_64`

## Configuration properties

Build-time:

- `native.cmake.executable`: CMake executable (default: `cmake`)
- `mojo.configure.extraArgs`: extra CMake configure arguments
- `mojo.build.extraArgs`: extra CMake build arguments

Runtime:

- `jota.mojo.compiler`: Mojo compiler executable (`JOTA_MOJO_COMPILER` fallback, default `mojo`)
- `jota.mojo.target`: Mojo target architecture (`JOTA_MOJO_TARGET` fallback, default auto, then `gfx1103`)
- `jota.mojo.compile.flags`: extra Mojo compile flags (space-separated)
- `jota.mojo.compile.timeout.seconds`: kernel compile timeout in seconds (default: `30`)
- `jota.mojo.kernel.summary`: emits compile/cache summary at shutdown (`true`/`false`)
- `jota.verifyScratch`: scratch-buffer verification (`true`/`false`)

Execution backend dependencies (HIP path used by Mojo v1):

- `jota.hip.compiler`, `jota.hip.compile.flags`, `jota.hip.arch`, `jota.hip.compile.timeout.seconds`

## Environment variables

- `ROCM_PATH`: ROCm installation path hint for native build configuration
- `JOTA_MOJO_COMPILER`: compiler fallback when `jota.mojo.compiler` is not set
- `JOTA_MOJO_TARGET`: target fallback when `jota.mojo.target` is not set
- `HIPCC`: HIP compiler fallback used by HIP runtime probe
- `PATH`: must contain `mojo` and `hipcc`
- `LD_LIBRARY_PATH` (Linux), `PATH` (Windows): optional fallback locations for loading `jota_mojo`, `jota_hip`, and ROCm runtime libraries
