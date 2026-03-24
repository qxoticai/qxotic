# Jota Backend C

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

C backend for Jota CPU execution.

## Compile-time requirements

- Java 25+
- Maven (`mvnd` recommended)
- CMake 3.16+
- C compiler toolchain (`gcc` or `clang`)
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
  <artifactId>jota-backend-c</artifactId>
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
- C compiler on `PATH` (`gcc` or `clang`), unless overridden via config
- `jota_c` native library loaded from bundled resources or system library path

## Bundled native library

The backend ships a minimal JNI shared library `jota_c`.

- The file name is platform-specific (for example, Linux: `libjota_c.so`).
- It is packaged in the backend JAR and auto-extracted at runtime.
- Manual library path overrides are still supported.

Supported packaged targets:

- Linux `x86_64`
- Linux `aarch64`
- Windows `x86_64`
- macOS `aarch64`

## Configuration properties

Build-time:

- `native.cmake.executable`: CMake executable (default: `cmake`)
- `c.configure.extraArgs`: extra CMake configure arguments
- `c.build.extraArgs`: extra CMake build arguments

Runtime:

- `jota.c.compiler`: compiler executable (`CC` fallback, default `gcc`)
- `jota.c.compile.timeout.seconds`: compile timeout in seconds (default: `10`)
- `jota.c.compile.opt`: optimization level (default: `2`)
- `jota.c.compile.flags`: extra compile flags (space-separated)
- `jota.c.link.flags`: extra link flags (space-separated)
- `jota.c.openmp`: enable/disable OpenMP (`true`/`false`, auto by platform if unset)
- `jota.c.openmp.compile.flags`: override OpenMP compile flags
- `jota.c.openmp.link.flags`: override OpenMP link flags

## Environment variables

- `CC`: compiler executable fallback when `jota.c.compiler` is not set
- `PATH`: must contain compiler executable
- `LD_LIBRARY_PATH` (Linux), `DYLD_LIBRARY_PATH` (macOS), `PATH` (Windows): optional fallback locations for loading `jota_c`
