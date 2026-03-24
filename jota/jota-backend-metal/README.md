# Jota Backend Metal

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

Metal backend for Jota on macOS.

## Compile-time requirements

- Java 25+
- Maven (`mvnd` recommended)
- CMake 3.16+
- Apple toolchain with Objective-C++ support (Xcode/Command Line Tools)
- JNI headers available from your JDK (`JAVA_HOME` must point to the JDK)
- macOS `aarch64` (native backend build requires Apple platform)

## Compile dependencies

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jota</artifactId>
  <version>${qxotic.version}</version>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jota-backend-metal</artifactId>
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
- macOS `aarch64` (Apple Silicon)
- Xcode command line tools available (`xcrun`, `metal`, `metallib`)

## Bundled native library

The backend ships a minimal JNI shared library `jota_metal`.

- The file name is platform-specific (for example, macOS: `libjota_metal.dylib`).
- It is packaged in the backend JAR and auto-extracted at runtime.
- Manual library path overrides are still supported.

Supported packaged targets:

- macOS `aarch64`

## Configuration properties

Build-time:

- `native.cmake.executable`: CMake executable (default: `cmake`)
- `metal.configure.extraArgs`: extra CMake configure arguments
- `metal.build.extraArgs`: extra CMake build arguments

Runtime:

- `jota.metal.compiler`: Metal tool driver executable (`JOTA_METAL_COMPILER` fallback, default `xcrun`)
- `jota.metal.compile.flags`: extra flags for `xcrun ... metal` (space-separated)
- `jota.metal.link.flags`: extra flags for `xcrun ... metallib` (space-separated)
- `jota.metal.compile.timeout.seconds`: kernel compile timeout in seconds (default: `15`)
- `jota.metal.compile.opt`: optimization level (default: `2`)
- `jota.metal.interpreter.fallback`: fallback to interpreter on kernel failure (`true`/`false`, default `true`)
- `jota.verifyScratch`: scratch-buffer verification (`true`/`false`)

## Environment variables

- `JOTA_METAL_COMPILER`: compiler driver fallback when `jota.metal.compiler` is not set
- `PATH`: must contain `xcrun`
- `DYLD_LIBRARY_PATH`: optional fallback location for loading `jota_metal`
