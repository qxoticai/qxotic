# Jota Backend OpenCL

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

OpenCL backend for Jota on OpenCL-capable GPUs/CPUs.

## Compile-time requirements

- Java 25+
- Maven (`mvnd` recommended)
- CMake 3.16+
- C compiler toolchain (`gcc` or `clang`)
- OpenCL development files (headers + ICD loader library)
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
  <artifactId>jota-backend-opencl</artifactId>
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
- OpenCL ICD/runtime installed and discoverable
- At least one OpenCL-capable device (GPU or CPU)

## Bundled native library

The backend ships a minimal JNI shared library `jota_opencl`.

- The file name is platform-specific (for example, Linux: `libjota_opencl.so`).
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
- `opencl.configure.extraArgs`: extra CMake configure arguments
- `opencl.build.extraArgs`: extra CMake build arguments

Runtime:

- `jota.opencl.device.type`: preferred device type selector
- `jota.opencl.platform.index`: preferred OpenCL platform index
- `jota.opencl.device.index`: preferred device index within selected platform/type
- `jota.opencl.device.name.contains`: substring match for device selection
- `jota.opencl.compile.flags`: OpenCL compiler flags passed to runtime kernel compilation
- `jota.opencl.interpreter.fallback`: fallback to interpreter on kernel failure (`true`/`false`, default `true`)
- `jota.verifyScratch`: scratch-buffer verification (`true`/`false`)

## Environment variables

- `LD_LIBRARY_PATH` (Linux), `DYLD_LIBRARY_PATH` (macOS), `PATH` (Windows): optional fallback locations for loading `jota_opencl` and OpenCL ICD/runtime libraries
