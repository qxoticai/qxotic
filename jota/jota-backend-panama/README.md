# Jota Backend Panama

[![Java](https://img.shields.io/badge/Java-25+-blue)](https://openjdk.org/projects/jdk/25/)

Panama/FFM backend for Jota JVM execution.

`jota-backend-panama` is not compatible with GraalVM Native Image.
It generates Java kernel sources and relies on compiling and loading those classes at runtime.

## Compile-time requirements

- Java 25+
- Maven (`mvnd` recommended)

## Compile dependencies

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jota</artifactId>
  <version>${qxotic.version}</version>
</dependency>
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>jota-backend-panama</artifactId>
  <version>${qxotic.version}</version>
</dependency>
```

## Runtime dependencies

- Java 25+ (JVM runtime)

## Configuration properties

- `jota.panama.compile.flags`: extra Java compiler flags for generated kernels
- `jota.verifyScratch`: enables scratch-buffer verification in LIR execution (`true`/`false`)

## Environment variables

- No backend-specific environment variables are required.
