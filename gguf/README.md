# GGUF

[![Maven Central](https://img.shields.io/maven-central/v/com.qxotic/gguf)](https://search.maven.org/artifact/com.qxotic/gguf)
[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

**GGUF for the JVM.** Read and write llama.cpp's model format in pure Java — zero dependencies,
Java 11+, GraalVM native-image ready.

Every LLM worth running locally ships as GGUF. This library opens them: metadata, tensor layouts,
tokenizer config, quantization types — type-safe, bounds-checked, embeddable anywhere a JVM runs.

## Why this library

- **Zero dependencies.** `java.nio` and collections only. A few classes, no transitive tree.
- **Read and write.** Inspect any GGUF, then modify metadata or build new files with the builder API.
- **Type-safe.** `getValue(String.class, "general.name")` — no casting gymnastics.
- **Every GGML type.** Q4_0, Q4_K, Q6_K, Q8_0, MXFP4, F16, F32 — the whole quantization zoo.
- **Pure Java.** HotSpot, OpenJ9, GraalVM — native-image compiles with zero configuration.

## Quick example

```java
// Reading
GGUF gguf = GGUF.read(Path.of("model.gguf"));
String name = gguf.getValue(String.class, "general.name");

// Writing
GGUF modified = Builder.newBuilder(gguf)
    .putString("general.description", "My model")
    .build();
GGUF.write(modified, Path.of("output.gguf"));
```

## Peek without downloading

The [JBang script](scripts/gguf.java) reads GGUF metadata straight off HuggingFace (or any URL) —
no multi-gigabyte download:

```bash
jbang scripts/gguf.java hf unsloth/Qwen3-0.6B-GGUF/Q8_0 --no-tensors
```

## Installation

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>gguf</artifactId>
    <version>0.2.0</version>
</dependency>
```

## Deliberately out of scope

- **No tensor payload I/O** — you read/write raw bytes at the offsets the library provides
- **No quantization math** — raw bytes only
- **No inference** — that's [jinfer](../jinfer)

Small, focused, dependency-free — by design.

## Documentation

[Complete API documentation](https://qxotic.ai/docs/gguf): reading from files, channels and URLs; type-safe
metadata access; creating and modifying files; tensor offsets; every GGML data type.

## Development

```bash
mvn test
mvn spotless:apply
```

## License

Apache 2.0
