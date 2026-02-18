# GGUF

[![Maven Central](https://img.shields.io/maven-central/v/ai.qxotic/gguf)](https://search.maven.org/artifact/ai.qxotic/gguf)
[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

A pure Java library for reading and writing [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files - the binary format used by [llama.cpp](https://github.com/ggml-org/llama.cpp) for storing machine learning model weights and metadata.

**Zero dependencies · Java 11+ · GraalVM Native Image ready**

---

## Why This Library?

- **Zero Runtime Dependencies** - Only uses standard Java library (`java.nio`, collections). No external libraries required.
- **Pure Java Implementation** - No native code, works on any JVM platform (HotSpot, OpenJ9, GraalVM).
- **Java 11+ Compatible** - Minimum Java 11, tested on LTS versions (11, 17, 21).
- **GraalVM Native Image Ready** - Compile to native binaries with zero configuration.
- **Read & Write GGUF Files** - Full support for reading metadata/tensor info and creating/modifying files via builder API.
- **Type-Safe Metadata Access** - Automatic casting with generic `getValue(Class<T>, key)`.
- **Lightweight** - Small footprint, embeddable in CLI tools, servers, or desktop applications.

---

## Installation

### Maven

```xml
<dependency>
    <groupId>ai.qxotic</groupId>
    <artifactId>gguf</artifactId>
    <version>0.1-SNAPSHOT</version>
</dependency>
```
---

## Quick Example

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
---

## Peek GGUF metadata with [JBang](https://www.jbang.dev/)

Peek GGUF metadata from [HuggingFace](https://huggingface.co) (or any URL) without downloading the full file:

```bash
jbang scripts/gguf.java hf unsloth/Qwen3-0.6B-GGUF/Q8_0 --no-tensors
```

---

## Documentation

See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for complete API documentation with detailed examples:

- Reading from files, channels, and URLs
- Accessing metadata with type-safe getters
- Creating and modifying GGUF files
- Tensor information and offset calculations
- All GGML data types (Q4_0, Q8_0, F16, F32, etc.)

---

## What This Library Does NOT Do

- **No tensor data I/O** - You read/write raw tensor bytes at offsets the library provides
- **No quantization/dequantization** - Raw bytes only, no math operations
- **No inference** - Structure and metadata only, not a runtime

This keeps the library focused, lightweight, and dependency-free.

---

## Development

```bash
mvnd compile
mvnd test
mvnd spotless:apply
```

---

## License

Apache License 2.0
