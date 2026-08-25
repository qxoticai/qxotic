# Safetensors

[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

**Safetensors for the JVM.** Read and write HuggingFace's model format in pure Java — zero
dependencies, Java 11+, GraalVM native-image ready.

Strict schema validation, single-file and sharded models, and a JBang CLI that inspects headers
straight off the hub — no multi-gigabyte download.

## What you get

- **Read headers** — `__metadata__` plus every tensor entry (dtype, shape, offsets)
- **Write headers** — build or modify with the builder API
- **Strict validation** — dtype, shape, offset and overlap checks; malformed files fail loudly
- **Sharded models** — `SafetensorsIndex` resolves which shard holds each tensor, from
  `model.safetensors` or `model.safetensors.index.json`

Deliberately out of scope: tensor payload I/O and dtype conversion. Small API, predictable
behavior — by design.

## Quick example

```java
Safetensors st = Safetensors.read(Path.of("model.safetensors"));
TensorEntry tensor = st.requireTensor("model.embed_tokens.weight");
System.out.println(tensor.byteSize());

Safetensors modified = Builder.newBuilder(st)
    .putMetadataKey("format", "pt")
    .setAlignment(64)
    .build();
Safetensors.write(modified, Path.of("output.safetensors"));
```

## Sharded models

```java
SafetensorsIndex index = SafetensorsIndex.load(Path.of("/path/to/model-dir"));
Path shard = index.requireSafetensorsPath("model.layers.0.self_attn.q_proj.weight");
```

## Peek from the CLI

```bash
jbang scripts/safetensors.java hf HuggingFaceTB/SmolLM2-135M --no-tensors
jbang scripts/safetensors.java modelscope Qwen/Qwen3-4B --no-tensors
```

Pure JSON on stdout. Local files and arbitrary URLs work too.

## Installation

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>safetensors</artifactId>
    <version>0.2.0</version>
</dependency>
```

## Documentation

Full docs and examples: [qxotic.ai/docs/safetensors](https://qxotic.ai/docs/safetensors).

## License

Apache 2.0
