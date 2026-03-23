# Safetensors

[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

A pure Java library for reading and writing HuggingFace's [Safetensors](https://github.com/huggingface/safetensors) format.

**Zero dependencies · Java 11+ · GraalVM Native Image ready**

---

## What this library does

- Reads Safetensors headers (`__metadata__` + tensor entries)
- Writes Safetensors headers from `Safetensors` instances
- Validates strict schema constraints (dtype, shape, offsets, overlap)
- Supports single-file and sharded model indexing via `SafetensorsIndex`

## What this library does not do

- Does **not** read tensor payload bytes for you
- Does **not** perform dtype conversion or inference

This keeps the API small and predictable.

---

## Installation

### Maven

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>safetensors</artifactId>
    <version>0.1.0</version>
</dependency>
```

---

## Quick Example

```java
Safetensors st = Safetensors.read(Path.of("model.safetensors"));

System.out.println(st.getAlignment());
System.out.println(st.getTensorDataOffset());
System.out.println(st.getMetadata());
System.out.println(st.getTensor("model.embed_tokens.weight"));
```

---

## Build or Modify Headers

```java
Safetensors modified = Builder.newBuilder(st)
    .putMetadataKey("format", "pt")
    .setAlignment(64)
    .build();

Safetensors.write(modified, Path.of("output.safetensors"));
```

---

## Sharded Models

Use `SafetensorsIndex` to resolve which shard contains each tensor:

```java
SafetensorsIndex index = SafetensorsIndex.load(Path.of("/path/to/model-dir"));
Path shard = index.getSafetensorsPath("model.layers.0.self_attn.q_proj.weight");
```

The index loader handles both:
- `model.safetensors` (single-file)
- `model.safetensors.index.json` (sharded)

---

## JBang CLI Script 

Inspect headers directly from local files, URLs, HuggingFace or ModelScope:

```bash
jbang scripts/safetensors.java hf HuggingFaceTB/SmolLM2-135M --no-tensors
jbang scripts/safetensors.java modelscope Qwen/Qwen3-4B --no-tensors
```

Output is pure JSON on stdout.

---

## Documentation

Full docs and examples are in `docs/index.md`.
