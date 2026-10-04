---
sidebar_position: 1
---

# Safetensors

**Safetensors for the JVM.** Read and write [Hugging Face's model format](https://huggingface.co/docs/safetensors/index) in pure Java. One dependency (`com.qxotic:json`, itself dependency-free), Java 11+, GraalVM native-image ready. Strict schema validation, single-file and sharded models.

## Quick Start

```java
import com.qxotic.format.safetensors.*;
import java.nio.file.Path;

Safetensors st = Safetensors.read(Path.of("model.safetensors"));

// Metadata
Map<String, String> metadata = st.getMetadata();

// Tensors
for (TensorEntry tensor : st.getTensors()) {
    System.out.println(tensor.name() + ": " + tensor.dtype() + " " + Arrays.toString(tensor.shape()));
}

// Specific tensor
TensorEntry weights = st.getTensor("model.embed_tokens.weight");
long filePosition = st.absoluteOffset(weights);
long byteSize = weights.byteSize();
```

## Installation

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
  <TabItem value="maven" label="Maven">

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>safetensors</artifactId>
    <version>0.2.0</version>
</dependency>
```

  </TabItem>
  <TabItem value="gradle" label="Gradle">

```groovy
implementation 'com.qxotic:safetensors:0.2.0'
```

  </TabItem>
  <TabItem value="mill" label="Mill">

```scala
mvn"com.qxotic:safetensors:0.2.0"
```

  </TabItem>
</Tabs>

## Reading

### From a File

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="read-path"
```

### From a ByteChannel

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="read-channel"
```

### From HuggingFace

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="read-from-huggingface"
```

```java
Safetensors st = readFromHuggingFace("HuggingFaceTB", "SmolLM2-135M", "model.safetensors");
```

## Metadata

Metadata values are always strings per the Safetensors spec:

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="metadata"
```

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="metadata-keys"
```

## Tensors

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="tensors"
```

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="tensor-one"
```

### Reading Tensor Data

The library provides tensor offsets and sizes. You read/write tensor bytes yourself using `FileChannel`:

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="read-tensor-bytebuffer"
```

For large models, memory-map instead of copying:

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="read-tensor-mmap-buffer"
```

## Data Types

Every `DType`, in the specification's alignment order, with its size and the Java primitive that carries it:

| Type | Bits | Java carrier |
|------|------|--------------|
| BOOL | 8 | `boolean` |
| F4 | 4 | `byte` |
| F6_E2M3 | 6 | `byte` |
| F6_E3M2 | 6 | `byte` |
| U8 | 8 | `byte` |
| I8 | 8 | `byte` |
| F8_E5M2 | 8 | `byte` |
| F8_E4M3 | 8 | `byte` |
| F8_E8M0 | 8 | `byte` |
| F8_E4M3FNUZ | 8 | `byte` |
| F8_E5M2FNUZ | 8 | `byte` |
| I16 | 16 | `short` |
| U16 | 16 | `short` |
| F16 | 16 | `short` |
| BF16 | 16 | `short` |
| I32 | 32 | `int` |
| U32 | 32 | `int` |
| F32 | 32 | `float` |
| C64 | 64 | `float` |
| F64 | 64 | `double` |
| I64 | 64 | `long` |
| U64 | 64 | `long` |

Sub-byte and FP8 types are carried as `byte`; `C64` is a pair of `float`s.
Java lacks unsigned primitives.
Use `Byte.toUnsignedInt()`, `Short.toUnsignedInt()`, `Integer.toUnsignedLong()`, or `Long.toUnsignedString()` when needed.

## Writing

### Builder

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="builder-create"
```

### Modifying Existing Files

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="builder-modify"
```

### Writing a Complete File

`Safetensors.write()` writes the header only. Write tensor data separately at each tensor's offset:

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="write-tensor-buffer"
```

#### Multiple Tensors

```java
try (FileChannel channel = FileChannel.open(Path.of("output.safetensors"),
        StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
    Safetensors.write(st, channel);

    for (TensorEntry tensor : st.getTensors()) {
        ByteBuffer data = getTensorData(tensor.name()); // your data source
        channel.position(st.absoluteOffset(tensor));
        channel.write(data);
    }
}
```

### Alignment

Tensor data starts at aligned byte boundaries (default: 1, the upstream layout; set a power of 2 for aligned tensor data). Padding is added automatically.

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="builder-alignment"
```

### Tensor Offsets

`build()` recomputes tensor offsets by default. Use `build(false)` to preserve original offsets.

## Sharded Models

Use `SafetensorsIndex` to locate tensors across shards:

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="index-load"
```

Handles both `model.safetensors` (single-file) and `model.safetensors.index.json` (sharded).

## Error Handling

```snippet path="safetensors/src/test/java/com/qxotic/format/safetensors/Snippets.java" tag="error-handling"
```

## Command Line

```bash
jbang scripts/safetensors.java hf HuggingFaceTB/SmolLM2-135M --no-tensors
jbang scripts/safetensors.java modelscope Qwen/Qwen3-4B --no-tensors
```
