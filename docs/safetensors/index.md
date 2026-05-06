---
sidebar_position: 1
---

# Safetensors

A Java library for reading and writing [Safetensors](https://github.com/huggingface/safetensors) files, the format used by [Hugging Face](https://huggingface.co) for storing model weights and metadata. See the [Safetensors specification](https://huggingface.co/docs/safetensors/index).

**Java 11+ · GraalVM Native Image ready**

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
    <version>0.1.0</version>
</dependency>
```

  </TabItem>
  <TabItem value="gradle" label="Gradle">

```groovy
implementation 'com.qxotic:safetensors:0.1.0'
```

  </TabItem>
  <TabItem value="mill" label="Mill">

```scala
mvn"com.qxotic:safetensors:0.1.0"
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

| Type | Size | Java Type |
|------|------|-----------|
| F64 | 8 bytes | `double` |
| F32 | 4 bytes | `float` |
| F16 | 2 bytes | `short` |
| BF16 | 2 bytes | `short` |
| I64 | 8 bytes | `long` |
| I32 | 4 bytes | `int` |
| I16 | 2 bytes | `short` |
| I8 | 1 byte | `byte` |
| U64 | 8 bytes | `long` |
| U32 | 4 bytes | `int` |
| U16 | 2 bytes | `short` |
| U8 | 1 byte | `byte` |
| BOOL | 1 byte | `boolean` |

Java lacks unsigned primitives. Use `Byte.toUnsignedInt()`, `Short.toUnsignedInt()`, `Integer.toUnsignedLong()`, or `Long.toUnsignedString()` when needed.

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

Tensor data starts at aligned byte boundaries (default: 64 bytes, must be a power of 2). Padding is added automatically.

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
