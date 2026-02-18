# GGUF

A Java library for reading and writing [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files, the binary format used by [llama.cpp](https://github.com/ggml-org/llama.cpp) for storing machine learning model weights and metadata.

## What is GGUF?

GGUF (GGML Universal Format) is a binary file format for storing large language models. It contains:

- **Metadata**: Key-value pairs describing the model (name, architecture, context length, tokenizer info, etc.)
- **Tensor info**: Names, shapes, and data types for each weight tensor
- **Tensor data**: The actual weight values, often compressed using quantization

This library lets you read and write GGUF files from Java. It handles the format parsing and provides type-safe access to metadata and tensor information. It does **not** perform inference or read tensor data for you.

## Installation

=== "Maven"

    ```xml
    <dependency>
        <groupId>ai.qxotic</groupId>
        <artifactId>gguf</artifactId>
        <version>0.1-SNAPSHOT</version>
    </dependency>
    ```

=== "Gradle"

    ```groovy
    implementation 'ai.qxotic:gguf:0.1-SNAPSHOT'
    ```

=== "SBT"

    ```scala
    libraryDependencies += "ai.qxotic" % "gguf" % "0.1-SNAPSHOT"
    ```

=== "Mill"

    ```scala
    ivy"ai.qxotic::gguf:0.1-SNAPSHOT"
    ```

## Reading GGUF Files

### From a Local File

The simplest way to read a GGUF file is from a `Path`:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:read-path"
```

### From a Byte Channel

For streaming scenarios or custom I/O, use a `ReadableByteChannel`:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:read-channel"
```

### From HuggingFace

You can download models directly from HuggingFace. Here's a reusable helper:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:read-from-huggingface"
```

Usage:

```java
GGUF gguf = readFromHuggingFace("TheBloke", "Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf");
```

## Accessing Metadata

GGUF files contain rich metadata describing the model. Keys follow a dotted naming convention like `general.name` or `llama.context_length`.

### Getting Values

Use `getValue()` with the expected type. The library handles type casting for you:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:metadata-access"
```

### Arrays

Metadata can contain arrays of primitive values or strings:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:metadata-arrays"
```

For example, `tokenizer.ggml.tokens` contains all token strings in the model's vocabulary.

### Checking for Keys

Check if a key exists before accessing it:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:metadata-check"
```

### Listing All Keys

Enumerate all metadata keys and their types:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:metadata-keys"
```

### Common Metadata Keys

| Key | Type | Description |
|-----|------|-------------|
| `general.name` | String | Model name |
| `general.architecture` | String | Architecture (e.g., "llama") |
| `general.parameter_count` | Uint64 | Total parameters |
| `llama.context_length` | Uint32 | Maximum context window |
| `llama.embedding_length` | Uint32 | Hidden dimension size |
| `llama.block_count` | Uint32 | Number of transformer layers |
| `tokenizer.ggml.tokens` | String[] | Token vocabulary |

## Working with Tensors

In machine learning, a **tensor** is a multi-dimensional array of numbers. Model weights are stored as tensors - for example, a 4096×4096 matrix of 16-bit floats.

### Tensor Information

Each tensor entry provides metadata without loading the actual data:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:tensor-access"
```

### Getting a Specific Tensor

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:tensor-info"
```

The `offset()` is relative to `gguf.getTensorDataOffset()`. The actual file position is:

```java
long position = gguf.getTensorDataOffset() + tensor.offset();
```

### Reading Tensor Data

This library provides tensor offsets and sizes, but does **not** read tensor data. You read the bytes yourself, typically using memory-mapped files for efficiency:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:read-tensor-mmap"
```

The buffer contains raw tensor bytes. For quantized types (Q4_0, Q8_0, etc.), you'll need to dequantize based on the `GGMLType`.

### Quantization Types

GGUF supports various quantization formats to reduce model size:

| Type | Bits/Weight | Block Size | Description |
|------|-------------|------------|-------------|
| F32 | 32 | 1 | Full 32-bit float |
| F16 | 16 | 1 | Half precision |
| Q4_0 | 4.5 | 32 | 4-bit quantization |
| Q5_0 | 5.5 | 32 | 5-bit quantization |
| Q8_0 | 8.5 | 32 | 8-bit quantization |
| Q4_K | 4.5 | 256 | K-quantized 4-bit |
| Q6_K | 6.56 | 256 | K-quantized 6-bit |

You can query type properties programmatically:

```java
GGMLType type = GGMLType.Q8_0;
type.getBlockByteSize();    // 34 bytes per block
type.getElementsPerBlock(); // 32 elements per block
type.getBitsPerWeight();    // 8.5 bits per weight
type.isQuantized();         // true
type.byteSizeForShape(new long[]{1024, 1024}); // total bytes for shape
```

## Writing GGUF Files

### The Builder Pattern

Use `Builder` to create or modify GGUF files. The builder handles all format details including header structure and alignment.

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:builder-create"
```

### Automatic Tensor Offsets

When building GGUF files, you typically don't need to calculate tensor offsets manually. The builder automatically computes them when you call `build()`:

- Tensors are packed sequentially in the order you add them
- Each tensor is aligned according to the alignment setting (default: 32 bytes)
- Offsets are relative to the tensor data section

You can pass `offset=0` when creating `TensorEntry` since it will be overwritten:

```java
builder.putTensor(TensorEntry.create("weight", new long[]{1024, 4096}, GGMLType.F16, 0));
```

If you need to preserve existing offsets (e.g., when only modifying metadata), use `build(false)`:

```java
// Preserves original tensor offsets
GGUF modified = builder.build(false);
```

### Modifying Existing Files

Create a builder from an existing GGUF to modify it:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:builder-modify"
```

When modifying metadata only, use `build(false)` to keep tensor offsets unchanged - this allows in-place updates if the metadata section doesn't grow.

### Alignment

Tensor data is aligned to improve memory access performance. The default is 32 bytes:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:builder-alignment"
```

### Writing to a File

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:write-file"
```

### Writing to a Channel

For custom output destinations:

```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:write-channel"
```

## Quick Inspect with JBang

For ad-hoc inspection, use the included JBang script:

```bash
jbang scripts/gguf.java hf unsloth/Qwen3-0.6B-GGUF/Q8_0 --no-tensors
```

## Thread Safety

`GGUF` instances are immutable and safe to share across threads. `Builder` is mutable and not thread-safe.

