# GGUF

A Java library for reading and writing [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files, the binary format used by [llama.cpp](https://github.com/ggml-org/llama.cpp) for storing machine learning model weights and metadata.

## What is GGUF?

GGUF (GGML Universal Format) is a binary file format for storing large language models. It contains:

- **Metadata**: Key-value pairs describing the model (name, architecture, context length, tokenizer info, etc.)
- **Tensor information**: Names, shapes, and data types for each weight tensor
- **Tensor data**: The actual weight values, often compressed using quantization

This library lets you read and write GGUF files from Java. It handles the format parsing and provides type-safe access to metadata and tensor information.

!!! info "Library Scope"
    This library provides **metadata and tensor layout information only**. It does **not**:
    - Read or write tensor data (you do this yourself)
    - Perform inference or model execution
    - Handle quantization/dequantization
    - Provide tensor operations

## Quick Start

A complete example reading a GGUF file and accessing its contents:

```java
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.nio.file.Path;

// Read the GGUF file
GGUF gguf = GGUF.read(Path.of("model.gguf"));

// Access metadata
String name = gguf.getValue(String.class, "general.name");
int contextLength = gguf.getValueOrDefault(int.class, "llama.context_length", 4096);

// List tensors
for (TensorEntry tensor : gguf.getTensors()) {
    System.out.println(tensor.name() + ": " + tensor.ggmlType());
}

// Get specific tensor info
TensorEntry weights = gguf.getTensor("token_embd.weight");
long filePosition = weights.absoluteOffset(gguf);
long byteSize = weights.byteSize();
```

### Inspect a Model from URL

Simple program to inspect any GGUF file hosted online:

```java
--8<-- "UtilitySnippets.java:inspector-complete"
```

Usage:
```bash
java GGUFInspector.java https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

## Installation

=== "Maven"

    ```xml
    <dependency>
        <groupId>com.qxotic</groupId>
        <artifactId>gguf</artifactId>
        <version>0.1-SNAPSHOT</version>
    </dependency>
    ```

=== "Gradle"

    ```groovy
    implementation 'com.qxotic:gguf:0.1-SNAPSHOT'
    ```

=== "Mill"

    ```scala
    ivy"com.qxotic::gguf:0.1-SNAPSHOT"
    ```

## Reading GGUF Files

### From a Local File

```java
--8<-- "ReadingSnippets.java:read-path"
```

### From a ByteChannel

```java
--8<-- "ReadingSnippets.java:read-channel"
```

### From HuggingFace

```java
--8<-- "UtilitySnippets.java:read-from-huggingface"
```

Usage:

```java
GGUF gguf = readFromHuggingFace("TheBloke", "Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf");
```

## Accessing Metadata

### Getting Values

```java
--8<-- "MetadataSnippets.java:metadata-access"
```

### Unsigned Types

GGUF stores unsigned integer types (UINT8, UINT16, UINT32, UINT64), but Java doesn't have unsigned primitive types. When reading unsigned values, use the corresponding signed type and conversion methods if you need to work with values larger than the signed maximum.

**Type Mapping:**

| GGUF Type | Java Type | Notes |
|-----------|-----------|-------|
| UINT8 | `byte` | Values 0-255 stored as -128 to 127 |
| UINT16 | `short` | Use `Short.toUnsignedInt()` if needed |
| UINT32 | `int` | Use `Integer.toUnsignedLong()` if needed |
| UINT64 | `long` | Use `Long.toUnsignedString()` for display |

Example reading an unsigned 64-bit parameter count:

```java
long paramCount = gguf.getValue(long.class, "general.parameter_count");
String display = Long.toUnsignedString(paramCount);
```

### Default Values

Use `getValueOrDefault()` when a key might not exist and you want to provide a fallback:

```java
--8<-- "MetadataSnippets.java:metadata-or-default"
```

### Checking for Keys

```java
--8<-- "MetadataSnippets.java:metadata-check"
```

### Listing All Keys

```java
--8<-- "MetadataSnippets.java:metadata-keys"
```

## Working with Tensors

### Tensor Information

```java
--8<-- "TensorSnippets.java:tensor-access"
```

### Getting a Specific Tensor

```java
--8<-- "TensorSnippets.java:tensor-info"
```

The `offset()` is relative to `gguf.getTensorDataOffset()`. The actual file position is:

```java
--8<-- "TensorSnippets.java:tensor-offset"
```

### Reading Tensor Data

!!! info "Reading Tensor Bytes"
    This library provides tensor offsets and sizes, but does **not** read tensor data. You read the bytes yourself using the approaches below.

#### Read tensor data into a ByteBuffer

For smaller tensors, allocate a heap-based ByteBuffer:

```java
--8<-- "TensorDataSnippets.java:read-tensor-bytebuffer"
```

#### Using a Memory-Mapped ByteBuffer

For large models, memory mapping is more efficient as the OS loads data on-demand:

```java
--8<-- "TensorDataSnippets.java:read-tensor-mmap"
```

!!! tip "Memory Mapping on Windows"
    On Windows, memory-mapped files keep the file locked until the buffer is garbage collected. 
    Use `buffer.clear()` and `System.gc()` if you need to modify or delete the file after reading.

#### Reading Multiple Tensors

When reading all tensors, reuse the channel:

```java
--8<-- "TensorDataSnippets.java:read-all-tensors"
```

## Quantization Types

| Type | Bits/Weight | Block Size | Description |
|------|-------------|------------|-------------|
| F32 | 32 | 1 | Full 32-bit float |
| F16 | 16 | 1 | Half precision |
| Q4_0 | 4.5 | 32 | 4-bit quantization |
| Q5_0 | 5.5 | 32 | 5-bit quantization |
| Q8_0 | 8.5 | 32 | 8-bit quantization |
| Q4_K | 4.5 | 256 | K-quantized 4-bit |
| Q6_K | 6.56 | 256 | K-quantized 6-bit |

## Writing GGUF Files

### Basic Writing

Write a complete GGUF file (metadata only):

```java
--8<-- "WritingSnippets.java:write-file"
```

### Creating New Files

```java
--8<-- "BuildingSnippets.java:builder-create"
```

### Modifying Existing Files

```java
--8<-- "BuildingSnippets.java:builder-modify"
```

### Alignment

Alignment determines the byte boundary where tensor data must start in the GGUF file. This is important for performance:

- **Memory access**: Aligned data can be read more efficiently by the CPU
- **SIMD operations**: Vector instructions often require aligned memory addresses
- **GPU transfers**: Aligned data transfers faster to GPU memory

The alignment value must be a power of 2 (e.g., 32, 64, 128). The default is 32 bytes. When writing, padding bytes are automatically added between tensors to ensure each tensor starts at an aligned address:

```
[Tensor 1 Data][padding][Tensor 2 Data][padding][Tensor 3 Data]
               ^^^^^^^^               ^^^^^^^^
               aligned to 32 bytes    aligned to 32 bytes
```

```java
--8<-- "BuildingSnippets.java:builder-alignment"
```

### Tensor Offsets

When building a GGUF file, tensor offsets specify where each tensor's data begins in the file. These offsets are relative to the start of the tensor data section (after all metadata).

**When modifying existing files**, you typically want to re-compute tensor offsets automatically:

- Adding or removing metadata changes the tensor data section's starting position
- Changing alignment requires re-calculating padding between tensors
- Adding or removing tensors changes the layout

By default, `build()` automatically re-computes offsets. Use `build(false)` only if you need to preserve exact offsets from the original file (e.g., when making metadata-only changes where tensor data positions must not change).

```java
// Default: automatically re-compute offsets
GGUF gguf = builder.build();

// Or explicitly
GGUF gguf = builder.build(true);

// Preserve original offsets (use with caution)
GGUF gguf = builder.build(false);
```

### Writing Tensor Data

This library writes GGUF metadata only. Tensor data must be written separately after the metadata.

!!! note "Writing Metadata Only"
    `GGUF.write()` writes only the metadata section (up to `tensorDataOffset`). It does not truncate the file. If the output file already exists and contains tensor data, that data will remain in the file beyond the metadata. To create a clean file with metadata only, truncate the channel after writing.

#### Writing from ByteBuffer

The simplest approach - write metadata, seek to the tensor's position, then write the data:

```java
--8<-- "WritingSnippets.java:write-tensor-buffer"
```

#### Writing Multiple Tensors

When writing all tensors, loop through and write each at its offset:

```java
try (FileChannel channel = FileChannel.open(Path.of("output.gguf"),
        StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
    GGUF.write(gguf, channel);

    for (TensorEntry tensor : gguf.getTensors()) {
        ByteBuffer data = getTensorData(tensor.name()); // your data source
        channel.position(tensor.absoluteOffset(gguf));
        channel.write(data);
    }
}
```

## Error Handling

The library throws `GGUFFormatException` for invalid GGUF files:

- **Corrupted files**: Invalid magic number, version mismatch, truncated data
- **Format violations**: Tensor names > 64 characters, more than 4 dimensions, unaligned offsets
- **Type mismatches**: Wrong metadata value types, missing required keys
- **Duplicate entries**: Duplicate tensor names or metadata keys

```java
try {
    GGUF gguf = GGUF.read(Path.of("model.gguf"));
} catch (GGUFFormatException e) {
    System.err.println("Invalid GGUF file: " + e.getMessage());
}
```

## Validation

### Quick Validation

```java
--8<-- "ValidationSnippets.java:quick-validate"
```

### Comprehensive Validation

```java
--8<-- "ValidationSnippets.java:comprehensive-validate"
```

## Thread Safety

!!! info "Concurrency"
    - `GGUF` instances are **immutable** and safe to share across threads
    - `Builder` is **mutable** and not thread-safe - create a new Builder per thread
    - `TensorEntry` objects are immutable value objects
