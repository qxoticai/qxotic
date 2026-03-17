# Safetensors

A Java library for reading and writing [Safetensors](https://github.com/huggingface/safetensors) files - the format used by Hugging Face for storing machine learning model weights and metadata.

## What is Safetensors?

Safetensors is a binary file format for storing tensor data safely and efficiently. It was created by Hugging Face as a secure alternative to Python's pickle format. The format contains:

- **Header**: JSON metadata describing the file structure
- **Tensor information**: Names, shapes, data types, and byte offsets  
- **Tensor data**: The actual weight values in contiguous binary form

This library lets you read and write Safetensors files from Java. It handles the format parsing and provides type-safe access to metadata and tensor information.

!!! info "Library Scope"
    This library provides **metadata and tensor layout information only**. It does **not**:
    - Read or write tensor data (you do this yourself)
    - Perform inference or model execution
    - Handle dtype conversion
    - Provide tensor operations

## Quick Start

A complete example reading a Safetensors file and accessing its contents:

```java
import com.qxotic.format.safetensors.*;
import java.nio.file.Path;

// Read the safetensors file
Safetensors st = Safetensors.read(Path.of("model.safetensors"));

// Access metadata
Map<String, String> metadata = st.getMetadata();
String format = metadata.get("format");

// List tensors
for (TensorEntry tensor : st.getTensors()) {
    System.out.println(tensor.name() + ": " + tensor.dtype() + " " + Arrays.toString(tensor.shape()));
}

// Get specific tensor info
TensorEntry weights = st.getTensor("model.embed_tokens.weight");
long filePosition = weights.absoluteOffset(st);
long byteSize = weights.byteSize();
```

### Inspect a Model from URL

Simple program to inspect any Safetensors file hosted online:

```java
--8<-- "scripts/PeekSafetensors.java"
```

Usage:
```bash
java PeekSafetensors.java https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/model.safetensors
```

## Installation

=== "Maven"

    ```xml
    <dependency>
        <groupId>com.qxotic</groupId>
        <artifactId>safetensors</artifactId>
        <version>0.1.0</version>
    </dependency>
    ```

=== "Gradle"

    ```groovy
    implementation 'com.qxotic:safetensors:0.1.0'
    ```

=== "Mill"

    ```scala
    ivy"com.qxotic::safetensors:0.1.0"
    ```

## Reading Safetensors Files

### From a Local File

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:read-path"
```

### From a ByteChannel

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:read-channel"
```

### From HuggingFace

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:read-from-huggingface"
```

Usage:

```java
Safetensors st = readFromHuggingFace("HuggingFaceTB", "SmolLM2-135M", "model.safetensors");
```

## Accessing Metadata

### Getting Metadata

All metadata values are strings per the Safetensors specification:

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:metadata"
```

### Checking for Keys

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:basic-info"
```

### Listing All Keys

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:metadata-keys"
```

## Working with Tensors

### Tensor Information

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:tensors"
```

### Getting a Specific Tensor

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:tensor-one"
```

The `byteOffset()` is relative to `st.getTensorDataOffset()`. The actual file position is:

```java
long absoluteOffset = tensor.absoluteOffset(st);  // Convenience method
// Or manually: st.getTensorDataOffset() + tensor.byteOffset()
```

### Reading Tensor Data

!!! info "Reading Tensor Bytes"
    This library provides tensor offsets and sizes, but does **not** read tensor data. You read the bytes yourself using the approaches below.

#### Read tensor data into a ByteBuffer

For smaller tensors, allocate a heap-based ByteBuffer:

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:read-tensor-bytebuffer"
```

#### Using a Memory-Mapped ByteBuffer

For large models, memory mapping is more efficient as the OS loads data on-demand:

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:read-tensor-mmap-buffer"
```

!!! tip "Memory Mapping on Windows"
    On Windows, memory-mapped files keep the file locked until the buffer is garbage collected. 
    Use `buffer.clear()` and `System.gc()` if you need to modify or delete the file after reading.

#### Using MemorySegment (Java 21+)

If you're using Java 21+, you can use Panama's MemorySegment for zero-copy access:

```java
TensorEntry tensor = st.getTensor("model.embed_tokens.weight");
long absoluteOffset = tensor.absoluteOffset(st);
long byteSize = tensor.byteSize();

// Open file and create memory segment
try (var channel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
    MemorySegment segment = channel.map(
            MapMode.READ_ONLY, absoluteOffset, byteSize, Arena.ofAuto());
    // segment provides zero-copy access to tensor data
    // Use MemorySegment methods or convert to ByteBuffer via segment.asByteBuffer()
}
```

## Data Types

| Type | Size | Java Type | Description |
|------|------|-----------|-------------|
| F64 | 8 bytes | `double` | 64-bit float |
| F32 | 4 bytes | `float` | 32-bit float |
| F16 | 2 bytes | `short` | 16-bit float |
| BF16 | 2 bytes | `short` | BFloat16 |
| I64 | 8 bytes | `long` | 64-bit signed int |
| I32 | 4 bytes | `int` | 32-bit signed int |
| I16 | 2 bytes | `short` | 16-bit signed int |
| I8 | 1 byte | `byte` | 8-bit signed int |
| U64 | 8 bytes | `long` | 64-bit unsigned int |
| U32 | 4 bytes | `int` | 32-bit unsigned int |
| U16 | 2 bytes | `short` | 16-bit unsigned int |
| U8 | 1 byte | `byte` | 8-bit unsigned int |
| BOOL | 1 byte | `boolean` | Boolean |

!!! note "Unsigned Type Handling"
    Java doesn't have unsigned primitive types. When reading unsigned values, use the corresponding signed type and conversion methods:
    
    - **U8** → `byte` - Use `Byte.toUnsignedInt()` if needed
    - **U16** → `short` - Use `Short.toUnsignedInt()` if needed
    - **U32** → `int` - Use `Integer.toUnsignedLong()` if needed
    - **U64** → `long` - Use `Long.toUnsignedString()` for display

## Writing Safetensors Files

### Basic Writing

Write a Safetensors header to a file:

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:write-file"
```

### Creating New Files

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:builder-create"
```

### Modifying Existing Files

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:builder-modify"
```

### Alignment

Alignment determines the byte boundary where tensor data must start in the Safetensors file. This is important for performance:

- **Memory access**: Aligned data can be read more efficiently by the CPU
- **SIMD operations**: Vector instructions often require aligned memory addresses
- **GPU transfers**: Aligned data transfers faster to GPU memory

The alignment value must be a power of 2 (e.g., 32, 64). The default is 64 bytes. When writing, padding bytes are automatically added between tensors to ensure each tensor starts at an aligned address:

```
[Tensor 1 Data][padding][Tensor 2 Data][padding][Tensor 3 Data]
               ^^^^^^^^               ^^^^^^^^
               aligned to 64 bytes    aligned to 64 bytes
```

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:builder-alignment"
```

### Tensor Offsets

When building a Safetensors file, tensor offsets specify where each tensor's data begins in the file. These offsets are relative to the start of the tensor data section.

**When modifying existing files**, you typically want to re-compute tensor offsets automatically:

- Adding or removing metadata changes the tensor data section's starting position
- Changing alignment requires re-calculating padding between tensors
- Adding or removing tensors changes the layout

By default, `build()` automatically re-computes offsets. Use `build(false)` only if you need to preserve exact offsets from the original file.

```java
// Default: automatically re-compute offsets
Safetensors st = builder.build();

// Or explicitly
Safetensors st = builder.build(true);

// Preserve original offsets (use with caution)
Safetensors st = builder.build(false);
```

### Writing Tensor Data

This library writes Safetensors headers only. Tensor data must be written separately after the header.

!!! note "Writing Headers Only"
    `Safetensors.write()` writes only the header section (up to `tensorDataOffset`). It does not truncate the file. If the output file already exists and contains tensor data, that data will remain in the file beyond the header. To create a clean file with metadata only, truncate the channel after writing.

#### Writing from ByteBuffer

The simplest approach - write header, seek to the tensor's position, then write the data:

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:write-tensor-buffer"
```

#### Writing from MemorySegment (Java 21+)

For zero-copy writes using Panama, memory-map the file region and copy from your source segment:

```java
Safetensors st = Safetensors.read(Path.of("model.safetensors"));
TensorEntry tensor = st.getTensor("model.embed_tokens.weight");

// Your source data as a MemorySegment
MemorySegment sourceSegment = loadTensorData(); // your data loading logic

try (FileChannel channel = FileChannel.open(Path.of("output.safetensors"),
        StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
    // Write header
    Safetensors.write(st, channel);

    // Extend file to fit tensor data
    long endPosition = tensor.absoluteOffset(st) + tensor.byteSize();
    channel.truncate(endPosition);

    // Map file region and copy data
    try (Arena arena = Arena.ofConfined()) {
        MemorySegment mapped = channel.map(
                FileChannel.MapMode.READ_WRITE,
                tensor.absoluteOffset(st),
                tensor.byteSize(),
                arena);
        MemorySegment.copy(sourceSegment, 0, mapped, 0, tensor.byteSize());
    }
}
```

#### Writing Multiple Tensors

When writing all tensors, loop through and write each at its offset:

```java
try (FileChannel channel = FileChannel.open(Path.of("output.safetensors"),
        StandardOpenOption.CREATE, StandardOpenOption.WRITE)) {
    Safetensors.write(st, channel);

    for (TensorEntry tensor : st.getTensors()) {
        ByteBuffer data = getTensorData(tensor.name()); // your data source
        channel.position(tensor.absoluteOffset(st));
        channel.write(data);
    }
}
```

## Sharded Models

Large models are often split into multiple shards. Use `SafetensorsIndex` to locate tensors across shards:

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:index-load"
```

The index loader handles both:

- `model.safetensors` (single-file)
- `model.safetensors.index.json` (sharded)

## Error Handling

The library throws `SafetensorsFormatException` for invalid Safetensors files:

- **Corrupted files**: Invalid header structure, malformed JSON
- **Format violations**: Missing required fields, overlapping tensor data
- **Type mismatches**: Invalid dtype values, non-string metadata

```java
--8<-- "src/test/java/com/qxotic/format/safetensors/Snippets.java:error-handling"
```

## Thread Safety

!!! info "Concurrency"
    - `Safetensors` instances are **immutable** and safe to share across threads
    - `Builder` is **mutable** and not thread-safe - create a new Builder per thread
    - `TensorEntry` objects are immutable value objects

## Command Line Utility

A [JBang](https://www.jbang.dev/) script for inspecting Safetensors files is available:

[`scripts/safetensors.java`](https://github.com/qxoticai/llm4j/blob/main/safetensors/scripts/safetensors.java)

Usage:
```bash
jbang scripts/safetensors.java hf HuggingFaceTB/SmolLM2-135M --no-tensors
jbang scripts/safetensors.java modelscope Qwen/Qwen3-4B --no-tensors
```

## API Reference

### Main Types

| Class | Description |
|-------|-------------|
| `Safetensors` | Read/write safetensors headers |
| `TensorEntry` | Tensor metadata (name, dtype, shape, offset) |
| `Builder` | Create or modify safetensors files |
| `SafetensorsIndex` | Locate tensors across sharded models |
| `DType` | Enum of supported data types |
| `SafetensorsFormatException` | Invalid format exception |

### TensorEntry

```java
TensorEntry tensor = st.getTensor("weight");
String name = tensor.name();              // tensor name
DType dtype = tensor.dtype();             // data type
long[] shape = tensor.shape();            // tensor shape
long offset = tensor.byteOffset();        // offset in data section
long size = tensor.byteSize();            // payload size in bytes
long absOffset = tensor.absoluteOffset(st);  // absolute file offset
```
