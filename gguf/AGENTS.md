# GGUF - AGENTS.md

## Project Overview

GGUF (GGML Universal Format) is a Java library for reading and writing GGUF files, the binary format used by [llama.cpp](https://github.com/ggml-org/llama.cpp) for storing machine learning model weights and metadata.

### Key Features

- Read GGUF files from paths, channels, or URLs
- Type-safe metadata access with automatic casting
- Tensor information with offset/size calculations
- Builder API for creating and modifying GGUF files
- Support for all GGML quantization types (Q4_0, Q8_0, Q2_K, etc.)

### Architecture

```
ai.qxotic.format.gguf/
├── GGUF.java              # Main interface - read/write operations
├── Builder.java           # Builder interface for creating/modifying
├── GGMLType.java          # Enum of tensor data types
├── TensorEntry.java       # Tensor metadata (name, shape, type, offset)
├── MetadataValueType.java # Metadata type enum
├── GGUFFormatException.java
└── impl/                  # Internal implementations
    ├── GGUFImpl.java      # GGUF implementation
    ├── BuilderImpl.java   # Builder implementation
    ├── ReaderImpl.java    # Binary format reader
    ├── WriterImpl.java    # Binary format writer
    └── TypeDescriptor.java
```

### What This Library Does NOT Provide

- Tensor data I/O - users must read/write tensor bytes manually
- Quantization/dequantization - raw bytes only
- Inference capabilities - metadata and structure only

Users are responsible for reading tensor data at offsets provided by the API.

---

## Build Commands

Use `mvnd` (Maven Daemon) for faster builds.

### Compile

```bash
mvnd compile
```

### Run All Tests

```bash
mvnd test
```

### Run Single Test

```bash
# Run specific test class
mvnd test -Dtest=GGUFTest

# Run specific test method
mvnd test -Dtest=BuilderTest#testPutMetadata

# Run tests matching pattern
mvnd test -Dtest=*Test
```

### Run with Verbose Output

```bash
mvnd test -Dtest=GGUFTest -X
```

---

## Code Formatting

This project uses Spotless with Google Java Format (AOSP style).

### Check Formatting

```bash
mvnd spotless:check
```

### Apply Formatting

```bash
mvnd spotless:apply
```

### Format Rules

- 4-space indentation (AOSP style)
- Maximum line width: 100 characters
- Remove unused imports
- Unix line endings (LF)
- Trim trailing whitespace
- End files with newline

---

## Documentation

Documentation uses MkDocs Material theme with `pymdownx.snippets` extension.

### Snippet System

Code examples are embedded in `src/test/java/ai/qxotic/format/gguf/Snippets.java` using snippet markers:

```java
// --8<-- [start:snippet-name]
code here
// --8<-- [end:snippet-name]
```

In markdown, reference with:

```markdown
```java
--8<-- "src/test/java/ai/qxotic/format/gguf/Snippets.java:snippet-name"
```
```

### Preview Documentation

```bash
pip install mkdocs-material
cd gguf
mkdocs serve
```

### Build Documentation

```bash
mkdocs build
```

---

## API Quick Reference

### Reading GGUF Files

```java
// From path
GGUF gguf = GGUF.read(Path.of("model.gguf"));

// From channel
try (var channel = Files.newByteChannel(path)) {
    GGUF gguf = GGUF.read(channel);
}

// From URL
URL url = new URL("https://huggingface.co/user/repo/resolve/main/model.gguf");
try (var channel = Channels.newChannel(url.openStream())) {
    GGUF gguf = GGUF.read(channel);
}
```

### Accessing Metadata

```java
// Get all keys
Set<String> keys = gguf.getMetadataKeys();

// Get typed values
String name = gguf.getValue(String.class, "general.name");
int ctxLen = gguf.getValue(int.class, "llama.context_length");
float[] freqs = gguf.getValue(float[].class, "rope.freqs");

// Check existence
if (gguf.containsKey("key")) { ... }

// Get type information
MetadataValueType type = gguf.getType("key");
```

### Accessing Tensors

```java
// List all tensors
for (TensorEntry tensor : gguf.getTensors()) {
    System.out.println(tensor.name());
}

// Get specific tensor
TensorEntry tensor = gguf.getTensor("token_embd.weight");

// Tensor properties
String name = tensor.name();
GGMLType type = tensor.ggmlType();
long[] shape = tensor.shape();
long offset = tensor.offset();        // Relative to tensorDataOffset
long byteSize = tensor.byteSize();    // Bytes required
```

### Reading Tensor Data

```java
TensorEntry tensor = gguf.getTensor("weights");
long absoluteOffset = gguf.getTensorDataOffset() + tensor.offset();

try (var raf = new RandomAccessFile("model.gguf", "r");
     var channel = raf.getChannel()) {
    var buffer = channel.map(FileChannel.MapMode.READ_ONLY, 
                             absoluteOffset, tensor.byteSize());
    buffer.order(ByteOrder.nativeOrder());
    // buffer contains raw tensor data
}
```

### Building GGUF Files

```java
Builder builder = Builder.newBuilder()
    .putString("general.name", "my-model")
    .putInteger("llama.context_length", 4096)
    .putFloat("llama.rope.freq_base", 10000.0f)
    .putTensor(TensorEntry.create("weights", new long[]{1024, 1024}, GGMLType.F32, 0));

GGUF gguf = builder.build();
GGUF.write(gguf, Path.of("output.gguf"));
```

### Modifying Existing Files

```java
GGUF existing = GGUF.read(Path.of("model.gguf"));
Builder builder = Builder.newBuilder(existing)
    .putString("general.description", "Modified")
    .removeKey("old_key");

GGUF modified = builder.build();
```

---

## Code Style Guidelines

### Imports

- Use explicit imports, avoid wildcards
- Order: java.*, javax.*, org.*, com.*, ai.qxotic.*
- Static imports for test assertions

### Naming

- Interfaces: noun (GGUF, Builder)
- Classes: noun (TensorEntry, MetadataValueType)
- Methods: verb or noun (read, write, getMetadataKeys)
- Constants: UPPER_SNAKE_CASE

### Types

- Prefer `long` for file offsets and sizes
- Use `int` for counts and small values
- Always use `this.` prefix for field access

### Error Handling

- Throw `GGUFFormatException` for format violations
- Throw `IllegalArgumentException` for invalid arguments
- Throw `IOException` for I/O errors

### Comments

- Use `// fall-through` in switch statements
- No unnecessary comments - code should be self-documenting
- Javadoc on public API only

---

## Common Tasks

### Add New Snippet

1. Add method to `Snippets.java` with markers
2. Reference in `docs/DOCUMENTATION.md`
3. Run `mvnd compile -Pdocs`

### Add New GGMLType

1. Add to `GGMLType.java` enum with correct block size
2. Mark deprecated types with `@Deprecated`

### Add New Metadata Type

1. Add to `MetadataValueType.java`
2. Update `ReaderImpl` and `WriterImpl`
3. Add builder method to `AbstractBuilder`
