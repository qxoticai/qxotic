# GGUF module: architecture summary

`gguf` (artifact `com.qxotic:gguf`, JPMS module `com.qxotic.format.gguf`) is a small,
dependency-free Java library that **reads and writes the header of GGUF files**, the binary
container llama.cpp / GGML use for model weights.

It handles only the *header*: metadata key/values and tensor descriptors (name, shape, type,
offset). **Tensor payload bytes are never read, copied or written.** Callers use
`GGUF.getTensorDataOffset()` / `GGUF.absoluteOffset(tensor)` to locate the data themselves
(typically by memory-mapping the file).

See [ARCHITECTURE.md](ARCHITECTURE.md) for class diagrams.

## Size and layout

16 source files, ~3,000 lines, in two packages:

| Package | Visibility | Role |
|---|---|---|
| `com.qxotic.format.gguf` | exported | Public API: interfaces, value types, enums |
| `com.qxotic.format.gguf.impl` | **not exported** | Reader, writer, builder and immutable model implementations |

`module-info.java` exports only the API package, so the implementation classes are
unreachable from other modules. The public interfaces reach the implementation through
one bridge class, `impl.ImplAccessor`, through static factory methods
(`GGUF.read`, `GGUF.write`, `Builder.newBuilder`).

## Public API (`com.qxotic.format.gguf`)

| Type | Kind | Responsibility |
|---|---|---|
| `GGUF` | interface | Read-only view of a parsed header: version, alignment, tensor-data offset, ordered metadata, ordered tensors. Static entry points `read(Path / ReadableByteChannel)` and `write(GGUF, Path / WritableByteChannel)`. `at(base)` returns a view whose offsets are shifted for a GGUF embedded in a larger channel (e.g. an archive entry). |
| `Builder` | interface | Mutable fluent builder that creates a new `GGUF` or edits a copy of an existing one. Typed `putXxx` / `putArrayOfXxx` for every metadata type, tensor add/remove, `build(recomputeTensorOffsets)`. Not thread-safe. |
| `TensorEntry` | final class | Immutable tensor descriptor: `name`, `shape` (defensively copied), `GGMLType`, `offset` relative to the tensor-data section. Computes `totalNumberOfElements()` and `byteSize()`. Validates that every dimension is positive and that the element count fits in a `long`. |
| `GGMLType` | enum | Every `ggml_type` id: F32/F16/BF16/integers, legacy quants (Q4_0…Q8_1), K-quants, IQ quants, ternary (TQ), MXFP4/NVFP4, Q1_0/Q2_0, and the PrismML fork types (`PQ2_0`, `PTQ1_0`, ids 142/143). Each holds `blockByteSize` and `elementsPerBlock`, and provides `byteSizeFor(n)`, `elementsForByteSize(b)`, `getBitsPerWeight()` and `fromId(id)`. Types removed from GGML stay as `@Deprecated` placeholders with size 0 so the ids stay correct. |
| `MetadataValueType` | enum | The 13 `gguf_metadata_value_type` values (UINT8 … FLOAT64, STRING, ARRAY). The ordinal is the on-disk tag. `byteSize()` is positive for fixed-size types and negative (−8, the length prefix) for STRING and ARRAY. |
| `GGUFFormatException` | exception | Unchecked (`extends IllegalArgumentException`) error for content that breaks the spec: bad magic, unsupported version, invalid keys/types, bad tensor layout. |

### Value mapping

Metadata values are stored as plain Java objects. Unsigned GGUF types map to the
**signed Java type of the same width** (UINT32 → `int`), and callers do the unsigned
conversion themselves. Arrays map to primitive arrays or `String[]` and are **shared by
reference** (no defensive copies) between builders, clones and built instances. Nested
arrays (array of arrays) are rejected everywhere.

## Implementation (`com.qxotic.format.gguf.impl`)

| Class | Role |
|---|---|
| `ImplAccessor` | Public-but-unexported bridge: factories for the builder, `read`/`write`, `toString`, default alignment (32) and alignment key (`general.alignment`). |
| `GGUFImpl` | Immutable `GGUF` implementation: three unmodifiable `LinkedHashMap`s (metadata values, `TypeDescriptor`s, tensors) plus version and tensor-data offset. Also holds shared helpers `padding(pos, align)` and `toBoxedClass`. |
| `TypeDescriptor` | Interned pair *(type, componentType)*: one cached instance per scalar type and per array-of-component type. |
| `ReaderImpl` | Streaming little-endian parser over a `ReadableByteChannel`. Counts bytes read to compute the tensor-data offset. Does defensive validation (see below). |
| `WriterImpl` | Serialises the whole header **into memory first**, then writes it in a single pass. A refused GGUF therefore writes no bytes. Checks that the serialised size equals `getTensorDataOffset()`. `write(Path)` uses `CREATE_NEW` and never overwrites. |
| `AbstractBuilder` | Turns the 24 typed `put*` methods into two primitives, `putValue(key, type, value)` and `putArray(key, componentType, array)`. |
| `BuilderImpl` | Concrete builder backed by `LinkedHashMap`s (insertion order = file order). `build()` computes the exact header size, i.e. the tensor-data offset, and optionally repacks tensor offsets one after another at the alignment. |
| `GGUFFormatter` | Human-readable `toString` with options to show keys/tensors and to shorten long arrays/strings. |

## Main flows

**Read**: `GGUF.read(path)` → `ImplAccessor.read` → `ReaderImpl.readImpl`:
1. Header: magic `GGUF` (0x46554747), version must be **3**, `tensor_count`, `metadata_kv_count`.
2. Metadata KVs: key (validated), type tag, value (scalar, string or 1-level array).
3. Tensor infos: name (≤ 63 bytes), ≤ 4 dims, `GGMLType` id, aligned offset.
4. `requireSizedSlots`: checks the tensor layout (see below).
5. Pad to alignment → `tensorDataOffset`; return an immutable `GGUFImpl`.

**Build / edit**: `Builder.newBuilder()` or `newBuilder(existing)` → `put*` / `remove*` →
`build(true)` repacks tensors and computes the new tensor-data offset, or `build(false)`
keeps the current offsets (layout-preserving metadata edits).

**Write**: `GGUF.write(gguf, path)` → `WriterImpl.serialize` (validate + encode to a byte
array, check size against tensor-data offset) → one write to the file or channel.

## Robustness and validation

The reader assumes its input may be corrupt or hostile:

- **Plausibility caps** on length fields to avoid multi-GB allocations: string ≤ 2^28 bytes,
  counts/array lengths ≤ 2^26, dims ≤ 4.
- **Keys** must use `[a-z0-9_.-]` and be shorter than 65,536 characters. Duplicate keys and
  duplicate tensor names are rejected.
- **Alignment** (`general.alignment`) must be UINT32 and a power of two. Every tensor offset
  must be aligned to it.
- **Tensor-layout check** (`requireSizedSlots`): tensors sorted by offset must fill the data
  section with only alignment padding between them. A tensor may sit entirely inside
  another (views, e.g. Q/K slices of a stacked QK), but tensors that partly overlap are
  rejected. This catches a `GGMLType` whose size is wrong, such as a fork's layout stored
  under an upstream id, before it decodes garbage.
- Truncated input raises `EOFException`. Everything else raises `GGUFFormatException`.

The writer applies the same rules (version 3, key format, name length, ≤ 4 dims, aligned
offsets) before it writes anything.

## Design traits

- **Small API, hidden implementation**: interfaces + static factories, impl package not exported.
- **Immutable results, mutable builder**: `GGUF`/`TensorEntry` are immutable (apart from
  shared arrays); `Builder` is the only mutable type.
- **Order preserving**: `LinkedHashMap` everywhere, so a read → write round trip keeps the
  key and tensor order.
- **Header only**: no tensor I/O, no dequantisation. `GGMLType` provides only sizing
  information that other modules can use for mmap/slicing.
- **No dependencies**: JDK only (`java.nio` channels, `ByteBuffer` little-endian).
