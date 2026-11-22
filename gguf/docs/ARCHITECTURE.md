# GGUF module: architecture (class diagrams)

Companion to [ARCHITECTURE-summary.md](ARCHITECTURE-summary.md). The diagrams below cover
every class in `gguf/src/main/java`, one part of the library per diagram.

1. [Module and package overview](#1-module-and-package-overview)
2. [Public API: the `GGUF` model](#2-public-api-the-gguf-model)
3. [Type system: `GGMLType`, `MetadataValueType`, `TypeDescriptor`](#3-type-system)
4. [Builder hierarchy](#4-builder-hierarchy)
5. [Reading and writing (I/O)](#5-reading-and-writing-io)
6. [Sequence: read → edit → write](#6-sequence-read--edit--write)
7. [On-disk layout handled by the library](#7-on-disk-layout)

---

## 1. Module and package overview

The JPMS module exports only the API package. Public static factories on the interfaces
call `ImplAccessor`, the only `public` class in `impl`, which creates the package-private
implementations.

```mermaid
classDiagram
    direction LR

    namespace com_qxotic_format_gguf {
        class GGUF {
            <<interface>>
        }
        class Builder {
            <<interface>>
        }
        class TensorEntry {
            <<final>>
        }
        class GGMLType {
            <<enumeration>>
        }
        class MetadataValueType {
            <<enumeration>>
        }
        class GGUFFormatException
    }

    namespace com_qxotic_format_gguf_impl {
        class ImplAccessor {
            <<public final>>
        }
        class GGUFImpl
        class BuilderImpl
        class AbstractBuilder {
            <<abstract>>
        }
        class ReaderImpl
        class WriterImpl
        class GGUFFormatter
        class TypeDescriptor
    }

    GGUF ..> ImplAccessor : read / write / toString
    Builder ..> ImplAccessor : newBuilder
    ImplAccessor ..> ReaderImpl : creates
    ImplAccessor ..> WriterImpl : delegates
    ImplAccessor ..> BuilderImpl : creates
    ImplAccessor ..> GGUFFormatter : delegates
    GGUFImpl ..|> GGUF
    AbstractBuilder ..|> Builder
    BuilderImpl --|> AbstractBuilder
    ReaderImpl ..> GGUFImpl : produces
    BuilderImpl ..> GGUFImpl : produces
    GGUFImpl o-- TypeDescriptor
    GGUFImpl o-- TensorEntry
```

> `module com.qxotic.format.gguf { exports com.qxotic.format.gguf; }`, so the `impl`
> package cannot be reached from outside the module.

---

## 2. Public API: the `GGUF` model

`GGUF` is a read-only view of a header. `GGUFImpl` stores it in three ordered, unmodifiable
maps. `at(base)` returns an anonymous delegating `GGUF` whose offsets are shifted by `base`.

```mermaid
classDiagram
    direction TB

    class GGUF {
        <<interface>>
        +getVersion() int
        +getAlignment() int
        +getTensorDataOffset() long
        +getMetadataKeys() Set~String~
        +getValue(Class~T~ cls, String key) T
        +getValueOrDefault(Class~T~ cls, String key, T def) T
        +getString(String key) String
        +getStringOrDefault(String key, String def) String
        +containsKey(String key) boolean
        +getType(String key) MetadataValueType
        +getComponentType(String key) MetadataValueType
        +getTensors() Collection~TensorEntry~
        +getTensor(String name) TensorEntry
        +containsTensor(String name) boolean
        +absoluteOffset(TensorEntry t) long
        +at(long base) GGUF
        +toString(boolean showKeys, boolean showTensors) String
        +toString(boolean, boolean, int maxArray, int maxString) String
        +read(Path)$ GGUF
        +read(ReadableByteChannel)$ GGUF
        +write(GGUF, Path)$ void
        +write(GGUF, WritableByteChannel)$ void
    }

    class GGUFImpl {
        <<final>>
        -int version
        -long tensorDataOffset
        -Map~String,Object~ metadata
        -Map~String,TypeDescriptor~ metadataTypes
        -Map~String,TensorEntry~ tensorInfos
        ~padding(long pos, long align)$ long
        ~toBoxedClass(Class cls)$ Class
        +toString() String
    }

    class ShiftedGGUF {
        <<anonymous, from GGUF.at>>
        -GGUF self
        -long base
        +getTensorDataOffset() long
        +at(long further) GGUF
    }

    class TensorEntry {
        <<final, immutable>>
        -String name
        -long[] shape
        -GGMLType ggmlType
        -long offset
        +create(String, long[], GGMLType, long)$ TensorEntry
        +name() String
        +shape() long[]
        +ggmlType() GGMLType
        +type() GGMLType
        +offset() long
        +byteSize() long
        +totalNumberOfElements() long
        +withOffset(long) TensorEntry
    }

    class GGUFFormatException {
        +GGUFFormatException(String)
        +GGUFFormatException(String, Throwable)
    }
    class IllegalArgumentException

    GGUFImpl ..|> GGUF
    ShiftedGGUF ..|> GGUF
    ShiftedGGUF --> GGUF : delegates to self
    GGUFImpl "1" *-- "*" TensorEntry : tensorInfos (ordered)
    TensorEntry --> GGMLType
    GGUF ..> MetadataValueType
    GGUFFormatException --|> IllegalArgumentException
```

---

## 3. Type system

Two enums describe the two kinds of types in GGUF. `TypeDescriptor` (internal) pairs a
metadata type with its array component type, using one interned instance per
combination.

```mermaid
classDiagram
    direction LR

    class GGMLType {
        <<enumeration>>
        F32 · F16 · BF16 · F64
        I8 · I16 · I32 · I64
        Q4_0 · Q4_1 · Q5_0 · Q5_1 · Q8_0 · Q8_1
        Q2_K · Q3_K · Q4_K · Q5_K · Q6_K · Q8_K
        IQ1_S · IQ1_M · IQ2_XXS · IQ2_XS · IQ2_S
        IQ3_XXS · IQ3_S · IQ4_NL · IQ4_XS
        TQ1_0 · TQ2_0 · MXFP4 · NVFP4 · Q1_0 · Q2_0
        PQ2_0(id 142) · PTQ1_0(id 143)
        Q4_2 · Q4_3 · Q4_0_x_y · IQ4_NL_x_y  (deprecated, size 0)
        -int id
        -int blockByteSize
        -int elementsPerBlock
        -GGMLType[] BY_ID$
        +getId() int
        +getBlockByteSize() int
        +getElementsPerBlock() int
        +isQuantized() boolean
        +getBitsPerWeight() double
        +byteSizeFor(long elements) long
        +elementsForByteSize(long bytes) long
        +fromId(int id)$ GGMLType
    }

    class MetadataValueType {
        <<enumeration>>
        UINT8 · INT8 · UINT16 · INT16
        UINT32 · INT32 · FLOAT32 · BOOL
        STRING · ARRAY
        UINT64 · INT64 · FLOAT64
        -int byteSize
        +byteSize() int
        +fromIndex(int)$ MetadataValueType
    }

    class TypeDescriptor {
        <<final, interned>>
        -TypeDescriptor[] SCALARS$
        -TypeDescriptor[] ARRAYS$
        -MetadataValueType type
        -MetadataValueType componentType
        ~scalar(MetadataValueType)$ TypeDescriptor
        ~array(MetadataValueType)$ TypeDescriptor
        ~type() MetadataValueType
        ~componentType() MetadataValueType
    }

    class TensorEntry
    TensorEntry --> GGMLType : byteSize() = byteSizeFor(n)
    TypeDescriptor --> "1..2" MetadataValueType
```

| Concept | On disk | In Java |
|---|---|---|
| `GGMLType` | `uint32 ggml_type` in each tensor info | `getId()` (ordinal, except PrismML ids) |
| `MetadataValueType` | `uint32` tag before each value | ordinal |
| scalar `UINTn`/`INTn` | n-bit little-endian | `byte`/`short`/`int`/`long` (signed) |
| `STRING` | `uint64 len` + UTF-8 bytes | `String` |
| `ARRAY` | `uint32 componentType` + `uint64 len` + elements | primitive array or `String[]` (no nesting) |

---

## 4. Builder hierarchy

`AbstractBuilder` maps the 24 typed setters to two abstract primitives. `BuilderImpl` owns
the mutable ordered maps and computes the header size and tensor offsets in `build`.

```mermaid
classDiagram
    direction TB

    class Cloneable {
        <<interface>>
    }

    class Builder {
        <<interface>>
        +newBuilder()$ Builder
        +newBuilder(GGUF)$ Builder
        +build() GGUF
        +build(boolean recomputeTensorOffsets) GGUF
        +clone() Builder
        +setVersion(int) Builder
        +getVersion() int
        +setAlignment(int) Builder
        +getAlignment() int
        +putTensor(TensorEntry) Builder
        +removeTensor(String) Builder
        +getTensor(String) TensorEntry
        +getTensors() Collection~TensorEntry~
        +containsTensor(String) boolean
        +containsKey(String) boolean
        +getValue(Class~T~, String) T
        +getType(String) MetadataValueType
        +getComponentType(String) MetadataValueType
        +getMetadataKeys() Set~String~
        +removeKey(String) Builder
        +putString / putBoolean / putByte / putShort
        +putInteger / putLong / putFloat / putDouble
        +putUnsignedByte / Short / Integer / Long
        +putArrayOfXxx(String, xxx[]) Builder
    }

    class AbstractBuilder {
        <<abstract>>
        #putValue(String key, MetadataValueType t, Object v)* BuilderImpl
        #putArray(String key, MetadataValueType comp, Object arr)* BuilderImpl
        +putXxx(...) Builder
        +putArrayOfXxx(...) Builder
    }

    class BuilderImpl {
        <<final>>
        -int version = 3
        -LinkedHashMap~String,Object~ metadata
        -LinkedHashMap~String,TypeDescriptor~ metadataTypes
        -LinkedHashMap~String,TensorEntry~ tensorInfos
        ~fromExisting(GGUF)$ BuilderImpl
        +build(boolean) GGUF
        +clone() BuilderImpl
        -computeTensorDataOffset() long
        -computeTensorOffsets() Map
        -sizeOfTaggedValue(String, Object) long
        -sizeOfArray(String, Object) long
        -sizeOfTensorInfo(TensorEntry)$ long
        -castTo(String, Object, Class)$ T
    }

    Builder --|> Cloneable
    AbstractBuilder ..|> Builder
    BuilderImpl --|> AbstractBuilder
    BuilderImpl o-- TypeDescriptor
    BuilderImpl o-- TensorEntry
    BuilderImpl ..> GGUFImpl : build() creates
    BuilderImpl ..> GGUF : fromExisting() copies
```

`build(recomputeTensorOffsets)`:

```mermaid
flowchart LR
    A[build] --> B{keys and types in sync?}
    B -- no --> X[IllegalStateException]
    B -- yes --> C["computeTensorDataOffset()<br/>magic+version+counts<br/>+ Σ key+tagged value<br/>+ Σ tensor info<br/>+ padding(alignment)"]
    C --> D{recompute offsets?}
    D -- true --> E["computeTensorOffsets()<br/>offset_i = align(end_(i-1))"]
    D -- false --> F[keep existing offsets]
    E --> G[new GGUFImpl]
    F --> G
```

---

## 5. Reading and writing (I/O)

`ReaderImpl` is a stateful single-use parser. `WriterImpl` encodes into a
`ByteArrayOutputStream`, so a header that fails validation writes nothing.
`GGUFFormatter` is used only by `toString`.

```mermaid
classDiagram
    direction LR

    class ImplAccessor {
        <<public final>>
        +newBuilder()$ Builder
        +newBuilder(GGUF)$ Builder
        +defaultAlignment()$ int
        +alignmentKey()$ String
        +read(ReadableByteChannel)$ GGUF
        +write(GGUF, WritableByteChannel)$ void
        +write(GGUF, Path)$ void
        +toString(GGUF, boolean, boolean)$ String
        +toString(GGUF, boolean, boolean, int, int)$ String
    }

    class ReaderImpl {
        <<final>>
        ~GGUF_MAGIC = 0x46554747$
        ~GGUF_VERSION = 3$
        ~ALIGNMENT_DEFAULT_VALUE = 32$
        ~ALIGNMENT_KEY = "general.alignment"$
        ~MAX_TENSOR_NAME_BYTES = 63$
        -MAX_STRING_BYTES = 2^28$
        -MAX_COUNT = 2^26$
        -int version
        -int alignment
        -Map metadata
        -Map metadataTypes
        -long totalBytesRead
        -ByteBuffer BB_8 (LE)
        ~readImpl(ReadableByteChannel) GGUF
        -readHeader(ch) int
        -readTensorEntry(ch) TensorEntry
        -readArray(ch, key) Object
        -readMetadataValueOfType(ch, key, t) Object
        -requireSizedSlots(Collection~TensorEntry~)
        -plausible(long, int, String)$ int
        ~isValidKey(String)$ boolean
        -readFully(ch, ByteBuffer) ByteBuffer
        -getAlignment() int
    }

    class WriterImpl {
        <<final>>
        -ByteBuffer scratch (LE)
        -ByteArrayOutputStream header
        -GGUF gguf
        ~write(GGUF, WritableByteChannel)$
        ~write(GGUF, Path)$  CREATE_NEW
        -serialize(GGUF)$ byte[]
        -writeHeader()
        -writeTensorEntry(TensorEntry)
        -writeArray(MetadataValueType, Object)
        -writeValue(MetadataValueType, Object)
        -writeString(String)
    }

    class GGUFFormatter {
        <<final>>
        +toString(GGUF, boolean, boolean)$ String
        +toString(GGUF, boolean, boolean, int, int)$ String
        -formatMetadata(...)$
        -formatArrayValue(...)$
        -formatTensors(...)$
    }

    ImplAccessor ..> ReaderImpl
    ImplAccessor ..> WriterImpl
    ImplAccessor ..> GGUFFormatter
    ImplAccessor ..> BuilderImpl
    ReaderImpl ..> GGUFImpl : creates
    ReaderImpl ..> TypeDescriptor
    ReaderImpl ..> TensorEntry : creates
    ReaderImpl ..> GGMLType : fromId
    ReaderImpl ..> GGUFFormatException : throws
    WriterImpl ..> GGUF : reads from
    WriterImpl ..> ReaderImpl : shares constants and isValidKey
    GGUFFormatter ..> GGUF
```

### Reader validation pipeline

```mermaid
flowchart TD
    S([ReadableByteChannel]) --> M{"magic == 'GGUF'?"}
    M -- no --> E1[GGUFFormatException]
    M -- yes --> V{version == 3?}
    V -- no --> E1
    V -- yes --> C["tensor_count, kv_count<br/>≤ 2^26 (plausible)"]
    C --> KV["for each KV:<br/>key ≤ 2^16, [a-z0-9_.-]<br/>type tag valid, no duplicates<br/>arrays: no nesting, len ≤ 2^26<br/>strings ≤ 2^28 bytes"]
    KV --> AL["alignment: UINT32,<br/>power of two (default 32)"]
    AL --> TI["for each tensor info:<br/>name ≤ 63 bytes, unique<br/>n_dims ≤ 4, dims > 0<br/>known GGMLType id<br/>offset % alignment == 0"]
    TI --> SL["requireSizedSlots:<br/>gaps < alignment<br/>nested allowed, crossing refused"]
    SL --> P["tensorDataOffset =<br/>bytesRead + padding"]
    P --> R([immutable GGUFImpl])
    TI -. EOF .-> E2[EOFException]
```

---

## 6. Sequence: read → edit → write

```mermaid
sequenceDiagram
    autonumber
    actor App
    participant GGUF as GGUF (interface)
    participant IA as ImplAccessor
    participant R as ReaderImpl
    participant B as BuilderImpl
    participant W as WriterImpl

    App->>GGUF: read(Path)
    GGUF->>IA: read(channel)
    IA->>R: new ReaderImpl().readImpl(channel)
    R-->>App: GGUFImpl (immutable)

    App->>B: Builder.newBuilder(gguf)  (via IA.fromExisting)
    App->>B: putString(...), removeKey(...), putTensor(...)
    App->>B: build(true)
    B->>B: computeTensorDataOffset() and computeTensorOffsets()
    B-->>App: new GGUFImpl

    App->>GGUF: write(gguf, Path)
    GGUF->>IA: write(gguf, path)
    IA->>W: serialize(gguf) into byte[]
    W->>W: check size == tensorDataOffset
    W-->>App: Files.write(CREATE_NEW)
    Note over App: Tensor payload bytes are copied by the caller,<br/>using absoluteOffset(tensor) and byteSize()
```

---

## 7. On-disk layout

The library reads and writes everything up to `tensorDataOffset`. The `tensor_data`
section is left to the caller.

```mermaid
flowchart LR
    subgraph header["header: GGUF.read / GGUF.write"]
        direction TB
        H1["uint32 magic 'GGUF'"] --> H2["uint32 version = 3"]
        H2 --> H3["uint64 tensor_count"]
        H3 --> H4["uint64 metadata_kv_count"]
        H4 --> H5["metadata_kv[]<br/>(key, type, value)"]
        H5 --> H6["tensor_info[]<br/>(name, n_dims, dims[], ggml_type, offset)"]
        H6 --> H7["padding to alignment"]
    end
    subgraph data["tensor_data: not accessed"]
        direction TB
        D1["tensor 0 @ tensorDataOffset + offset0"] --> D2["padding"]
        D2 --> D3["tensor 1 @ tensorDataOffset + offset1"]
        D3 --> D4["..."]
    end
    header -->|"getTensorDataOffset()"| data
```
