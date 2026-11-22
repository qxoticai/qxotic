# safetensors — Architecture (class diagrams)

Source: `safetensors/src/main/java` — 14 types in two packages (~1.5 kLOC).

| Package | Exported | Types |
|---|---|---|
| `com.qxotic.format.safetensors` | yes (`module-info`) | `Safetensors`, `Builder`, `SafetensorsIndex`, `TensorEntry`, `DType`, `SafetensorsFormatException` |
| `com.qxotic.format.safetensors.impl` | no | `ImplAccessor` (public bridge), `ReaderImpl`, `WriterImpl`, `BuilderImpl`, `SafetensorsImpl`, `SafetensorsIndexImpl`, `AlignmentSupport`, `SafetensorsFormatter` |

See [`ARCHITECTURE-overview.md`](ARCHITECTURE-overview.md) for the high-level picture.

Diagram conventions: `$` = static, `*` = abstract, `+` public, `~` package-private, `-` private;
methods marked `«default»` in notes are interface default methods.

---

## 1. Module and package layering

```mermaid
flowchart TB
    subgraph M["module com.qxotic.format.safetensors"]
        subgraph P["com.qxotic.format.safetensors (exported)"]
            Safetensors
            Builder
            SafetensorsIndex
            TensorEntry
            DType
            SafetensorsFormatException
        end
        subgraph I["com.qxotic.format.safetensors.impl (internal)"]
            ImplAccessor
            ReaderImpl
            WriterImpl
            BuilderImpl
            SafetensorsImpl
            SafetensorsIndexImpl
            AlignmentSupport
            SafetensorsFormatter
        end
    end
    JSON["module com.qxotic.format.json<br/>(Json.parse / parseMap / stringify)"]

    Safetensors & Builder & SafetensorsIndex -- "static factories" --> ImplAccessor
    ImplAccessor --> ReaderImpl & WriterImpl & BuilderImpl & SafetensorsIndexImpl & AlignmentSupport & SafetensorsFormatter
    ReaderImpl & WriterImpl & SafetensorsIndexImpl & SafetensorsFormatter --> JSON
    I -. "implements / uses" .-> P
```

The API package depends on `impl` **only** through `ImplAccessor` (the one public class in
`impl`). Because `impl` is not exported, the JPMS boundary keeps every implementation class
invisible to clients while the interfaces can still offer static factories (`Safetensors.read`,
`Builder.newBuilder`, `SafetensorsIndex.load`).

---

## 2. Public API: the data model

```mermaid
classDiagram
    direction LR

    class Safetensors {
        <<interface>>
        +getTensorDataOffset()* long
        +getAlignment()* int
        +getMetadata()* Map~String,String~
        +getTensors()* Collection~TensorEntry~
        +getTensor(String name)* TensorEntry
        +findTensor(String name) Optional~TensorEntry~
        +requireTensor(String name) TensorEntry
        +containsTensor(String name) boolean
        +absoluteOffset(TensorEntry t) long
        +read(ReadableByteChannel ch)$ Safetensors
        +read(Path path)$ Safetensors
        +write(Safetensors st, WritableByteChannel ch)$ void
        +write(Safetensors st, Path path)$ void
    }

    class TensorEntry {
        <<final, immutable>>
        -String name
        -DType dtype
        -long[] shape
        -long byteOffset
        +create(String name, DType dtype, long[] shape, long offset)$ TensorEntry
        +name() String
        +dtype() DType
        +shape() long[]
        +byteOffset() long
        +byteSize() long
        +totalNumberOfElements() long
        +withOffset(long newOffset) TensorEntry
        +equals(Object o) boolean
        +hashCode() int
        +toString() String
    }

    class DType {
        <<enumeration>>
        BOOL
        F4
        F6_E2M3
        F6_E3M2
        U8
        I8
        F8_E5M2
        F8_E4M3
        F8_E8M0
        F8_E4M3FNUZ
        F8_E5M2FNUZ
        I16
        U16
        F16
        BF16
        I32
        U32
        F32
        C64
        F64
        I64
        U64
        -int bitSize
        -Class javaType
        +bitSize() int
        +javaType() Class
        +byteSizeFor(long numElements) long
        +byteSizeForShape(long[] shape) long
    }

    class SafetensorsFormatException {
        +SafetensorsFormatException(String msg)
        +SafetensorsFormatException(String msg, Throwable cause)
    }
    class RuntimeException

    Safetensors "1" o-- "*" TensorEntry : tensors (ordered)
    TensorEntry --> DType : dtype
    RuntimeException <|-- SafetensorsFormatException
    Safetensors ..> SafetensorsFormatException : read() throws
```

Notes:

- `Safetensors` describes **one file's header**: metadata (`__metadata__`, string → string),
  ordered tensor entries, and `tensorDataOffset = 8 + headerSize`. `absoluteOffset(t)` =
  `tensorDataOffset + t.byteOffset()` is the file position of the payload.
- `TensorEntry.byteOffset` is **relative** to the start of the data section. `shape` is copied
  in and out (defensive copies). `byteSize()` is derived from `dtype` and `shape`, so an entry
  holds only the begin offset; the end offset is recomputed on write.
- `DType` constants are declared in increasing alignment order, as upstream. `byteSizeFor` works
  in **bits** so sub-byte types (`F4` = 4 bits, `F6_*` = 6 bits) are sized exactly; it throws
  `IllegalArgumentException` when the bit total is not a whole number of bytes and
  `ArithmeticException` on overflow. `javaType()` is the primitive carrier (`byte.class` for
  FP8/sub-byte, `float.class` for `C64`).

---

## 3. Public API: building and indexing

```mermaid
classDiagram
    direction LR

    class Builder {
        <<interface>>
        +newBuilder()$ Builder
        +newBuilder(Safetensors st)$ Builder
        +build() Safetensors
        +build(boolean recomputeTensorOffsets)* Safetensors
        +clone()* Builder
        +setAlignment(int alignment) Builder
        +getAlignment() int
        +putTensor(TensorEntry e)* Builder
        +removeTensor(String name)* Builder
        +containsTensor(String name)* boolean
        +getTensor(String name)* TensorEntry
        +findTensor(String name) Optional~TensorEntry~
        +requireTensor(String name) TensorEntry
        +getTensors()* Collection~TensorEntry~
        +setMetadata(Map~String,String~ m)* Builder
        +getMetadata()* Map~String,String~
        +putMetadataKey(String k, String v)* Builder
        +getMetadataValue(String k)* String
        +findMetadataValue(String k) Optional~String~
        +removeMetadataKey(String k)* Builder
        +getMetadataKeys()* Set~String~
    }
    class Cloneable {
        <<interface>>
    }

    class SafetensorsIndex {
        <<interface>>
        +load(Path rootPath)$ SafetensorsIndex
        +getRootPath()* Path
        +getSafetensorsPath(String name)* Path
        +containsTensor(String name) boolean
        +findSafetensorsPath(String name) Optional~Path~
        +requireSafetensorsPath(String name) Path
        +getTensorNames()* Collection~String~
    }

    class Safetensors {
        <<interface>>
    }
    class TensorEntry

    Cloneable <|-- Builder
    Builder ..> Safetensors : newBuilder(st) copies / build() creates
    Builder "1" o-- "*" TensorEntry : putTensor
    SafetensorsIndex ..> Safetensors : load() reads single files
```

Notes:

- `Builder.setAlignment` / `getAlignment` are **default** methods implemented purely on top of
  the metadata map: they store/remove the `__alignment__` key (removed when set back to the
  default `1`) and validate "positive power of 2" through `ImplAccessor`.
- `build()` = `build(true)`: offsets are recomputed so tensors are packed back to back in
  insertion order. `build(false)` keeps the entries' offsets as given, but they must still form a
  valid hole-free, overlap-free layout.
- `SafetensorsIndex` maps a **tensor name → shard file `Path`**; it does not keep the parsed
  headers. Callers then `Safetensors.read(path)` the shard they need.
- Every interface has the same accessor triad: `getX` (nullable), `findX` (`Optional`),
  `requireX` (throws `IllegalArgumentException("tensor not found: …")`).

---

## 4. Implementation classes

```mermaid
classDiagram
    direction TB

    class Safetensors {
        <<interface>>
    }
    class Builder {
        <<interface>>
    }
    class SafetensorsIndex {
        <<interface>>
    }

    class ImplAccessor {
        <<public final, static bridge>>
        +read(ReadableByteChannel ch)$ Safetensors
        +write(Safetensors st, WritableByteChannel ch)$ void
        +loadIndex(Path root)$ SafetensorsIndex
        +newBuilder()$ Builder
        +newBuilder(Safetensors st)$ Builder
        +defaultAlignment()$ int
        +alignmentKey()$ String
        +isValidAlignment(int a)$ boolean
        +parseAlignment(String a)$ int
        +toString(Safetensors st, boolean showKeys, boolean showTensors)$ String
    }

    class SafetensorsImpl {
        <<final, immutable>>
        -long tensorDataOffset
        -Map~String,String~ metadata
        -Map~String,TensorEntry~ tensors
        ~SafetensorsImpl(long offset, Map metadata, Map tensors)
        +getAlignment() int
        +toString() String
    }

    class BuilderImpl {
        <<final, mutable>>
        -Map~String,String~ metadata
        -Map~String,TensorEntry~ tensorEntries
        ~fromExisting(Safetensors st)$ BuilderImpl
        +build(boolean recompute) Safetensors
        +clone() BuilderImpl
        -setTensors(Map m) BuilderImpl
        -computeTensorOffsets() Map
        -fromCollection(Collection c)$ Map
    }

    class SafetensorsIndexImpl {
        <<final, immutable>>
        -Path rootPath
        -Map~String,Path~ tensorIndex
        +load(Path path)$ SafetensorsIndex
        -loadSharded(Path root, Path index, Map out)$ void
        -loadSingleFile(Path file, Map out)$ void
        -parseWeightMap(Object v)$ Map
        -resolveShard(Path root, String fileName)$ Path
        -resolveRootPath(Path file)$ Path
    }

    class ReaderImpl {
        <<static utility>>
        ~METADATA_KEY = "__metadata__"$
        ~HEADER_SIZE_BYTES = 8$
        -MAX_HEADER_SIZE = 100_000_000$
        -MAX_SIZE_T = 2^48-1$
        ~read(ReadableByteChannel ch)$ Safetensors
        ~validateTensors(Collection~TensorEntry~ t)$ void
        -parseHeader(Map header)$ ParsedHeader
        -parseMetadata(Object o, Map out)$ void
        -parseTensor(String name, Object o)$ TensorEntry
        -parseDType / parseShape / parseDataOffsets()$
        -parseSizeT(Object v, String field, String name)$ long
        -readFully(ReadableByteChannel ch, ByteBuffer b)$ int
    }
    class ParsedHeader {
        <<private static>>
        ~Map~String,String~ metadata
        ~Map~String,TensorEntry~ tensors
    }

    class WriterImpl {
        <<final, static utility>>
        ~computeTensorDataOffset(Map metadata, Collection tensors, int alignment)$ long
        ~writeImpl(Safetensors st, WritableByteChannel ch)$ void
        ~padding(long position, long alignment)$ long
        -getHeaderBytes(Map metadata, Collection tensors)$ byte[]
        -box(long[] shape)$ List~Long~
    }

    class AlignmentSupport {
        <<final, static utility>>
        ~DEFAULT_VALUE = 1$
        ~KEY = "__alignment__"$
        ~isValid(int a)$ boolean
        ~parse(String raw)$ int
        ~parseMetadataAlignment(Map m)$ int
        ~validateHeaderMetadata(Map m)$ void
    }

    class SafetensorsFormatter {
        <<static utility>>
        +toString(Safetensors st, boolean showMetadata, boolean showTensors)$ String
        -formatMetadata(StringBuilder sb, Safetensors st)$ void
        -formatTensors(StringBuilder sb, Safetensors st)$ void
    }

    Safetensors <|.. SafetensorsImpl
    Builder <|.. BuilderImpl
    SafetensorsIndex <|.. SafetensorsIndexImpl

    ImplAccessor ..> ReaderImpl : read
    ImplAccessor ..> WriterImpl : write
    ImplAccessor ..> BuilderImpl : newBuilder
    ImplAccessor ..> SafetensorsIndexImpl : loadIndex
    ImplAccessor ..> AlignmentSupport : alignment helpers
    ImplAccessor ..> SafetensorsFormatter : toString

    ReaderImpl ..> SafetensorsImpl : creates
    ReaderImpl *-- ParsedHeader
    ReaderImpl ..> AlignmentSupport : validateHeaderMetadata
    BuilderImpl ..> SafetensorsImpl : build() creates
    BuilderImpl ..> ReaderImpl : validateTensors
    BuilderImpl ..> WriterImpl : computeTensorDataOffset
    SafetensorsImpl ..> AlignmentSupport : getAlignment
    SafetensorsImpl ..> ImplAccessor : toString
    SafetensorsFormatter ..> ReaderImpl : METADATA_KEY
    WriterImpl ..> ReaderImpl : METADATA_KEY
    SafetensorsIndexImpl ..> Safetensors : read(file)
```

Responsibilities:

| Class | Role |
|---|---|
| `ImplAccessor` | Only public class of `impl`; static trampolines used by the API interfaces' static/default methods. Keeps the API package free of implementation types. |
| `SafetensorsImpl` | Immutable header snapshot. Copies the incoming maps into unmodifiable `LinkedHashMap`s (a builder keeps mutating its own maps after `build()`). `getAlignment()` is re-parsed from metadata on each call. |
| `BuilderImpl` | Mutable, insertion-ordered maps of metadata and tensors. `build()` optionally re-packs offsets, validates the layout like a reader, computes the padded `tensorDataOffset`, and creates a `SafetensorsImpl`. Rejects null keys/values and duplicate tensor names. |
| `ReaderImpl` | Parses and strictly validates a header from a channel (see §5). Also hosts `validateTensors`, the shared back-to-back layout check. |
| `WriterImpl` | Serializes the header JSON deterministically (insertion order, `dtype` / `shape` / `data_offsets`), computes the data offset (`8 + json`, padded to `max(8, alignment)`), and writes the header padded with spaces up to `tensorDataOffset`. |
| `AlignmentSupport` | The `__alignment__` metadata rule: default `1`, value must be a positive power of 2. |
| `SafetensorsIndexImpl` | Resolves tensor name → shard `Path` from an index JSON, a `model.safetensors`, or a single file; guards against unsafe shard paths. |
| `SafetensorsFormatter` | Human-readable `toString()` (alignment, data offset in hex, metadata, tensors with offsets). |

---

## 5. Reading a file (`ReaderImpl.read`)

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant S as Safetensors (interface)
    participant A as ImplAccessor
    participant R as ReaderImpl
    participant J as Json
    participant AS as AlignmentSupport
    participant DT as DType
    participant SI as SafetensorsImpl

    C->>S: read(path)
    S->>S: open BufferedInputStream (64 KiB) as channel
    S->>A: read(channel)
    A->>R: read(channel)
    R->>R: readFully 8 bytes, u64 LE headerSize in (0, 100_000_000]
    R->>R: readFully headerSize bytes, first byte must be '{'
    R->>R: strict UTF-8 decode
    R->>J: parseMap(headerJson)
    loop each top-level key (insertion order)
        alt key == "__metadata__"
            R->>R: object of string → string
            R->>AS: validateHeaderMetadata (__alignment__)
        else tensor
            R->>R: exactly {dtype, shape, data_offsets}
            R->>DT: valueOf(dtype)
            R->>R: shape / offsets: integers in [0, 2^48), begin ≤ end
            R->>DT: byteSizeForShape(shape) == end - begin ?
            R->>R: TensorEntry.create(name, dtype, shape, begin)
        end
    end
    R->>R: validateTensors: sorted by offset, contiguous from 0
    R->>SI: new SafetensorsImpl(8 + headerSize, metadata, tensors)
    SI-->>C: Safetensors
```

Every validation failure raises `SafetensorsFormatException` (with the tensor name in the
message); short reads raise `IOException`.

---

## 6. Building and writing (`BuilderImpl.build` → `WriterImpl.writeImpl`)

```mermaid
sequenceDiagram
    autonumber
    participant C as Caller
    participant B as BuilderImpl
    participant R as ReaderImpl
    participant W as WriterImpl
    participant SI as SafetensorsImpl
    participant Ch as WritableByteChannel

    C->>B: newBuilder(existing) / newBuilder()
    C->>B: putTensor / putMetadataKey / setAlignment(64)
    C->>B: build()
    opt recomputeTensorOffsets
        B->>B: computeTensorOffsets(): offset += dtype.byteSizeForShape(shape)
    end
    B->>R: validateTensors(entries)
    B->>W: computeTensorDataOffset(metadata, entries, alignment)
    W-->>B: 8 + len(json) padded to max(8, alignment)
    B->>SI: new SafetensorsImpl(offset, metadata, entries)
    SI-->>C: Safetensors
    C->>W: Safetensors.write(st, channel) via ImplAccessor
    W->>W: json = getHeaderBytes(metadata, tensors)
    W->>W: headerSize = tensorDataOffset - 8 (must be ≥ len(json))
    W->>Ch: u64 LE headerSize + json + ' ' padding
    Note over C,Ch: caller then writes tensor payload bytes in offset order
```

Because `writeImpl` pads up to the instance's **own** `tensorDataOffset`, a file that was read and
then written back unchanged keeps its exact layout, so its tensor offsets stay valid.

---

## 7. Sharded model index (`SafetensorsIndexImpl.load`)

```mermaid
flowchart TD
    A["SafetensorsIndex.load(path)"] --> B{path is directory?}
    B -- yes --> C{"model.safetensors.index.json<br/>exists?"}
    C -- yes --> D["loadSharded:<br/>Json.parse → object<br/>weight_map: string → string"]
    D --> E["resolveShard(fileName) for each entry:<br/>• valid Path, relative<br/>• normalized path stays under root<br/>• ends with .safetensors<br/>• is a regular file"]
    C -- no --> F{"model.safetensors<br/>exists?"}
    F -- yes --> G["loadSingleFile:<br/>Safetensors.read(file)<br/>every tensor → file"]
    F -- no --> X1[IOException: no safetensors files]
    B -- no --> H{"regular file<br/>ending .safetensors?"}
    H -- yes --> G2["root = parent dir<br/>loadSingleFile(path)"]
    H -- no --> X2[IOException: bad path]
    E --> R["SafetensorsIndexImpl(root,<br/>unmodifiable name → Path map)"]
    G --> R
    G2 --> R
```

Schema problems in the index JSON or unsafe shard paths raise `SafetensorsFormatException`;
missing files raise `IOException`.

---

## 8. Invariants and design rules (summary)

| Rule | Enforced in |
|---|---|
| Header size in `(0, 100 MB]`, starts with `{`, strict UTF-8 | `ReaderImpl.read` |
| Tensor object has exactly `dtype`, `shape`, `data_offsets` | `ReaderImpl.validateTensorFields` |
| Integers are plain JSON integers in `[0, 2^48)` (no `1.0`, no `1e0`) | `ReaderImpl.parseSizeT` |
| `end - begin == dtype.byteSizeForShape(shape)`, sub-byte sizes byte-aligned | `ReaderImpl.parseTensor`, `DType.byteSizeFor` |
| Tensors contiguous from 0: no hole, no overlap (anti-polyglot) | `ReaderImpl.validateTensors` (reader **and** builder) |
| `__metadata__` is string → string; `__alignment__` is a power of 2 | `ReaderImpl.parseMetadata`, `AlignmentSupport` |
| Data section starts on a multiple of `max(8, alignment)`; padding is spaces | `WriterImpl.computeTensorDataOffset`, `writeImpl` |
| Deterministic header bytes (insertion order everywhere) | `LinkedHashMap` in `BuilderImpl`, `ReaderImpl`, `WriterImpl` |
| Shard paths relative, inside root, `.safetensors`, existing | `SafetensorsIndexImpl.resolveShard` |
| Immutable results, defensive copies of `shape` and maps | `TensorEntry`, `SafetensorsImpl`, `SafetensorsIndexImpl` |
