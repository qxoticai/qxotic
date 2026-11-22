# jota-memory — Architecture

`jota-memory` is the storage layer under jota tensors.
It answers three questions:

| Question | Abstraction |
|---|---|
| *Where are the bytes?* | `Memory<B>` — a block of bytes on a `Device`, backed by `B` (a Java array, a `ByteBuffer`, a `MemorySegment`, or an opaque GPU handle) |
| *How do I read them as a tensor?* | `MemoryView<B>` — dtype + `Layout` (shape/stride) + byte offset over a `Memory<B>` |
| *Who allocates, reads and copies them?* | `MemoryDomain<B>` — bundles a `MemoryAllocator<B>`, an optional `MemoryAccess<B>` and `MemoryOperations<B>` |

The type parameter `B` (the *backing type*) runs through every interface, so a `MemoryView<float[]>` can only be handled by a `MemoryDomain<float[]>`, and the compiler keeps backends apart.

The module depends only on `jota-core` (`DataType`, `Device`/`DeviceType`, `Shape`, `Stride`, `Layout`, `View`, `ViewTransforms`).

Packages:

- `com.qxotic.jota.memory` — the public API: interfaces, the four entry-point classes (`Memories`, `MemoryAllocators`, `MemoryDomains`, `MemoryViews`), copy and printing helpers.
- `com.qxotic.jota.memory.internal` — the implementations; not API.
  Only the `*Factory` classes are `public`, as bridges for the entry points.

For the Java 25 side (`java.lang.foreign`, `Unsafe`, `ByteBuffer`, mapped files), see [ARCHITECTURE-java25.md](ARCHITECTURE-java25.md).
For files (read-only weights, read-write state, the prompt cache on disk), see [ARCHITECTURE-in-out.md](ARCHITECTURE-in-out.md).

The diagrams below go from the big picture to the details:

1. [Core model](#1-core-model)
2. [Memory and allocator hierarchy (lifetimes)](#2-memory-and-allocator-hierarchy-lifetimes)
3. [Domains: the backends](#3-domains-the-backends)
4. [Element access and bulk operations per backend](#4-element-access-and-bulk-operations-per-backend)
5. [Views, strided copy and printing](#5-views-strided-copy-and-printing)
6. [Entry points and internal factories](#6-entry-points-and-internal-factories)
7. [Checks and low-level utilities](#7-checks-and-low-level-utilities)
8. [How a copy is routed](#8-how-a-copy-is-routed)
9. [How jota-tensor and the backends use the memory layer](#9-how-jota-tensor-and-the-backends-use-the-memory-layer)

---

## 1. Core model

The four public interfaces a backend implements, how `MemoryDomain` groups them, and how a `MemoryView` sits on top of a `Memory`.
Types in the `jota-core` namespace come from the `jota-core` module.

```mermaid
classDiagram
    direction LR

    namespace jota_core {
        class Device
        class DataType
        class Layout
        class Shape
        class Stride
        class View {
            <<interface>>
        }
    }

    class Memory~B~ {
        <<interface>>
        +byteSize() long
        +isReadOnly() boolean
        +device() Device
        +base() B
        +memoryGranularity() long
        +supportsDataType(DataType) boolean
    }

    class MemoryView~B~ {
        <<interface>>
        +memory() Memory~B~
        +dataType() DataType
        +layout() Layout
        +byteOffset() long
        +byteStride() Stride
    }

    class MemoryDomain~B~ {
        <<interface>>
        +device() Device
        +memoryAllocator() MemoryAllocator~B~
        +directAccess() MemoryAccess~B~
        +memoryOperations() MemoryOperations~B~
        +copy(MemoryView~B~ src, MemoryView~B~ dst)
        +copy(srcDomain, src, dstDomain, dst)$
        +close()
    }

    class MemoryAllocator~B~ {
        <<interface>>
        +device() Device
        +allocateMemory(byteSize, byteAlignment) Memory~B~
        +allocateMemory(DataType, Shape) Memory~B~
        +defaultByteAlignment() long
        +memoryGranularity() long
    }

    class MemoryAccess~B~ {
        <<interface>>
        +readByte/Short/Int/Long/Float/Double(memory, byteOffset)
        +writeByte/Short/Int/Long/Float/Double(memory, byteOffset, value)
    }

    class MemoryOperations~B~ {
        <<interface>>
        +copy(src, srcOff, dst, dstOff, byteSize)
        +copyFromNative(Memory~MemorySegment~ src, ..., Memory~B~ dst, ...)
        +copyToNative(Memory~B~ src, ..., Memory~MemorySegment~ dst, ...)
        +fillByte/Short/Int/Long/Float/Double(memory, off, size, value)
        +copy(srcOps, src, ..., dstOps, dst, ...)$
    }

    View <|-- MemoryView~B~
    MemoryView~B~ --> "1" Memory~B~ : over
    MemoryView~B~ --> Layout
    MemoryView~B~ --> DataType
    Layout --> Shape
    Layout --> Stride
    Memory~B~ --> Device : lives on

    MemoryDomain~B~ *-- "1" MemoryAllocator~B~ : allocates
    MemoryDomain~B~ *-- "0..1" MemoryAccess~B~ : reads/writes elements (null = opaque, e.g. GPU)
    MemoryDomain~B~ *-- "1" MemoryOperations~B~ : bulk copy / fill
    MemoryAllocator~B~ ..> Memory~B~ : creates
    MemoryAccess~B~ ..> Memory~B~ : by byte offset
    MemoryOperations~B~ ..> Memory~B~ : by byte range
```

Key points:

- **Byte addressing everywhere.** Offsets and sizes in `MemoryAccess` and `MemoryOperations` are in bytes, whatever the backing type.
  A `float[]` is addressed as bytes too.
- **Granularity.** `memoryGranularity()` is 1 for byte-addressable memory (segments, buffers, `byte[]`), otherwise the element size of the backing array (4 for `float[]`).
  A non-byte memory only supports dtypes of exactly that size (`supportsDataType`); `boolean[]` only supports `BOOL`.
- **`MemoryAccess` is optional.** A domain whose memory the host cannot address (a GPU) returns `null` from `directAccess()`.
  Everything generic (strided copies, `arange`, printing staging) then goes through `copyToNative` / `copyFromNative`, which every backend must implement.
- **`MemorySegment` is the common currency.** Cross-backend transfers stage through a `Memory<MemorySegment>` (heap or native), so N backends need N pairs of `copy*Native` methods, not N² copy routines.

---

## 2. Memory and allocator hierarchy (lifetimes)

Who owns the bytes, and when are they freed?
The interfaces split *plain* memory (GC- or arena-managed) from *scoped* memory that can be freed one buffer at a time.

```mermaid
classDiagram
    direction TB

    class Memory~B~ {
        <<interface>>
    }
    class ScopedMemory~B~ {
        <<interface>>
        +close()
    }
    class AutoCloseable {
        <<interface>>
    }

    class MemoryAllocator~B~ {
        <<interface>>
        +allocateMemory(byteSize, byteAlignment) Memory~B~
    }
    class ScopedMemoryAllocator~B~ {
        <<interface>>
        +allocateMemory(byteSize, byteAlignment) ScopedMemory~B~
    }
    class MemoryArena~B~ {
        <<interface>>
        +close()
        +isAlive() boolean
    }
    class ScopedArena~B~ {
        <<interface>>
    }

    Memory~B~ <|-- ScopedMemory~B~
    AutoCloseable <|-- ScopedMemory~B~
    MemoryAllocator~B~ <|-- ScopedMemoryAllocator~B~
    MemoryAllocator~B~ <|-- MemoryArena~B~
    AutoCloseable <|-- MemoryArena~B~
    ScopedMemoryAllocator~B~ <|-- ScopedArena~B~
    MemoryArena~B~ <|-- ScopedArena~B~

    namespace internal_memories {
        class ArrayMemory~B~ {
            -B array
            -long byteSize
            -int elementByteSize
            isReadOnly() false
        }
        class ByteBufferMemory {
            -ByteBuffer byteBuffer
            +asReadOnly() ByteBufferMemory
        }
        class NativeMemorySegmentMemory {
            <<record>>
            MemorySegment memorySegment
            +asReadOnly()
        }
        class RawScopedMemory {
            -long mallocAddress
            -MemorySegment memorySegment
            -AtomicBoolean isMemoryFreed
            +close() Unsafe.freeMemory
        }
        class ArenaScopedMemory {
            <<anonymous>>
            close() untracks, then frees
        }
    }

    namespace internal_allocators {
        class ArrayMemoryAllocator~B~ {
            -Class~B~ arrayType
            -int elementByteSize
            -IntFunction~B~ arrayFactory
        }
        class ByteBufferAllocator {
            -boolean direct
            native byte order, aligned slice
        }
        class ArenaAllocator {
            -Arena arena
            -boolean owns
            -GLOBAL$ ArenaAllocator
            +of(Arena)$
            +adopt(Arena)$
            defaultByteAlignment() 64
        }
        class NativeUnsafeAllocator {
            <<singleton>>
            Unsafe.allocateMemory, over-allocate + align
        }
        class NativeUnsafeAllocatorArena {
            -Set~ScopedMemory~ allocations
            -volatile boolean closed
            synchronized allocate / close
        }
    }

    Memory~B~ <|.. ArrayMemory~B~
    Memory~B~ <|.. ByteBufferMemory
    Memory~B~ <|.. NativeMemorySegmentMemory
    ScopedMemory~B~ <|.. RawScopedMemory
    ScopedMemory~B~ <|.. ArenaScopedMemory

    MemoryAllocator~B~ <|.. ArrayMemoryAllocator~B~
    MemoryAllocator~B~ <|.. ByteBufferAllocator
    MemoryArena~B~ <|.. ArenaAllocator
    ScopedMemoryAllocator~B~ <|.. NativeUnsafeAllocator
    ScopedArena~B~ <|.. NativeUnsafeAllocatorArena

    ArrayMemoryAllocator~B~ ..> ArrayMemory~B~ : new array
    ByteBufferAllocator ..> ByteBufferMemory : allocate / allocateDirect
    ArenaAllocator ..> NativeMemorySegmentMemory : arena.allocate (zero-filled)
    NativeUnsafeAllocator ..> RawScopedMemory : malloc (not zeroed)
    NativeUnsafeAllocatorArena --> NativeUnsafeAllocator : delegates
    NativeUnsafeAllocatorArena ..> ArenaScopedMemory : wraps each RawScopedMemory
```

| Allocator | Entry point | Backing `B` | Device | Freed when | Zero-filled | Use after free |
|---|---|---|---|---|---|---|
| `ArrayMemoryAllocator` (×7) | `MemoryDomains.floats()` … | `boolean[]` … `double[]` | `JAVA:0` | GC | yes | impossible |
| `ByteBufferAllocator` | `MemoryAllocators.newByteBuffer(direct)` | `ByteBuffer` | `JAVA:0` | GC (Cleaner for direct) | yes | impossible |
| `ArenaAllocator` (borrowed) | `MemoryAllocators.ofArena(arena)` | `MemorySegment` | `PANAMA:0` | the arena's creator closes it; `close()` is a no-op | yes | `IllegalStateException` |
| `ArenaAllocator` (adopted) | `MemoryAllocators.adoptArena(arena)` | `MemorySegment` | `PANAMA:0` | `close()` closes the arena | yes | `IllegalStateException` |
| `NativeUnsafeAllocatorArena` | `MemoryAllocators.newScopedArena()` | `MemorySegment` | `PANAMA:0` | each `ScopedMemory.close()`, the rest at arena `close()` | **no** | **undefined** |

There is intentionally **no default native allocator**: native memory always names its lifetime.

`StridedCopy.releaseTemp` uses the scoped branch: a temporary staging buffer that happens to be a `ScopedMemory` is freed right away; arena memory waits for its arena.

---

## 3. Domains: the backends

A `MemoryDomain` is a backend: allocator + access + operations for one backing type on one device.
Three record implementations cover all the host backends; GPU backends (in other modules) implement the same interface with `directAccess() == null`.

```mermaid
classDiagram
    direction LR

    class MemoryDomain~B~ {
        <<interface>>
        +device() Device
        +memoryAllocator() MemoryAllocator~B~
        +directAccess() MemoryAccess~B~
        +memoryOperations() MemoryOperations~B~
        +memoryGranularity() long
        +supportsDataType(DataType) boolean
        +copy(src, dst)
        +copy(srcDomain, src, dstDomain, dst)$
        +close()
    }

    class ArrayMemoryDomain~B~ {
        <<record>>
        Class~B~ arrayType
        MemoryAllocator~B~ memoryAllocator
        MemoryAccess~B~ directAccess
        MemoryOperations~B~ memoryOperations
        close() no-op
    }
    class ByteBufferDomain {
        <<record>>
        MemoryAllocator~ByteBuffer~ memoryAllocator
        close() no-op
    }
    class NativeMemoryDomain {
        <<record>>
        MemoryAllocator~MemorySegment~ memoryAllocator
        requires a PANAMA allocator
        close() closes it if it is a MemoryArena
    }
    class GpuDomain {
        <<other modules>>
        directAccess() null
    }

    MemoryDomain~B~ <|.. ArrayMemoryDomain~B~
    MemoryDomain~B~ <|.. ByteBufferDomain
    MemoryDomain~B~ <|.. NativeMemoryDomain
    MemoryDomain~B~ <|.. GpuDomain

    class DomainFactory {
        <<internal bridge>>
        -BOOLEANS$ / BYTES$ / SHORTS$ / INTS$ / LONGS$ / FLOATS$ / DOUBLES$
        +ofBooleans()$ ... ofDoubles()$
        +ofMemorySegment(allocator)$
        +ofByteBuffer(allocator)$
    }
    DomainFactory ..> ArrayMemoryDomain~B~ : 7 shared singletons
    DomainFactory ..> ByteBufferDomain : new per allocator
    DomainFactory ..> NativeMemoryDomain : via NativeMemoryFactory

    ArrayMemoryDomain~B~ o-- ArrayMemoryAllocator~B~
    ArrayMemoryDomain~B~ o-- ArrayAccess : one of 7
    ArrayMemoryDomain~B~ o-- ArrayOperations : one of 7
    ByteBufferDomain o-- ByteBufferAllocator
    ByteBufferDomain o-- ByteBufferMemoryAccess
    ByteBufferDomain o-- ByteBufferMemoryOperations
    NativeMemoryDomain o-- PanamaAllocator : any PANAMA allocator

    class ArrayAccess {
        <<placeholder>>
        BooleansMemoryAccess ... DoublesMemoryAccess
    }
    class ArrayOperations {
        <<placeholder>>
        BooleansMemoryOperations ... DoublesMemoryOperations
    }
    class PanamaAllocator {
        <<placeholder>>
        ArenaAllocator
        NativeUnsafeAllocatorArena
        or a custom one
    }
    NativeMemoryDomain o-- NativeMemoryAccess
    NativeMemoryDomain o-- NativeMemoryOperations
```

Ownership rule: **closing a domain closes its allocator**.
The array and `ByteBuffer` domains own nothing (GC-managed), so their `close()` is a no-op and the seven array domains are shared stateless singletons.
A native domain is always built *of* an allocator you name (`MemoryDomains.of(allocator)`); whether its `close()` frees memory depends on that allocator (borrowed arena: no; adopted or new arena: yes).

---

## 4. Element access and bulk operations per backend

Each backing type has one stateless singleton `MemoryAccess` (typed scalar reads/writes) and one `MemoryOperations` (bulk copy, fill, and the `MemorySegment` bridge), all obtained via `instance()`.
There are 19 such classes, so this section splits them into three small diagrams and a table.

### 4.1 Which classes exist, per backing type

| Backing `B` | `MemoryAccess` | `MemoryOperations` | Granularity | How elements are reached |
|---|---|---|---|---|
| `boolean[]` | `BooleansMemoryAccess` | `BooleansMemoryOperations` | 1 | `Unsafe`, a boolean read as byte 0/1; byte width only |
| `byte[]` | `BytesMemoryAccess` | `BytesMemoryOperations` | 1 | `Unsafe`, every width, unaligned |
| `short[]` | `ShortsMemoryAccess` | `ShortsMemoryOperations` | 2 | array index; short width only |
| `int[]` | `IntsMemoryAccess` | `IntsMemoryOperations` | 4 | array index; int (and float bits) only |
| `long[]` | `LongsMemoryAccess` | `LongsMemoryOperations` | 8 | array index; long (and double bits) only |
| `float[]` | `FloatsMemoryAccess` | `FloatsMemoryOperations` | 4 | array index; float (and int bits) only |
| `double[]` | `DoublesMemoryAccess` | `DoublesMemoryOperations` | 8 | array index; double (and long bits) only |
| `ByteBuffer` | `ByteBufferMemoryAccess` | `ByteBufferMemoryOperations` | 1 | absolute `get`/`put`, in the buffer's own byte order |
| `MemorySegment` | `NativeMemoryAccess` | `NativeMemoryOperations` | 1 | `ValueLayout.JAVA_*_UNALIGNED`; `copy*Native` is plain `MemorySegment.copy` |
| `Void` (raw address) | `UnsafeMemoryAccess` | none | 1 | `Unsafe` on absolute addresses; only used by its test |

### 4.2 `MemoryAccess`: one implementation per backend family

```mermaid
classDiagram
    direction LR

    class MemoryAccess~B~ {
        <<interface>>
        readByte / readShort / readInt / readLong
        readFloat / readDouble
        writeByte / writeShort / writeInt / writeLong
        writeFloat / writeDouble
    }

    class ArrayAccesses {
        <<7 classes, see 4.4>>
        Booleans ... Doubles MemoryAccess
    }
    class ByteBufferMemoryAccess {
        absolute get/put
    }
    class NativeMemoryAccess {
        ValueLayout unaligned
    }
    class UnsafeMemoryAccess {
        Memory~Void~, raw address
    }
    class UnsafeAccess {
        <<utility>>
        +get()$ Unsafe
    }

    MemoryAccess~B~ <|.. ArrayAccesses
    MemoryAccess~B~ <|.. ByteBufferMemoryAccess
    MemoryAccess~B~ <|.. NativeMemoryAccess
    MemoryAccess~B~ <|.. UnsafeMemoryAccess
    ArrayAccesses ..> UnsafeAccess : Booleans, Bytes
    UnsafeMemoryAccess ..> UnsafeAccess
```

### 4.3 `MemoryOperations`: one implementation per backend family

```mermaid
classDiagram
    direction LR

    class MemoryOperations~B~ {
        <<interface>>
        copy(src, srcOff, dst, dstOff, size)
        copyFromNative(segment to B)
        copyToNative(B to segment)
        fillByte / fillShort / fillInt / fillLong
        fillFloat / fillDouble
    }

    class ArrayOperations {
        <<7 classes, see 4.4>>
        Booleans ... Doubles MemoryOperations
        System.arraycopy, Arrays.fill
        MemorySegment.ofArray for copy*Native
    }
    class ByteBufferMemoryOperations {
        copy: absolute bulk put
        copy*Native, fills: per-byte / per-value loops
    }
    class NativeMemoryOperations {
        MemorySegment.copy / fill
    }

    MemoryOperations~B~ <|.. ArrayOperations
    MemoryOperations~B~ <|.. ByteBufferMemoryOperations
    MemoryOperations~B~ <|.. NativeMemoryOperations
```

### 4.4 The array backends

The seven array backends repeat the same pattern, one access and one operations class per primitive array type.
`DomainFactory` combines each pair with an `ArrayMemoryAllocator` into one shared `ArrayMemoryDomain`.

Element access, one class per array type:

```mermaid
classDiagram
    direction LR

    class ArrayMemoryDomain~B~ {
        <<record>>
    }
    class MemoryAccess~B~ {
        <<interface>>
    }

    ArrayMemoryDomain~B~ o-- MemoryAccess~B~ : directAccess

    MemoryAccess~B~ <|.. BooleansMemoryAccess
    MemoryAccess~B~ <|.. BytesMemoryAccess
    MemoryAccess~B~ <|.. ShortsMemoryAccess
    MemoryAccess~B~ <|.. IntsMemoryAccess
    MemoryAccess~B~ <|.. LongsMemoryAccess
    MemoryAccess~B~ <|.. FloatsMemoryAccess
    MemoryAccess~B~ <|.. DoublesMemoryAccess
```

Bulk operations, one class per array type:

```mermaid
classDiagram
    direction LR

    class ArrayMemoryDomain~B~ {
        <<record>>
    }
    class MemoryOperations~B~ {
        <<interface>>
    }

    ArrayMemoryDomain~B~ o-- MemoryOperations~B~ : memoryOperations

    MemoryOperations~B~ <|.. BooleansMemoryOperations
    MemoryOperations~B~ <|.. BytesMemoryOperations
    MemoryOperations~B~ <|.. ShortsMemoryOperations
    MemoryOperations~B~ <|.. IntsMemoryOperations
    MemoryOperations~B~ <|.. LongsMemoryOperations
    MemoryOperations~B~ <|.. FloatsMemoryOperations
    MemoryOperations~B~ <|.. DoublesMemoryOperations
```

Notes on the array backends:

- The non-byte array accesses (`Shorts`, `Ints`, `Longs`, `Floats`, `Doubles`) only serve their element width, at element-aligned byte offsets.
  The same-width accessor reinterprets bits (`readInt` on `float[]` returns `floatToRawIntBits`); any other width throws `UnsupportedOperationException`.
- `BytesMemoryAccess` is byte-addressable and reads every width, unaligned, through `Unsafe`.
- `copyFromNative` / `copyToNative` go through `MemorySegment.ofArray(array)` and `MemorySegment.copy`, so arrays, buffers and native segments interoperate without extra code.
- Fills of another width replicate the bit pattern into each element (`fillByte` on `float[]` fills with `0x01010101 * b`); offset and size must stay element-aligned.

---

## 5. Views, strided copy and printing

`MemoryView` is immutable and cheap: every transform delegates the layout algebra to `ViewTransforms` (jota-core) and returns a new view over the same `Memory`, adjusting `byteOffset`.
The constructor validates dtype support and bounds.

```mermaid
classDiagram
    direction LR

    class View {
        <<interface, jota-core>>
        +shape() Shape
        +stride() Stride
    }

    class MemoryView~B~ {
        <<interface>>
        +layout() Layout
        +dataType() DataType
        +memory() Memory~B~
        +byteOffset() long
        +byteStride() Stride
        +isBroadcasted() boolean
        +isSpanContiguous() boolean
        +isRowMajorContiguous() boolean
        +isNonOverlapping() boolean
        +view(Shape) MemoryView~B~
        +viewCuTe(Shape) MemoryView~B~
        +permute(int...) MemoryView~B~
        +transpose(axis0, axis1) MemoryView~B~
        +expand(Shape) MemoryView~B~
        +broadcast(Shape) MemoryView~B~
        +slice(axis, from, to, step) MemoryView~B~
        +withLayout(Layout) / withStride(Stride)
        +toString(MemoryAccess, ViewPrintOptions) String
        +of(memory, byteOffset, dtype, layout)$
        +rowMajor(memory, dtype, shape)$
        +isWithinBounds(layout, dtype, memory, off)$ boolean
    }

    class MemoryViewImpl~T~ {
        <<final, internal>>
        -Layout layout
        -DataType dataType
        -Stride byteStride
        -Memory~T~ memory
        -long byteOffset
        +create(...)$
        -transformed(ViewTransforms.Result)
        -spanStartByteOffset() long
    }

    class ViewTransforms {
        <<jota-core>>
        +reshape(layout, shape)$ Optional~Result~
        +permute(layout, perm)$ Result
        +expand(layout, shape)$ Result
        +slice(layout, axis, from, to, step)$ Result
        +broadcast(layout, shape)$ Result
    }

    class MemoryViewFactory {
        <<internal bridge>>
        +of(dtype, memory, off, layout)$
        +rowMajor(dtype, memory, shape)$
        +allocate(allocator, dtype, shape)$
    }

    class StridedCopy {
        <<package-private>>
        -HOST_ACCESS$ MemoryAccess~MemorySegment~
        -HOST_OPS$ MemoryOperations~MemorySegment~
        ~copy(domain, src, dst)$
        -copyElements(access, ops, src, dst)$
        -copyViaHost(domain, src, dst)$
        -stage(arena, ops, view, download)$
        ~byteSpan(view)$ long[]
        ~releaseTemp(memory)$
    }

    class MemoryViewPrinter {
        +toString(view)$ String
        +toString(view, access)$ String
        +toString(view, access, options)$ String
    }
    class ViewPrintOptions {
        <<record>>
        int maxElements
        int edgeItems
        boolean includeValues
        boolean includeMetadata
        ValueFormatter formatter
        +compact()$
        +metaOnly()$
        +valuesOnly()$
    }
    class ValueFormatter {
        <<functional interface>>
        +format(DataType, Object) String
    }
    class DefaultValueFormatter {
        <<private>>
    }

    View <|-- MemoryView~B~
    MemoryView~B~ <|.. MemoryViewImpl~T~
    MemoryView~B~ ..> MemoryViewFactory : of / rowMajor
    MemoryViewFactory ..> MemoryViewImpl~T~ : create
    MemoryViewImpl~T~ ..> ViewTransforms : layout algebra
    MemoryDomain~B~ ..> StridedCopy : default copy(src, dst)
    StridedCopy ..> MemoryAccess~B~ : per element (odometer)
    StridedCopy ..> MemoryOperations~B~ : contiguous memcpy, host staging
    MemoryView~B~ ..> MemoryViewPrinter : toString(access)
    MemoryViewPrinter ..> ViewPrintOptions
    ViewPrintOptions --> ValueFormatter
    ValueFormatter <|.. DefaultValueFormatter
```

Contiguity vocabulary used throughout:

- **row-major contiguous** — dense C order; the only layout safe to copy as one block (`MemoryOperations.copy`).
- **span-contiguous** — gapless byte range in *some* order (a transposed square matrix).
  `viewCuTe` may reshape such a view (CuTe semantics) even when `view` would refuse.
- **non-overlapping** — no zero (broadcast) strides; two indices never alias one element.

`StridedCopy.copy` (used by `MemoryDomain.copy(src, dst)`):

1. both views row-major contiguous → a single `ops.copy`;
2. no `directAccess` (opaque domain) → `copyViaHost`: download each view's byte span into a confined `Arena`, copy element-wise on the host, upload the destination span (reading it back first if it has gaps);
3. same `Memory` on both sides → through a contiguous temporary (two passes) to survive aliasing;
4. otherwise → `copyElements`: an odometer over the flat dims with running byte offsets; 1/2/4/8-byte dtypes use typed access, any other width (block-quantized types) one `ops.copy` per element, so no dtype is ever special-cased.

---

## 6. Entry points and internal factories

The public constructors follow a naming convention that states ownership:

- `of(x)` — wraps `x`, inherits its lifetime, allocates nothing;
- `adopt(x)` — takes `x` over: closing the result closes `x`;
- `newX()` — creates something the caller owns (and closes, if `AutoCloseable`);
- bare noun (`floats()`) — a shared stateless instance (array backends only).

Each public entry point delegates to a `public` bridge in `internal`, so implementation classes stay package-private.

```mermaid
classDiagram
    direction LR

    namespace api {
        class Memories {
            <<final>>
            +of(boolean[])$ ... of(double[])$
            +of(ByteBuffer)$
            +of(MemorySegment)$
        }
        class MemoryAllocators {
            <<final>>
            +ofArena(Arena)$ MemoryArena
            +adoptArena(Arena)$ MemoryArena
            +newScopedArena()$ ScopedArena
            +newByteBuffer(direct)$ MemoryAllocator
        }
        class MemoryDomains {
            <<final>>
            +of(MemoryAllocator~MemorySegment~)$
            +ofByteBuffer(MemoryAllocator~ByteBuffer~)$
            +booleans()$ bytes()$ shorts()$ ints()$
            +longs()$ floats()$ doubles()$
        }
        class MemoryViews {
            <<final>>
            +of(memory, off, dtype, layout)$
            +rowMajor(memory, dtype, shape)$
            +allocate(allocator, dtype, shape)$
            +full(domain, dtype, count, Number)$
            +zeros(domain, dtype, shape)$
            +ones(domain, dtype, shape)$
            +arange(domain, dtype, end)$
        }
    }

    namespace internal_bridges {
        class MemoryFactory {
            +ofBooleans()$ ... ofDoubles()$
            +ofByteBuffer()$
            +ofMemorySegment()$
        }
        class MemoryAllocatorFactory {
            -BOOLEANS$ ... DOUBLES$ : ArrayMemoryAllocator
            +ofBooleans()$ ... ofDoubles()$
            +ofByteBuffer(direct)$
            +ofArena(Arena)$
            +adoptArena(Arena)$
            +newScopedArena()$
        }
        class DomainFactory {
            +ofBooleans()$ ... ofDoubles()$
            +ofMemorySegment(allocator)$
            +ofByteBuffer(allocator)$
        }
        class MemoryViewFactory {
            +of(...)$
            +rowMajor(...)$
            +allocate(...)$
        }
        class NativeMemoryFactory {
            +createDomain(allocator)$
            +newScopedArena()$
            +memoryAccess()$
            +memoryOperations()$
            +memory(MemorySegment)$
        }
    }

    Memories ..> MemoryFactory
    MemoryAllocators ..> MemoryAllocatorFactory
    MemoryDomains ..> DomainFactory
    MemoryViews ..> MemoryViewFactory : of / rowMajor / allocate
    MemoryViews ..> MemoryDomain : full / zeros / ones / arange use allocator + fill / copyFromNative

    MemoryFactory ..> ArrayMemory : arrays
    MemoryFactory ..> ByteBufferMemory
    MemoryFactory ..> NativeMemoryFactory : segments
    MemoryAllocatorFactory ..> ArrayMemoryAllocator
    MemoryAllocatorFactory ..> ByteBufferAllocator
    MemoryAllocatorFactory ..> ArenaAllocator : of / adopt
    MemoryAllocatorFactory ..> NativeMemoryFactory : newScopedArena
    NativeMemoryFactory ..> NativeUnsafeAllocatorArena
    NativeMemoryFactory ..> NativeMemorySegmentMemory
    NativeMemoryFactory ..> NativeMemoryDomain
    DomainFactory ..> ArrayMemoryDomain
    DomainFactory ..> ByteBufferDomain
    DomainFactory ..> NativeMemoryFactory
    MemoryViewFactory ..> MemoryViewImpl
```

`MemoryViews.full/zeros/ones` allocate from the domain's allocator and fill with the domain's `MemoryOperations` (works on opaque domains).
`arange` builds the sequence in a host segment and transfers it with one `copyFromNative`, for the same reason.

---

## 7. Checks and low-level utilities

```mermaid
classDiagram
    direction LR

    class MemoryAccessChecks {
        <<final>>
        -MODE$ Mode
        +mode()$ Mode
        +checkBounds(memory, byteOffset, byteSize)$
        +checkWriteable(memory)$
        +checkAlignedValue(value, alignment)$
        +checkBounds(condition, message)$
        +checkReadOnly(condition, message)$
        +checkAlignment(condition, message)$
        ~resolveMode(String)$ Mode
    }
    class Mode {
        <<enumeration>>
        OFF
        ASSERT
        RUNTIME
    }
    MemoryAccessChecks *-- Mode : nested enum

    class UnsafeAccess {
        <<internal>>
        +get()$ Unsafe
    }
    class Util {
        <<internal>>
        +isPowerOf2(long)$ boolean
    }

    class AccessImpls {
        <<placeholder>>
        every MemoryAccess impl
        except UnsafeMemoryAccess
    }
    class OperationsImpls {
        <<placeholder>>
        every MemoryOperations impl
    }
    class NativeAndBufferAllocators {
        <<placeholder>>
        NativeUnsafeAllocator
        ByteBufferAllocator
    }

    AccessImpls ..> MemoryAccessChecks : bounds, alignment, read-only
    OperationsImpls ..> MemoryAccessChecks : bounds, alignment, read-only
    AccessImpls ..> UnsafeAccess : Booleans / Bytes access
    NativeAndBufferAllocators ..> Util : alignment must be a power of 2
```

- `-Djota.memory.checks=off|assert|runtime` (default `runtime`) selects how a failed check fails: skipped, `assert`, or an exception.
- `Memory.isReadOnly()` is **advisory**: the implementations check it before writing, but `base()` hands out the raw storage and the JVM enforces nothing.
  Only `ByteBuffer`- and `MemorySegment`-backed memory can be read-only (`asReadOnly()`); arrays never are.

---

## 8. How a copy is routed

Not a class diagram, but it ties the pieces together: the static `MemoryDomain.copy(srcDomain, src, dstDomain, dst)` chooses between the same-backend path (§5) and the cross-backend path through `MemoryOperations.copy` and `MemorySegment` staging (§1).

```mermaid
flowchart TD
    A["MemoryDomain.copy(srcDomain, src, dstDomain, dst)"] --> B{same dtype and shape?}
    B -- no --> X[IllegalArgumentException]
    B -- yes --> C{same device and same MemoryOperations?}
    C -- yes --> S["domain.copy(src, dst)<br/>StridedCopy, or a GPU on-device override (§9)"]
    C -- no --> D{both views row-major contiguous?}
    D -- yes --> M["MemoryOperations.copy(srcOps, ..., dstOps, ...)"]
    D -- no --> E["make src contiguous on src domain (StridedCopy)"]
    E --> F["allocate contiguous tmp on dst domain"]
    F --> M2["MemoryOperations.copy(contig → tmp)"]
    M2 --> G["dstDomain.copy(tmp, dst)<br/>StridedCopy or GPU override"]
    G --> R["releaseTemp(both temps)"]

    M --> N{a side backed by MemorySegment?}
    N -- dst is segment --> N1["srcOps.copyToNative(src → dst)"]
    N -- src is segment --> N2["dstOps.copyFromNative(src → dst)"]
    N -- neither --> N3["confined Arena staging buffer (4 KiB-aligned chunks):<br/>srcOps.copyToNative → buffer → dstOps.copyFromNative"]
```

---

## 9. How jota-tensor and the backends use the memory layer

The IR never names a `MemoryOperations` call.
Arithmetic (`BinaryOp`, `ReductionOp`, …) is lowered to LIR and becomes kernel code; the memory layer only appears in three places:

- **binding**: kernel buffers are `MemoryView`s, and LIR offsets are computed from their byte strides;
- **element access**: LIR loads and stores turn into `MemoryAccess.read*` / `write*` on the Java paths;
- **data movement at the edges**: creating tensors, constants, and moving data between devices, through `MemoryDomain.copy` and `MemoryOperations`.

### 9.1 From tensor operations to memory calls

```mermaid
flowchart TD
    subgraph tensor["jota-tensor"]
        T["Tensor ops (eager or traced)"] --> TIR["TIRGraph"]
        TIR --> LOW["TIRToLIRLowerer<br/>offset = Σ index × buf.byteStrides()"]
        LOW --> LIR["LIRGraph: BufferRef, SLoad, Store, StructuredFor"]
    end

    subgraph memory["jota-memory"]
        MV["MemoryView<br/>memory + byteOffset + layout"]
        MA["MemoryAccess.read* / write*"]
        DC["MemoryDomain.copy (static)"]
        MO["MemoryOperations<br/>copy / copyFromNative / copyToNative / fill*"]
        VT["ViewTransforms (jota-core)<br/>new layout, same Memory"]
    end

    TIR -- "ViewTransform: transpose, reshape, broadcast" --> VT
    TIR -- "TensorInput" --> MV
    LIR -- "BufferRef bound to" --> MV

    LIR --> INT["LIRInterpreter"]
    LIR --> PAN["Panama LIRKernelCompiler<br/>generated Java"]
    LIR --> NAT["C / CUDA / HIP / Metal / OpenCL / Mojo<br/>generated native source"]

    INT -- "SLoad / Store" --> MA
    PAN -- "access.readFloat(mem, addr) ..." --> MA
    NAT -- "KernelArgs.addBuffer(view):<br/>base pointer + byteOffset" --> MV

    EDGE["Constants, random, Tensor.of(array), arange,<br/>device transfers, kernel inputs and outputs"] --> DC
    EDGE --> MO
    DC --> MO
```

### 9.2 LIR nodes and the memory layer

| LIR node | Meaning | Memory layer |
|---|---|---|
| `BufferRef` | a kernel input or output | bound to a `MemoryView` (`LIRInterpreter.Bindings`, or `KernelArgs.addBuffer(view)` for compiled kernels) |
| `SLoad(buffer, offset, dtype)` | read one scalar at `view.byteOffset() + offset` | `MemoryAccess.read*` |
| `Store(buffer, offset, value)` | write one scalar | `MemoryAccess.write*` |

`TIRToLIRLowerer` builds each offset expression from the loop indices and `buf.byteStrides()` (`layout.stride() × dtype.byteSize()`), the same formula as `MemoryView.byteStride()`.
Strided, broadcast and sliced inputs are therefore read in place, without a copy.

How each execution path performs `SLoad` / `Store`:

- `LIRInterpreter` (`jota-tensor`) calls `readScalar` / `writeScalar`, which pick the width from the dtype: `FP32` → `readFloat`, `FP64` → `readDouble`, `FP16` / `BF16` → `readShort` plus a conversion, `I8` / `BOOL` → `readByte`, the other integers → `readShort` / `readInt` / `readLong`.
- The Panama backend's `LIRKernelCompiler` generates Java source that fetches `memoryDomain.directAccess()` once and emits `access.readFloat(mem, addr)` and friends, so compiled Java kernels run on `NativeMemoryAccess`.
- The C and GPU backends generate C, CUDA, HIP, Metal, OpenCL or Mojo source with raw pointer arithmetic.
  `MemoryAccess` is not used inside those kernels; the host only passes `memory.base()` and `byteOffset` through `KernelArgs`.

### 9.3 TIR nodes and tensor operations

| TIR node or tensor operation | Memory layer used |
|---|---|
| `TensorInput` | an existing `MemoryView`, bound as is |
| `ViewTransform` (transpose, reshape, unsqueeze, broadcast, …) | `ViewTransforms` computes a new layout and an element offset delta: a new `MemoryView` over the same `Memory`, no copy (the same algebra as `MemoryView.permute` / `expand` / `view`) |
| `ViewTransform` that is not expressible as a view, `Contiguous` | `TIRInterpreter` copies element by element into a row-major temporary from a `MemoryAllocators.newScopedArena()` (`TIREvalContext`), with its own loop rather than `StridedCopy` |
| `ScalarConstant` (`ConstantComputation`) | allocate one element and write it with `MemoryAccess.write*`; on an opaque domain, write it on the host and upload it with `MemoryDomain.copy`; then broadcast it with zero strides |
| `IotaConstant` / range (`RangeComputation`) | `MemoryViews.arange`: the sequence is built on the host, then one `copyFromNative` |
| `RandomUniformOp` (`RandomComputation`), `Tensor.of(float[])` and friends (`TensorFactory`) | data in a host segment, then static `MemoryOperations.copy(HOST ops → device ops)`, which reduces to `copyFromNative` |
| device transfer (`ScheduledExecutor.ensureDeviceView`, `TensorSupport.transferToDevice`) | allocate a row-major destination on the target domain, then static `MemoryDomain.copy(srcDomain, src, dstDomain, dst)` |
| scalar tensor passed as a kernel argument (`LIRKernelArgsBuilder.readScalarBits`) | if its memory is not a `MemorySegment`, `MemoryDomain.copy` it to the native host domain, then read it |
| kernel inputs and outputs in each `*ComputeEngine` | static `MemoryDomain.copy` between the host and device domains |

Nothing in `jota-tensor` or the backends calls `fill*` directly; the only caller is `MemoryViews.full` / `zeros` / `ones`.
Constants avoid fills by allocating one element and broadcasting it with zero strides.

### 9.4 Backend memory domains

Every backend plugs into the memory layer with its own `MemoryDomain`.

```mermaid
classDiagram
    direction LR

    class MemoryDomain~B~ {
        <<interface>>
        +directAccess() MemoryAccess~B~
        +memoryOperations() MemoryOperations~B~
        +copy(src, dst)
    }

    class NativeMemoryDomain {
        <<jota-memory, Panama>>
        directAccess() NativeMemoryAccess
        copy() StridedCopy
    }
    class CMemoryDomain {
        <<jota-backend-c>>
        reuses the host segment access and operations
    }
    class GpuMemoryDomain {
        <<placeholder>>
        CudaMemoryDomain
        HipMemoryDomain
        MetalMemoryDomain
        OpenClMemoryDomain
        directAccess() null
        copy() memcpy or on-device kernel
    }
    class GpuMemoryOperations {
        <<placeholder>>
        Cuda / Hip / Metal / OpenCl MemoryOperations
        copy() memcpyDtoD
        copyFromNative() memcpyHtoD
        copyToNative() memcpyDtoH
        fill*() memsetD8/16/32/64 (CUDA)
    }
    class GpuStridedCopy {
        <<placeholder>>
        CudaStridedCopy, HipStridedCopy
        MetalStridedCopy, OpenClStridedCopy
    }
    class MojoMemoryDomain~T~ {
        <<jota-backend-mojo>>
        wraps another domain
        unwraps MojoMemory, then delegates
    }

    MemoryDomain~B~ <|.. NativeMemoryDomain
    MemoryDomain~B~ <|.. CMemoryDomain
    MemoryDomain~B~ <|.. GpuMemoryDomain
    MemoryDomain~B~ <|.. MojoMemoryDomain~T~
    GpuMemoryDomain *-- GpuMemoryOperations
    GpuMemoryDomain ..> GpuStridedCopy : non-contiguous copy
    CMemoryDomain ..> NativeMemoryDomain : HOST access and ops
    MojoMemoryDomain~T~ o-- MemoryDomain~B~ : delegate
```

On a GPU domain, `directAccess()` returns `null`, so everything generic in `jota-memory` falls back to `copyToNative` / `copyFromNative`.
A heap `MemorySegment` given to `copyFromNative` is first copied into a native staging segment, because the driver needs an address.

Each GPU domain overrides `MemoryDomain.copy(src, dst)`: a contiguous pair is one `memcpy`, anything else runs an on-device strided-copy kernel.
This is what keeps the rule from `jota/CLAUDE.md` ("every backend must handle strided operands on-device"): without the override, the default `StridedCopy` would download the views' byte spans to the host (`copyViaHost`).
