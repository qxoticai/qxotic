# jota-core — High-level summary

`jota-core` is the smallest module of jota: the dependency-free **value model** that every other module (`jota-memory`, `jota-tensor`, the backends) builds on.
It holds no memory and runs no kernels; it only describes *what* a tensor's storage looks like and *how* a coordinate becomes a byte offset.

It is one JPMS module, `com.qxotic.jota`, which exports a single package, `com.qxotic.jota`.
The `com.qxotic.jota.impl` package holds the implementations and is **not exported**.

| Concept | Type(s) | One-line answer |
|---|---|---|
| Element type | `DataType`, `BFloat16` | *What is one storage unit?* (bytes per unit, elements per unit, float/int) |
| Byte order | (`DataType.layout()` → `ValueLayout`) | *How are the bytes of a multi-byte value ordered?* — the platform's native order |
| Shape | `Shape` | *How many units along each axis?* (possibly nested) |
| Stride | `Stride` | *How far to jump in storage when an index grows by 1?* |
| Layout | `Layout` | Shape + Stride = a function `coordinate → offset` (with a CuTe-style algebra) |
| Quantization | `DataType.Q4_0`, `Q8_0`, `Q4_K`, … | Block formats: one storage unit = one block of N logical elements |
| View | `View`, `ViewTransforms`, `Indexing` | Layout + DataType + base byte offset = a typed window on storage |
| Device | `DeviceType`, `Device` | *Where does it run?* (`java`, `panama`, `c`, `cuda`, `hip`, `metal`, `opencl`, `mojo`) |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the class diagrams.

---

## 1. DataType — what one storage unit is

`DataType` is an interface with predefined constants.
Each one is backed by a `java.lang.foreign.MemoryLayout` (so its size, name, and alignment come from the Panama FFM API) and answers:

- `byteSize()` — bytes per **addressable storage unit**;
- `elementsPerBlock()` — **logical elements** per storage unit (1 for dense types);
- `isFloatingPoint()`, `isIntegral()`, `name()`, `aliases()`.

### Dense types (`elementsPerBlock() == 1`)

| Constant | Bytes | Kind | Aliases | Java carrier |
|---|---:|---|---|---|
| `BOOL` | 1 | neither | `boolean` | `byte` |
| `I8` | 1 | integral | `int8`, `byte` | `byte` |
| `I16` | 2 | integral | `int16`, `short` | `short` |
| `I32` | 4 | integral | `int32`, `int` | `int` |
| `I64` | 8 | integral | `int64`, `long` | `long` |
| `FP16` | 2 | floating | `float16` | `short` (raw bits) |
| `BF16` | 2 | floating | `bfloat16` | `short` (raw bits) |
| `FP32` | 4 | floating | `float32`, `float` | `float` |
| `FP64` | 8 | floating | `float64`, `double` | `double` |

Java has no half-precision primitive, so `FP16` and `BF16` values travel as `short` bit patterns.
`Float.float16ToFloat` / `floatToFloat16` (JDK) cover FP16; the `BFloat16` helper class covers BF16:

- `BFloat16.fromFloat(float)` — round to nearest, ties to even, NaN stays NaN;
- `BFloat16.toFloat(short)` — exact widening (put the 16 bits on top of a float);
- `BFloat16.fromFloat16 / toFloat16` — direct FP16 ⇄ BF16, keeping Inf and NaN.

### Little / Big Endian

jota-core has **no explicit byte-order setting** and never swaps bytes.
Every dense `DataType` is built from a `ValueLayout.JAVA_*_UNALIGNED` constant (`JAVA_BYTE` for 1-byte types), and those layouts use `ByteOrder.nativeOrder()`.
So a multi-byte value is read and written in the **byte order of the machine** — little-endian on every platform jota targets (x86-64, AArch64, and the GPUs).

Consequences:

- Data in memory and in files that jota maps directly (for example GGUF weights, which are little-endian) can be used as-is with no conversion.
- The `_UNALIGNED` variants allow a value to start at any byte address.
  This matters because block formats have odd sizes (Q8_0 = 34 bytes), so the next block's FP16 scale may not be 2-byte aligned.
- Block-quantized types are plain `byte` sequences (`MemoryLayout.sequenceLayout(n, JAVA_BYTE)`): their inner encoding (including the byte order of their scales) belongs to the kernels that decode them, not to jota-core.
- Endianness is a property of **bytes within one value**.
  It never affects shapes, strides or offsets, which all count whole storage units.

---

## 2. Shape — how many units per axis

A `Shape` is an immutable tuple of non-negative dimension sizes.
It may be **nested** (as in NVIDIA CuTe):

```java
Shape s = Shape.of(2, Shape.of(3, 4), 5);   // printed "(2, (3, 4), 5)"
s.rank();      // 3  — top-level modes: 2, (3,4), 5
s.flatRank();  // 4  — leaf dims: 2, 3, 4, 5
s.size();      // 120 — product of all leaves
s.size(1);     // 12  — size of mode 1
```

Words used throughout jota-core:

- **mode** — one top-level entry (a number or a nested tuple).
  `modeAt(i)` returns it.
- **flat** — the leaves in left-to-right order.
  `flatAt(i)`, `toArray()`, `flatten()`.
- **congruent** — same nesting profile, whatever the numbers.
  A `Shape` and its `Stride` must be congruent.
- **scalar** — rank 0, `()`, one element.

Indexes wrap around like Python: `-1` is the last axis (`Util.wrapAround`).

Ways to build a shape: `Shape.flat(2, 3)`, `Shape.of(2, Shape.of(3, 4))` (or `shape(...)` with a static import), `Shape.pattern("(batch, (n, m))", 2, 3, 4)` (bracket notation, names are ignored, only the structure counts), `Shape.template(otherTuple, dims...)` (copy another tuple's nesting), and `Shape.resolveShape(total, 4, -1)` (infer one `-1` dim like `reshape(-1)` in NumPy).

Structural editing returns new shapes: `replace`, `insert`, `remove`, `permute`, `flattenModes`.

Internally (`NestedTupleImpl`), a tuple is a `long[] flat` plus an `int[] nest` (null when flat): each leaf records how many brackets open before it (positive) or close after it (negative).
`(2, (3, 4), 5)` is `flat = [2, 3, 4, 5]`, `nest = [0, +1, -1, 0]`.

---

## 3. Stride — how far to jump

A `Stride` has exactly the same tuple structure as a `Shape` (both implement `NestedTuple<T>`), but its leaves are **step sizes in storage units** (elements for dense types, whole blocks for quantized types).
A coordinate `c` maps to offset `Σ c[i] × stride[i]`.

```java
Stride.rowMajor(Shape.flat(2, 3, 4));     // (12, 4, 1)  — C order, last axis fastest
Stride.columnMajor(Shape.flat(2, 3, 4));  // (1, 2, 6)   — Fortran order, first axis fastest
Stride.zeros(3);                          // (0, 0, 0)   — every coordinate maps to offset 0
s.scale(4);                               // multiply every step, e.g. element strides → byte strides
```

- A **zero stride** broadcasts an axis: all its indices read the same storage (`isAllZeros()` for a broadcast scalar).
- A **negative stride** walks backwards (reverse slicing).
- Strides may be nested to follow the shape: `Stride.rowMajor` keeps the shape's nesting.

---

## 4. Layout — Shape + Stride as a function

`Layout` = a congruent `(Shape, Stride)` pair, printed `shape:stride`, e.g. `(2, 3):(3, 1)`.
It is the affine map `coordinate → offset` (in storage units, relative to some base).

```java
Layout a = Layout.rowMajor(2, 3);              // (2, 3):(3, 1)
Layout t = Layout.of(Shape.flat(3, 2), Stride.flat(1, 3));  // its transpose: (3, 2):(1, 3)
```

### Properties (questions you can ask a layout)

| Method | Meaning | `a` | `t` |
|---|---|:-:|:-:|
| `isRowMajorContiguous()` | packed, C order | yes | no |
| `isSpanContiguous()` | the offsets fill a range with no gaps (any order) | yes | yes |
| `isNonOverlapping()` / `isInjective()` | no two coordinates share an offset (no broadcast) | yes | yes |
| `isBijective()` | injective and gapless | yes | yes |
| `cosize()` | `maxOffset − minOffset + 1` (size of storage spanned) | 6 | 6 |
| `isLocalContiguous(mode)` / `isSuffixContiguous(mode)` | contiguity of one mode / of a mode and everything after it | | |

### Layout algebra (CuTe-style)

`LayoutImpl` implements the main operations of NVIDIA CuTe's layout algebra, restricted to results that are still a plain `(shape, stride)`:

- `flatten()` — drop nesting.
- `coalesce()` — merge neighbouring modes when `stride[i] == dim[i+1] × stride[i+1]`, and drop size-1 modes.
  `(2, 3, 4):(12, 4, 1)` → `(24):(1)`.
  `coalesce(i)` / `canCoalesce(i)` work on one pair of modes.
- `compose(inner)` — `(this ∘ inner)(c) = this(inner(c))`: re-index a layout through another one.
- `inverse()` — the inverse map, for a bijective layout whose offsets fill `[0, size−1]`.
- `complement(n)` — the layout of the *gaps*: the offsets in `[0, n)` that this layout does not reach.
  `(4):(2)` with `n = 24` → `(2, 3):(1, 8)`.
- `logicalDivide(tiler)` = `compose((tiler, tiler.complement(size)))` — split the domain into *(inside one tile, which tile)*.
  `tile(n)` is `logicalDivide(rowMajor(n))`: `(8):(1).tile(4)` gives shape `(4, 2)` with strides `(1, 4)`.
- `logicalProduct(other)` = `(this, complement(...).compose(other))` — repeat this layout following `other`.

These are the building blocks for tiling a tensor across threads, warps or SIMD lanes in the backends.

---

## 5. Quantization — block data types

Block-quantized types (the GGML/GGUF formats plus some newer ones) store **N logical elements in one block of B bytes**.
Inside the block, scales and packed low-bit values are mixed; jota-core treats the block as **one opaque, atomic storage unit**.

| Constant | Elements / block | Bytes / block | Bits / element |
|---|---:|---:|---:|
| `Q4_0` | 32 | 18 | 4.5 |
| `Q4_1` | 32 | 20 | 5.0 |
| `Q5_0` | 32 | 22 | 5.5 |
| `Q5_1` | 32 | 24 | 6.0 |
| `Q8_0` | 32 | 34 | 8.5 |
| `Q4_K` | 256 | 144 | 4.5 |
| `Q5_K` | 256 | 176 | 5.5 |
| `Q6_K` | 256 | 210 | 6.5625 |
| `MXFP4` | 32 | 17 | 4.25 |
| `NVFP4` | 64 | 36 | 4.5 |
| `Q1_0` | 128 | 18 | 1.125 |
| `Q2_0` | 64 | 18 | 2.25 |
| `PQ2_0` | 128 | 34 | 2.125 |
| `PTQ1_0` | 128 | 28 | 1.75 |
| `TQ1_0` | 256 | 54 | 1.6875 |
| `TQ2_0` | 256 | 66 | 2.0625 |

For example, a Q8_0 block is a 2-byte FP16 scale followed by 32 signed bytes: 34 bytes for 32 values.

### Logical vs physical shape

- The **logical** shape counts real elements.
  The **physical** (storage) shape counts storage units.
- They differ **only for block types** and **only on the innermost axis** (the last flat dim), which is the axis that blocks tile, as in GGUF.
- `dtype.physicalShape(logical)` divides the last dim by `elementsPerBlock()` (it must divide exactly); `dtype.logicalShape(physical)` multiplies it back.
  Nesting is kept.
  For dense types both are the identity.
- A block type cannot have a rank-0 shape (no innermost axis to tile): both conversions throw.

```text
Q8_0 weight, logical (4096, 4096)
  physical shape  (4096, 128)            128 blocks per row
  row-major stride (128, 1)              in blocks
  byte stride      (4352, 34)            stride × 34
  total bytes      4096 × 128 × 34 = 17 825 792
```

### Blocks are atomic

Every `Layout` and every `View` uses **physical** units.
All view operations (reshape, slice, transpose, offset arithmetic) therefore move whole blocks and never split one.
The single formula

```text
byteOffset = view.byteOffset() + Σ coord[i] × stride[i] × dataType.byteSize()
```

works the same way for dense and block types.
`View.logicalShape()` / `logicalSize()` exist only for reading back element counts (checking a model config, sizing an output); they never drive addressing.

---

## 6. View, ViewTransforms, Indexing — from coordinates to bytes

### View

`View` is an interface: `layout()` + `dataType()` + `byteOffset()` (the absolute base, in bytes).
It holds no storage.
Default methods give `shape()`, `stride()` (physical, in units), `byteStride()` (in bytes), `logicalShape()` and `logicalSize()`.
In `jota-memory`, `MemoryView<B>` extends `View` and adds the `Memory<B>` the offsets point into.

### ViewTransforms — zero-copy reshaping

Static functions that change a `Layout` without moving data.
Each returns a `Result(layout, elementOffsetDelta)`; the delta (in units) is added to the view's base offset (only `slice` returns a non-zero one).

| Transform | What it does | Example |
|---|---|---|
| `permute(layout, perm...)` | reorder top-level modes (transpose) | `(2, 3):(3, 1)` → `(3, 2):(1, 3)` |
| `slice(layout, axis, from, to, step)` | strided range on one axis; `step < 0` reverses | `(4, 6):(6, 1)`, axis 1, `[2, 6)` step 2 → `(4, 2):(6, 2)`, delta 2 |
| `unsqueeze(layout, axis)` | insert a size-1 axis with stride 0 | `(3):(1)` → `(1, 3):(0, 1)` |
| `expand(layout, shape)` | grow size-1 dims, setting their stride to 0 | `(1, 3):(0, 1)` → `(4, 3):(0, 1)` |
| `broadcast(layout, shape)` | NumPy broadcasting (align from the right) | `(3):(1)` → `(2, 3):(0, 1)` |
| `reshape(layout, shape)` | change the shape **only if it can be done with strides**; `Optional.empty()` means a copy is needed | `(2, 3):(3, 1)` → `(6):(1)`; but `(3, 2):(1, 3)` → `(6)` is empty |

### Indexing

Static conversions, all in **row-major** order:

- `linearToCoord(shape, i)` / `coordToLinear(shape, coord)`;
- `coordToOffset(stride, coord)` (in units) / `coordToOffset(view, coord)` (absolute bytes);
- `linearToOffset(shape, stride, dtype, i)`, `linearToOffset(layout, dtype, i)`, `linearToOffset(view, i)` (bytes).

---

## 7. Device — where it runs

- `DeviceType` — a record around a string id, normalized (trimmed, lowercase) and interned.
  Built-in: `JAVA` (Java heap arrays), `PANAMA` (off-heap `MemorySegment`), `C`, `CUDA`, `HIP`, `METAL`, `OPENCL`, `MOJO`.
  `DeviceType.fromId("Cuda")` returns the same instance as `CUDA`, and new ids can be added at runtime.
- `Device` — a `(DeviceType, index ≥ 0)` record printed as `cuda:0`.
  `DeviceType.CUDA.deviceIndex(1)` creates one; `belongsTo(type)` and `runtimeId()` let backends route work.

jota-core only names devices; allocation and execution live in `jota-memory` and the backend modules.

---

## 8. Putting it together

```text
logical shape (elements) ──DataType.physicalShape──▶ physical shape (units)
                                                         │
                                  Stride.rowMajor / ViewTransforms
                                                         ▼
                         Layout (shape:stride, in units) + DataType + byteOffset  =  View
                                                         │
                                 Indexing.coordToOffset / linearToOffset
                                                         ▼
                                          absolute byte offset in storage
                                (read with the DataType's native-order ValueLayout)
```

## 9. Class list

| Class | Package | Role |
|---|---|---|
| `DataType` (+ package-private `DataTypeImpl`) | `com.qxotic.jota` | element types, dense and block-quantized; logical ⇄ physical shapes |
| `BFloat16` | `com.qxotic.jota` | BF16 ⇄ FP32 / FP16 bit conversions |
| `Device`, `DeviceType` | `com.qxotic.jota` | device identity |
| `Shape` | `com.qxotic.jota` | nested dimension tuple + factories + `resolveShape` |
| `Stride` | `com.qxotic.jota` | nested step tuple + row/column-major factories + `scale` |
| `Layout` | `com.qxotic.jota` | shape:stride map, contiguity checks, layout algebra |
| `View` | `com.qxotic.jota` | layout + dtype + byte offset |
| `ViewTransforms` (+ `Result`) | `com.qxotic.jota` | zero-copy permute / slice / reshape / broadcast / expand / unsqueeze |
| `Indexing` | `com.qxotic.jota` | linear ⇄ coordinate ⇄ offset |
| `Util` | `com.qxotic.jota` | `wrapAround` for negative indexes (internal, public only for compatibility) |
| `NestedTuple<T>` | `com.qxotic.jota.impl` | the common tuple contract of `Shape` and `Stride` (F-bounded generic) |
| `NestedTupleImpl<T>` | `com.qxotic.jota.impl` | `flat[]` + `nest[]` storage, printing, congruence |
| `ShapeImpl`, `StrideImpl` | `com.qxotic.jota.impl` | concrete tuples: nesting edits, `size`, `scale` |
| `LayoutImpl` | `com.qxotic.jota.impl` | coalesce, compose, inverse, complement, divide, product |
| `ShapeFactory`, `StrideFactory`, `LayoutFactory` | `com.qxotic.jota.impl` | public bridges used by the interfaces' static methods |
| `PatternParser` | `com.qxotic.jota.impl` | parses `"(a, (b, c))"` into a `nest[]` array |
