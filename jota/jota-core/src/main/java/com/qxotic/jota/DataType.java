package com.qxotic.jota;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * The element type of a tensor's storage: dense primitives (one addressable unit per logical
 * element) and block-quantized formats (one addressable unit - a block of {@link #byteSize()} bytes
 * - per {@link #elementsPerBlock()} consecutive logical elements; e.g. Q8_0 stores 32 logical
 * elements in one 34-byte block).
 *
 * <p><b>Logical vs physical shapes.</b> A tensor's <i>logical</i> shape counts represented elements
 * per axis; its <i>physical</i> (storage) shape counts addressable units per axis. The two differ
 * only for block-quantized dtypes, and only on the INNERMOST storage axis - the single axis
 * blocking tiles, matching GGUF/GGML, which interleave scales and mantissas along the contiguous
 * dimension. {@link #physicalShape(Shape)} divides the innermost dimension by {@link
 * #elementsPerBlock()} (exact division required); {@link #logicalShape(Shape)} multiplies it back.
 * For nested shapes the conversions scale the last dim in flatten order and preserve all other
 * structure.
 *
 * <p><b>Blocks are atomic.</b> Every view's layout (shape + strides) is in physical units, so all
 * view algebra - reshape, slice, transpose, offset arithmetic - operates on whole blocks and never
 * splits one: {@code byteOffset = Σ index × stride × byteSize()} holds uniformly for dense and
 * block dtypes alike.
 */
public interface DataType {
    /** Bytes per addressable storage unit: one element for dense dtypes, one block for quants. */
    long byteSize();

    /** Logical elements per storage unit: 1 for dense dtypes, the block width for quants. */
    long elementsPerBlock();

    MemoryLayout layout();

    boolean isFloatingPoint();

    boolean isIntegral();

    String name();

    List<String> aliases();

    DataType BOOL =
            new DataTypeImpl(
                    ValueLayout.JAVA_BYTE.withName("bool"), false, false, boolean.class, "boolean");

    DataType I8 =
            new DataTypeImpl(
                    ValueLayout.JAVA_BYTE.withName("i8"), false, true, byte.class, "int8", "byte");
    DataType I16 =
            new DataTypeImpl(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("i16"),
                    false,
                    true,
                    short.class,
                    "int16",
                    "short");
    DataType I32 =
            new DataTypeImpl(
                    ValueLayout.JAVA_INT_UNALIGNED.withName("i32"),
                    false,
                    true,
                    int.class,
                    "int32",
                    "int");
    DataType I64 =
            new DataTypeImpl(
                    ValueLayout.JAVA_LONG_UNALIGNED.withName("i64"),
                    false,
                    true,
                    long.class,
                    "int64",
                    "long");

    DataType FP16 =
            new DataTypeImpl(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("fp16"),
                    true,
                    false,
                    short.class,
                    "float16"); // no float16 in Java
    DataType BF16 =
            new DataTypeImpl(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("bf16"),
                    true,
                    false,
                    short.class,
                    "bfloat16"); // no bfloat16 in Java
    DataType FP32 =
            new DataTypeImpl(
                    ValueLayout.JAVA_FLOAT_UNALIGNED.withName("fp32"),
                    true,
                    false,
                    float.class,
                    "float32",
                    "float");
    DataType FP64 =
            new DataTypeImpl(
                    ValueLayout.JAVA_DOUBLE_UNALIGNED.withName("fp64"),
                    true,
                    false,
                    double.class,
                    "float64",
                    "double");

    DataType Q4_0 = blockType("q4_0", 32, 18);
    DataType Q4_1 = blockType("q4_1", 32, 20);
    DataType Q5_1 = blockType("q5_1", 32, 24);
    DataType Q8_0 = blockType("q8_0", 32, 34);
    DataType Q4_K = blockType("q4_k", 256, 144);
    DataType Q5_K = blockType("q5_k", 256, 176);
    DataType Q6_K = blockType("q6_k", 256, 210);
    DataType MXFP4 = blockType("mxfp4", 32, 17);
    DataType NVFP4 = blockType("nvfp4", 64, 36);
    DataType Q1_0 = blockType("q1_0", 128, 18);
    DataType TQ1_0 = blockType("tq1_0", 256, 54);
    DataType TQ2_0 = blockType("tq2_0", 256, 66);

    private static DataType blockType(String name, long elements, long bytes) {
        return new DataTypeImpl(
                elements,
                MemoryLayout.sequenceLayout(bytes, ValueLayout.JAVA_BYTE).withName(name),
                false,
                false,
                null);
    }

    /**
     * Bytes for {@code elementCount} addressable storage units - for block dtypes the "elements" of
     * a physical shape ARE blocks (Q8_0: {@code byteSizeFor(2)} is 68 bytes holding 64 logical
     * elements).
     */
    default long byteSizeFor(long elementCount) {
        if (elementCount < 0) {
            throw new IllegalArgumentException("negative count");
        }
        return Math.multiplyExact(byteSize(), elementCount);
    }

    /** Bytes for a physical shape: {@link #byteSizeFor(long)} of its size. */
    default long byteSizeFor(Shape shape) {
        return byteSizeFor(shape.size());
    }

    /**
     * The element-dimensioned shape for a physical (storage) shape. Only block-quantized types
     * ({@link #elementsPerBlock()} &gt; 1) distinguish the two: their physical shape counts storage
     * BLOCKS, and blocking always tiles the innermost (last) storage axis, so the logical shape is
     * the physical shape with its last dimension (in flatten order) multiplied by {@code
     * elementsPerBlock()}. For every other type the physical shape IS the logical shape and this
     * returns it unchanged.
     *
     * <p>This is a pure shape function over a shape in STORAGE axis order (blocked axis last): it
     * un-tiles the last dimension it is handed, so a permuted axis order must be transposed back
     * first. Nested shapes keep their structure - only the last dim is scaled.
     */
    default Shape logicalShape(Shape physical) {
        long epb = elementsPerBlock();
        if (epb == 1 || physical.isScalar()) {
            return physical;
        }
        long[] dims = physical.toArray(); // flatten order; the last entry is the blocked dim
        dims[dims.length - 1] = Math.multiplyExact(dims[dims.length - 1], epb);
        return Shape.template(physical, dims); // same nesting, last dim scaled
    }

    /**
     * The inverse of {@link #logicalShape(Shape)}: the storage shape for an element-dimensioned
     * shape, dividing the last dimension (in flatten order) by {@code elementsPerBlock()} - which
     * must divide it exactly, a block is never split. Identity for non-block types; nested shapes
     * keep their structure.
     */
    default Shape physicalShape(Shape logical) {
        long epb = elementsPerBlock();
        if (epb == 1 || logical.isScalar()) {
            return logical;
        }
        long[] dims = logical.toArray(); // flatten order; the last entry is the blocked dim
        long last = dims[dims.length - 1];
        if (last % epb != 0) {
            throw new IllegalArgumentException(
                    "innermost dimension "
                            + last
                            + " not divisible by block size "
                            + epb
                            + " of "
                            + name());
        }
        dims[dims.length - 1] = last / epb;
        return Shape.template(logical, dims); // same nesting, last dim scaled
    }
}

final class DataTypeImpl implements DataType {

    final String name;
    final long byteSize;
    final long elementsPerBlock;
    final MemoryLayout layout;
    final boolean isFloatingPoint;
    final boolean isIntegral;
    final Class<?> javaClass;
    final List<String> aliases;

    DataTypeImpl(
            String name,
            long elementsPerBlock,
            MemoryLayout layout,
            boolean isFloatingPoint,
            boolean isIntegral,
            Class<?> javaClass,
            String... aliases) {
        this.name = name;
        this.elementsPerBlock = elementsPerBlock;
        this.byteSize = layout.byteSize();
        this.layout = layout;
        this.isFloatingPoint = isFloatingPoint;
        this.isIntegral = isIntegral;
        this.javaClass = javaClass;
        this.aliases = aliases == null ? List.of() : List.of(aliases);
    }

    DataTypeImpl(
            long elementsPerBlock,
            MemoryLayout layout,
            boolean isFloatingPoint,
            boolean isIntegral,
            Class<?> javaClass,
            String... aliases) {
        this(
                layout.name().orElseThrow(),
                elementsPerBlock,
                layout,
                isFloatingPoint,
                isIntegral,
                javaClass,
                aliases);
    }

    DataTypeImpl(
            MemoryLayout layout,
            boolean isFloatingPoint,
            boolean isIntegral,
            Class<?> javaClass,
            String... aliases) {
        this(
                layout.name().orElseThrow(),
                1L,
                layout,
                isFloatingPoint,
                isIntegral,
                javaClass,
                aliases);
    }

    DataTypeImpl(
            MemoryLayout layout,
            String name,
            boolean isFloatingPoint,
            boolean isIntegral,
            Class<?> javaClass,
            String... aliases) {
        this(name, 1L, layout, isFloatingPoint, isIntegral, javaClass, aliases);
    }

    @Override
    public long byteSize() {
        return byteSize;
    }

    @Override
    public long elementsPerBlock() {
        return elementsPerBlock;
    }

    @Override
    public MemoryLayout layout() {
        return layout;
    }

    @Override
    public boolean isFloatingPoint() {
        return isFloatingPoint;
    }

    @Override
    public boolean isIntegral() {
        return isIntegral;
    }

    @Override
    public List<String> aliases() {
        return aliases;
    }

    @Override
    public String name() {
        return name;
    }

    @Override
    public String toString() {
        return name;
    }
}
