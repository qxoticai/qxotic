package com.qxotic.jota;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;
import java.util.List;

public interface DataType {
    long byteSize(); // block size in bytes

    long elementsPerBlock(); // how many elements per block

    MemoryLayout layout();

    boolean isFloatingPoint();

    boolean isIntegral();

    String name();

    List<String> aliases();

    static DataType defaultFloat() {
        return Environment.current().defaultFloat();
    }

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

    private static DataType blockType(String name, long elements, long bytes) {
        return new DataTypeImpl(
                elements,
                MemoryLayout.sequenceLayout(bytes, ValueLayout.JAVA_BYTE).withName(name),
                false,
                false,
                null);
    }

    default long byteSizeFor(long elementCount) {
        if (elementCount < 0) {
            throw new IllegalArgumentException("negative count");
        }
        return Math.multiplyExact(byteSize(), elementCount);
    }

    default long byteSizeFor(Shape shape) {
        return byteSizeFor(shape.size());
    }

    /**
     * The element-dimensioned shape for a physical (storage) shape. Only block-quantized types
     * ({@link #elementsPerBlock()} &gt; 1) distinguish the two: their {@code shape()} counts
     * storage BLOCKS, and blocking always tiles the innermost (last) storage axis, so the logical
     * shape is the physical shape with its last dimension multiplied by {@code elementsPerBlock()}.
     * For every other type the physical shape IS the logical shape and this returns it unchanged.
     *
     * <p>Nested shapes are not supported for block types (flat shapes only); the conversion follows
     * the storage axis order regardless of any permutation of the view.
     */
    default Shape logicalShape(Shape physical) {
        long epb = elementsPerBlock();
        if (epb == 1 || physical.isScalar()) {
            return physical;
        }
        if (!physical.isFlat()) {
            throw new UnsupportedOperationException(
                    "block-quantized type " + name() + " with nested shape " + physical);
        }
        int rank = physical.flatRank();
        long[] dims = new long[rank];
        for (int i = 0; i < rank; i++) {
            dims[i] = physical.flatAt(i);
        }
        dims[rank - 1] = Math.multiplyExact(dims[rank - 1], epb);
        return Shape.flat(dims);
    }

    /**
     * The inverse of {@link #logicalShape(Shape)}: the storage shape for an element-dimensioned
     * shape, dividing the innermost dimension by {@code elementsPerBlock()} (which must divide it
     * exactly). Identity for non-block types.
     */
    default Shape physicalShape(Shape logical) {
        long epb = elementsPerBlock();
        if (epb == 1 || logical.isScalar()) {
            return logical;
        }
        if (!logical.isFlat()) {
            throw new UnsupportedOperationException(
                    "block-quantized type " + name() + " with nested shape " + logical);
        }
        int rank = logical.flatRank();
        long last = logical.flatAt(rank - 1);
        if (last % epb != 0) {
            throw new IllegalArgumentException(
                    "innermost dimension "
                            + last
                            + " not divisible by block size "
                            + epb
                            + " of "
                            + name());
        }
        long[] dims = new long[rank];
        for (int i = 0; i < rank; i++) {
            dims[i] = logical.flatAt(i);
        }
        dims[rank - 1] = last / epb;
        return Shape.flat(dims);
    }
}

final class DataTypeImpl implements DataType {

    static DataType defaultFloatValue() {
        return DefaultFloatHolder.VALUE;
    }

    private static final class DefaultFloatHolder {
        private static final DataType VALUE = resolveDefaultFloat();

        private static DataType resolveDefaultFloat() {
            String name = System.getProperty("jota.defaultFloat");
            if (name == null) {
                return DataType.FP32;
            }
            DataType dataType = primitiveByName(name);
            if (!dataType.isFloatingPoint()) {
                throw new IllegalArgumentException("default float must be floating-point: " + name);
            }
            return dataType;
        }

        private DefaultFloatHolder() {}
    }

    private static DataType primitiveByName(String name) {
        List<DataType> primitives =
                List.of(
                        DataType.BOOL,
                        DataType.I8,
                        DataType.I16,
                        DataType.I32,
                        DataType.I64,
                        DataType.FP16,
                        DataType.BF16,
                        DataType.FP32,
                        DataType.FP64);
        for (DataType dt : primitives) {
            if (dt.name().equals(name) || dt.aliases().contains(name)) {
                return dt;
            }
        }
        throw new IllegalArgumentException("unknown primitive data type: " + name);
    }

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
