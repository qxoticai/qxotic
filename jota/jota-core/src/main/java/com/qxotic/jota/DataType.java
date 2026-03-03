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

    /** scale * q_values[i] */
    DataType Q8_0 =
            new DataTypeImpl(
                    32,
                    MemoryLayout.structLayout(
                                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
                                    MemoryLayout.sequenceLayout(32, ValueLayout.JAVA_BYTE)
                                            .withName("q_values"))
                            .withName("q8_0"),
                    false,
                    false,
                    null);

    //    /**
    //     * scale * q_values[i] + zero_point
    //     */
    //    DataType Q8_1 = new DataTypeImpl(
    //            32,
    //            MemoryLayout.structLayout(
    //                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
    //                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("zero_point"), // float16
    //                    MemoryLayout.sequenceLayout(32,
    // ValueLayout.JAVA_BYTE).withName("q_values")
    //            ).withName("q8_1"), true, null);

    DataType Q4_0 =
            new DataTypeImpl(
                    32,
                    MemoryLayout.structLayout(
                                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
                                    MemoryLayout.sequenceLayout(16, ValueLayout.JAVA_BYTE)
                                            .withName("q_values"))
                            .withName("q4_0"),
                    false,
                    false,
                    null);

    //    DataType Q4_1 = new DataTypeImpl(
    //            32,
    //            MemoryLayout.structLayout(
    //                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
    //                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("zero_point"), // float16
    //                    MemoryLayout.sequenceLayout(16,
    // ValueLayout.JAVA_BYTE).withName("q_nibbles")
    //            ).withName("q4_1"), true, null);

    default long byteSizeFor(long elementCount) {
        if (elementCount < 0) {
            throw new IllegalArgumentException("negative count");
        }
        return Math.multiplyExact(byteSize(), elementCount);
    }

    default long byteSizeFor(Shape shape) {
        return byteSizeFor(shape.size());
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
