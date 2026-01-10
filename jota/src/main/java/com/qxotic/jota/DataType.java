package com.qxotic.jota;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;

public interface DataType {
    long byteSize(); // block size in bytes

    long elementsPerBlock(); // how many elements per block

    MemoryLayout layout();

    boolean isFloatingPoint();

    boolean isIntegral();

    DataType I8 = new DataTypeImpl(ValueLayout.JAVA_BYTE.withName("i8"), false, true, byte.class);
    DataType I16 = new DataTypeImpl(ValueLayout.JAVA_SHORT_UNALIGNED.withName("i16"), false, true, short.class);
    DataType I32 = new DataTypeImpl(ValueLayout.JAVA_INT_UNALIGNED.withName("i32"), false, true, int.class);
    DataType I64 = new DataTypeImpl(ValueLayout.JAVA_LONG_UNALIGNED.withName("i64"), false, true, long.class);

    DataType F16 = new DataTypeImpl(ValueLayout.JAVA_SHORT_UNALIGNED.withName("f16"), true, false, short.class); // no float16 in Java
    DataType BF16 = new DataTypeImpl(ValueLayout.JAVA_SHORT_UNALIGNED.withName("bf16"), true, false, short.class); // no bfloat16 in Java
    DataType F32 = new DataTypeImpl(ValueLayout.JAVA_FLOAT_UNALIGNED.withName("f32"), true, false, float.class);
    DataType F64 = new DataTypeImpl(ValueLayout.JAVA_DOUBLE_UNALIGNED.withName("f64"), true, false, double.class);

    /**
     * scale * q_values[i]
     */
    DataType Q8_0 = new DataTypeImpl(
            32,
            MemoryLayout.structLayout(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
                    MemoryLayout.sequenceLayout(32, ValueLayout.JAVA_BYTE).withName("q_values")
            ).withName("q8_0"), false, false, null);

//    /**
//     * scale * q_values[i] + zero_point
//     */
//    DataType Q8_1 = new DataTypeImpl(
//            32,
//            MemoryLayout.structLayout(
//                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
//                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("zero_point"), // float16
//                    MemoryLayout.sequenceLayout(32, ValueLayout.JAVA_BYTE).withName("q_values")
//            ).withName("q8_1"), true, null);

    DataType Q4_0 = new DataTypeImpl(
            32,
            MemoryLayout.structLayout(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
                    MemoryLayout.sequenceLayout(16, ValueLayout.JAVA_BYTE).withName("q_values")
            ).withName("q4_0"), false, false, null);

//    DataType Q4_1 = new DataTypeImpl(
//            32,
//            MemoryLayout.structLayout(
//                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
//                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("zero_point"), // float16
//                    MemoryLayout.sequenceLayout(16, ValueLayout.JAVA_BYTE).withName("q_nibbles")
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

    final String name;
    final long byteSize;
    final long elementsPerBlock;
    final MemoryLayout layout;
    final boolean isFloatingPoint;
    final boolean isIntegral;
    final Class<?> javaClass;

    DataTypeImpl(String name, long elementsPerBlock, MemoryLayout layout, boolean isFloatingPoint, boolean isIntegral, Class<?> javaClass) {
        this.name = name;
        this.elementsPerBlock = elementsPerBlock;
        this.byteSize = layout.byteSize();
        this.layout = layout;
        this.isFloatingPoint = isFloatingPoint;
        this.isIntegral = isIntegral;
        this.javaClass = javaClass;
    }

    DataTypeImpl(long elementsPerBlock, MemoryLayout layout, boolean isFloatingPoint, boolean isIntegral, Class<?> javaClass) {
        this(layout.name().orElseThrow(), elementsPerBlock, layout, isFloatingPoint, isIntegral, javaClass);
    }

    DataTypeImpl(MemoryLayout layout, boolean isFloatingPoint, boolean isIntegral, Class<?> javaClass) {
        this(1L, layout, isFloatingPoint, isIntegral, javaClass);
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
    public String toString() {
        return name;
    }
}
