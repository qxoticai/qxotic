package com.llm4j.jota;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.ValueLayout;

public interface DataType {
    long byteSize(); // block size in bytes

    long elementsPerBlock(); // how many elements per block

    MemoryLayout layout();

    boolean isFloatingPoint();

    DataType I8 = new DataTypeImpl(ValueLayout.JAVA_BYTE.withName("i8"), false);
    DataType I16 = new DataTypeImpl(ValueLayout.JAVA_SHORT_UNALIGNED.withName("i16"), false);
    DataType I32 = new DataTypeImpl(ValueLayout.JAVA_INT_UNALIGNED.withName("i32"), false);
    DataType I64 = new DataTypeImpl(ValueLayout.JAVA_LONG_UNALIGNED.withName("i64"), false);

    DataType F16 = new DataTypeImpl(ValueLayout.JAVA_SHORT_UNALIGNED.withName("f16"), true);
    DataType BF16 = new DataTypeImpl(ValueLayout.JAVA_SHORT_UNALIGNED.withName("bf16"), true);
    DataType F32 = new DataTypeImpl(ValueLayout.JAVA_FLOAT_UNALIGNED.withName("f32"), true);
    DataType F64 = new DataTypeImpl(ValueLayout.JAVA_DOUBLE_UNALIGNED.withName("f64"), true);

    /**
     * scale * q_values[i]
     */
    DataType Q8_0 = new DataTypeImpl(
            32,
            MemoryLayout.structLayout(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
                    MemoryLayout.sequenceLayout(32, ValueLayout.JAVA_BYTE).withName("q_values")
            ).withName("q8_0"), true);

    /**
     * scale * q_values[i] + zero_point
     */
    DataType Q8_1 = new DataTypeImpl(
            32,
            MemoryLayout.structLayout(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("zero_point"), // float16
                    MemoryLayout.sequenceLayout(32, ValueLayout.JAVA_BYTE).withName("q_values")
            ).withName("q8_1"), true);

    DataType Q4_0 = new DataTypeImpl(
            32,
            MemoryLayout.structLayout(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
                    MemoryLayout.sequenceLayout(16, ValueLayout.JAVA_BYTE).withName("q_values")
            ).withName("q4_0"), true);

    DataType Q4_1 = new DataTypeImpl(
            32,
            MemoryLayout.structLayout(
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("scale"), // float16
                    ValueLayout.JAVA_SHORT_UNALIGNED.withName("zero_point"), // float16
                    MemoryLayout.sequenceLayout(16, ValueLayout.JAVA_BYTE).withName("q_nibbles")
            ).withName("q4_1"), true);

    DataType BITS = new DataTypeImpl(8, ValueLayout.JAVA_BYTE.withName("byte"), false);

    default long byteSizeFor(long elementCount) {
        if (elementCount < 0) {
            throw new IllegalArgumentException("negative count");
        }
        return Math.multiplyExact(byteSize(), elementCount);
    }

    default long byteSizeFor(Shape shape) {
        return byteSizeFor(shape.totalNumberOfElements());
    }

}

final class DataTypeImpl implements DataType {

    final String name;
    final long byteSize;
    final long elementsPerBlock;
    final MemoryLayout layout;
    final boolean isFloatingPoint;

    DataTypeImpl(String name, long elementsPerBlock, MemoryLayout layout, boolean isFloatingPoint) {
        this.name = name;
        this.elementsPerBlock = elementsPerBlock;
        this.byteSize = layout.byteSize();
        this.layout = layout;
        this.isFloatingPoint = isFloatingPoint;
    }

    DataTypeImpl(long elementsPerBlock, MemoryLayout layout, boolean isFloatingPoint) {
        this(layout.name().orElseThrow(), elementsPerBlock, layout, isFloatingPoint);
    }

    DataTypeImpl(MemoryLayout layout, boolean isFloatingPoint) {
        this(1L, layout, isFloatingPoint);
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
    public String toString() {
        return name;
    }
}
