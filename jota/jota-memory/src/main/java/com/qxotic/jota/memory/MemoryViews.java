package com.qxotic.jota.memory;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.internal.MemoryFactory;
import com.qxotic.jota.memory.internal.MemoryViewFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * The {@link MemoryView} constructors: wrap existing memory ({@code of}, {@code rowMajor}),
 * allocate fresh storage ({@code allocate}), or allocate and fill ({@code zeros}, {@code ones},
 * {@code full}, {@code arange}).
 */
public final class MemoryViews {

    private MemoryViews() {}

    /** A view over {@code memory} with the given layout; nothing is allocated. */
    public static <B> MemoryView<B> of(
            Memory<B> memory, long byteOffset, DataType dataType, Layout layout) {
        return MemoryView.of(memory, byteOffset, dataType, layout);
    }

    public static <B> MemoryView<B> of(Memory<B> memory, DataType dataType, Layout layout) {
        return MemoryView.of(memory, dataType, layout);
    }

    public static <B> MemoryView<B> rowMajor(Memory<B> memory, DataType dataType, Shape shape) {
        return MemoryView.rowMajor(memory, dataType, shape);
    }

    /** Allocates row-major storage for {@code shape} from {@code allocator}; contents undefined. */
    public static <B> MemoryView<B> allocate(
            MemoryAllocator<B> allocator, DataType dataType, Shape shape) {
        return MemoryViewFactory.allocate(allocator, dataType, shape);
    }

    public static <B> MemoryView<B> full(
            MemoryDomain<B> domain, DataType dataType, long count, Number value) {
        Shape shape = Shape.flat(count);
        MemoryAllocator<B> allocator = domain.memoryAllocator();
        long byteSize = dataType.byteSizeFor(count);
        Memory<B> memory = allocator.allocateMemory(dataType, count);
        MemoryOperations<B> memoryOperations = domain.memoryOperations();
        if (dataType == DataType.BOOL) {
            // doubleValue() handles all numeric types without truncation
            // != 0.0 treats both +0.0 and -0.0 as false, NaN as true (NumPy convention)
            byte boolByte = (byte) (value.doubleValue() != 0.0 ? 1 : 0);
            memoryOperations.fillByte(memory, 0, byteSize, boolByte);
        } else if (dataType == DataType.I8) {
            memoryOperations.fillByte(memory, 0, byteSize, value.byteValue());
        } else if (dataType == DataType.I16) {
            memoryOperations.fillShort(memory, 0, byteSize, value.shortValue());
        } else if (dataType == DataType.I32) {
            memoryOperations.fillInt(memory, 0, byteSize, value.intValue());
        } else if (dataType == DataType.I64) {
            memoryOperations.fillLong(memory, 0, byteSize, value.longValue());
        } else if (dataType == DataType.FP32) {
            memoryOperations.fillFloat(memory, 0, byteSize, value.floatValue());
        } else if (dataType == DataType.FP64) {
            memoryOperations.fillDouble(memory, 0, byteSize, value.doubleValue());
        } else if (dataType == DataType.FP16) {
            memoryOperations.fillShort(
                    memory, 0, byteSize, Float.floatToFloat16(value.floatValue()));
        } else if (dataType == DataType.BF16) {
            memoryOperations.fillShort(memory, 0, byteSize, BFloat16.fromFloat(value.floatValue()));
        } else {
            throw new IllegalArgumentException("unsupported value " + value);
        }
        return MemoryView.of(memory, dataType, Layout.rowMajor(shape));
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, boolean boolValue) {
        return full(domain, DataType.BOOL, count, boolValue ? 1 : 0);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, byte byteValue) {
        return full(domain, DataType.I8, count, byteValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, short shortValue) {
        return full(domain, DataType.I16, count, shortValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, int intValue) {
        return full(domain, DataType.I32, count, intValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, long longValue) {
        return full(domain, DataType.I64, count, longValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, float floatValue) {
        return full(domain, DataType.FP32, count, floatValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, double doubleValue) {
        return full(domain, DataType.FP64, count, doubleValue);
    }

    public static <B> MemoryView<B> full(
            MemoryDomain<B> domain, DataType dataType, Shape shape, Number value) {
        MemoryView<B> base = full(domain, dataType, shape.size(), value);
        return base.view(shape);
    }

    public static <B> MemoryView<B> ones(MemoryDomain<B> domain, DataType dataType, long count) {
        return full(domain, dataType, count, 1);
    }

    public static <B> MemoryView<B> ones(MemoryDomain<B> domain, DataType dataType, Shape shape) {
        return full(domain, dataType, shape, 1);
    }

    public static <B> MemoryView<B> zeros(MemoryDomain<B> domain, DataType dataType, long count) {
        return full(domain, dataType, count, 0);
    }

    public static <B> MemoryView<B> zeros(MemoryDomain<B> domain, DataType dataType, Shape shape) {
        return full(domain, dataType, shape, 0);
    }

    /**
     * {@code [0, endExclusive)} as {@code dataType}, row-major. Built on the host and transferred
     * with one {@link MemoryOperations#copyFromNative}, so opaque (GPU) domains work like host
     * ones. A non-positive end yields an empty view.
     */
    public static <B> MemoryView<B> arange(
            MemoryDomain<B> domain, DataType dataType, long endExclusive) {
        if (!domain.supportsDataType(dataType)) {
            throw new IllegalArgumentException(
                    "Domain does not support "
                            + dataType
                            + " (requires "
                            + dataType.byteSize()
                            + "-byte alignment, domain has "
                            + domain.memoryGranularity()
                            + "-byte granularity)");
        }
        long count = Math.max(0, endExclusive);
        Memory<B> memory = domain.memoryAllocator().allocateMemory(dataType, count);
        MemoryView<B> view = MemoryView.of(memory, dataType, Layout.rowMajor(Shape.flat(count)));
        if (count == 0) {
            return view;
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment sequence = arena.allocate(dataType.byteSizeFor(count));
            writeSequence(sequence, dataType, count);
            domain.memoryOperations()
                    .copyFromNative(
                            MemoryFactory.ofMemorySegment(sequence),
                            0,
                            memory,
                            0,
                            sequence.byteSize());
        }
        return view;
    }

    /** 0, 1, 2, ... encoded as {@code dataType}; the type is resolved once, not per element. */
    private static void writeSequence(MemorySegment s, DataType dataType, long count) {
        if (dataType == DataType.I8) {
            for (long i = 0; i < count; i++) s.set(JAVA_BYTE, i, (byte) i);
        } else if (dataType == DataType.I16) {
            for (long i = 0; i < count; i++) s.setAtIndex(JAVA_SHORT, i, (short) i);
        } else if (dataType == DataType.I32) {
            for (long i = 0; i < count; i++) s.setAtIndex(JAVA_INT, i, (int) i);
        } else if (dataType == DataType.I64) {
            for (long i = 0; i < count; i++) s.setAtIndex(JAVA_LONG, i, i);
        } else if (dataType == DataType.FP16) {
            for (long i = 0; i < count; i++) s.setAtIndex(JAVA_SHORT, i, Float.floatToFloat16(i));
        } else if (dataType == DataType.BF16) {
            for (long i = 0; i < count; i++) s.setAtIndex(JAVA_SHORT, i, BFloat16.fromFloat(i));
        } else if (dataType == DataType.FP32) {
            for (long i = 0; i < count; i++) s.setAtIndex(JAVA_FLOAT, i, (float) i);
        } else if (dataType == DataType.FP64) {
            for (long i = 0; i < count; i++) s.setAtIndex(JAVA_DOUBLE, i, (double) i);
        } else {
            throw new IllegalArgumentException("Unsupported data type for arange: " + dataType);
        }
    }
}
