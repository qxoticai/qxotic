
package ai.qxotic.jota.memory;

import ai.qxotic.jota.BFloat16;
import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;

public final class MemoryHelpers {

    private MemoryHelpers() {
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, DataType dataType, long count, Number value) {
        Shape shape = Shape.flat(count);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        long byteSize = dataType.byteSizeFor(count);
        Memory<B> memory = allocator.allocateMemory(dataType, count);
        MemoryOperations<B> memoryOperations = context.memoryOperations();
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
            memoryOperations.fillShort(memory, 0, byteSize, Float.floatToFloat16(value.floatValue()));
        } else if (dataType == DataType.BF16) {
            memoryOperations.fillShort(memory, 0, byteSize, BFloat16.fromFloat(value.byteValue()));
        } else {
            throw new IllegalArgumentException("unsupported value " + value);
        }
        return MemoryView.of(memory, dataType, Layout.rowMajor(shape));
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, long count, boolean boolValue) {
        return full(context, DataType.BOOL, count, boolValue ? 1 : 0);
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, long count, byte byteValue) {
        return full(context, DataType.I8, count, byteValue);
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, long count, short shortValue) {
        return full(context, DataType.I16, count, shortValue);
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, long count, int intValue) {
        return full(context, DataType.I32, count, intValue);
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, long count, long longValue) {
        return full(context, DataType.I64, count, longValue);
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, long count, float floatValue) {
        return full(context, DataType.FP32, count, floatValue);
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, long count, double doubleValue) {
        return full(context, DataType.FP64, count, doubleValue);
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, DataType dataType, Shape shape, Number value) {
        MemoryView<B> base = full(context, dataType, shape.size(), value);
        return base.view(shape);
    }

    public static <B> MemoryView<B> ones(MemoryContext<B> context, DataType dataType, long count) {
        return full(context, dataType, count, 1);
    }

    public static <B> MemoryView<B> ones(MemoryContext<B> context, DataType dataType, Shape shape) {
        return full(context, dataType, shape, 1);
    }

    public static <B> MemoryView<B> zeros(MemoryContext<B> context, DataType dataType, long count) {
        return full(context, dataType, count, 0);
    }

    public static <B> MemoryView<B> zeros(MemoryContext<B> context, DataType dataType, Shape shape) {
        return full(context, dataType, shape, 0);
    }

    public static <B> MemoryView<B> arange(MemoryContext<B> context, DataType dataType, long endExclusive) {
        if (dataType.isIntegral() || dataType == DataType.BOOL) {
            return arangeIntegral(context, dataType, 0L, endExclusive, 1L);
        } else if (dataType.isFloatingPoint()) {
            return arangeFloat(context, dataType, 0.0, (double) endExclusive, 1.0);
        } else {
            throw new IllegalArgumentException("Unsupported data type for arange: " + dataType);
        }
    }

    // ============================================================
    // INTERNAL IMPLEMENTATION - Integral types
    // ============================================================

    private static <B> MemoryView<B> arangeIntegral(MemoryContext<B> context, DataType dataType,
                                                    long start, long end, long step) {
        if (step == 0) {
            throw new IllegalArgumentException("step cannot be 0");
        }
        if (!dataType.isIntegral()) {
            throw new IllegalArgumentException("Integral arange requires integral DataType, got: " + dataType);
        }
        if (!context.supportsDataType(dataType)) {
            throw new IllegalArgumentException(
                "Context does not support " + dataType +
                " (requires " + dataType.byteSize() + "-byte alignment, context has " +
                context.memoryGranularity() + "-byte granularity)"
            );
        }

        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            throw new UnsupportedOperationException("Context does not support direct memory access");
        }

        long count = arangeCountLong(start, end, step);
        Shape shape = Shape.flat(count);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        Memory<B> memory = allocator.allocateMemory(dataType, count);
        MemoryView<B> view = MemoryView.of(memory, dataType, Layout.rowMajor(shape));

        for (long i = 0; i < count; i++) {
            long value = start + i * step;
            long offset = i * dataType.byteSize();
            if (dataType == DataType.I8) {
                memoryAccess.writeByte(memory, offset, (byte) value);
            } else if (dataType == DataType.I16) {
                memoryAccess.writeShort(memory, offset, (short) value);
            } else if (dataType == DataType.I32) {
                memoryAccess.writeInt(memory, offset, (int) value);
            } else if (dataType == DataType.I64) {
                memoryAccess.writeLong(memory, offset, value);
            } else {
                throw new IllegalArgumentException("Unsupported data type for arange: " + dataType);
            }
        }
        return view;
    }

    // ============================================================
    // INTERNAL IMPLEMENTATION - Floating-point types
    // ============================================================

    private static <B> MemoryView<B> arangeFloat(MemoryContext<B> context, DataType dataType,
                                                 double start, double end, double step) {
        if (step == 0.0) {
            throw new IllegalArgumentException("step cannot be 0");
        }
        if (!dataType.isFloatingPoint()) {
            throw new IllegalArgumentException("Floating-point arange requires floating-point DataType, got: " + dataType);
        }
        if (!context.supportsDataType(dataType)) {
            throw new IllegalArgumentException(
                "Context does not support " + dataType +
                " (requires " + dataType.byteSize() + "-byte alignment, context has " +
                context.memoryGranularity() + "-byte granularity)"
            );
        }

        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            throw new UnsupportedOperationException("Context does not support direct memory access");
        }

        long count = arangeCountDouble(start, end, step);
        Shape shape = Shape.flat(count);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        Memory<B> memory = allocator.allocateMemory(dataType, count);
        MemoryView<B> view = MemoryView.of(memory, dataType, Layout.rowMajor(shape));

        for (long i = 0; i < count; i++) {
            double value = start + i * step;
            long offset = i * dataType.byteSize();
            if (dataType == DataType.FP16) {
                memoryAccess.writeShort(memory, offset, Float.floatToFloat16((float) value));
            } else if (dataType == DataType.BF16) {
                memoryAccess.writeShort(memory, offset, BFloat16.fromFloat((float) value));
            } else if (dataType == DataType.FP32) {
                memoryAccess.writeFloat(memory, offset, (float) value);
            } else if (dataType == DataType.FP64) {
                memoryAccess.writeDouble(memory, offset, value);
            } else {
                throw new IllegalArgumentException("Unsupported data type for arange: " + dataType);
            }
        }
        return view;
    }



    // ============================================================
    // HELPER METHODS
    // ============================================================

    private static long arangeCountLong(long start, long end, long step) {
        if (step > 0) {
            if (start >= end) {
                return 0;
            }
            long span = end - start;
            long count = span / step;
            if (span % step != 0) {
                count++;
            }
            return count;
        }
        if (start <= end) {
            return 0;
        }
        long span = start - end;
        long stride = Math.abs(step);
        long count = span / stride;
        if (span % stride != 0) {
            count++;
        }
        return count;
    }

    private static long arangeCountDouble(double start, double end, double step) {
        if (step > 0.0) {
            if (start >= end) {
                return 0;
            }
            return (long) Math.ceil((end - start) / step);
        }
        if (start <= end) {
            return 0;
        }
        return (long) Math.ceil((start - end) / Math.abs(step));
    }
}

