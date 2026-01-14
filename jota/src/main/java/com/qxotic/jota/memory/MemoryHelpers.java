
package com.qxotic.jota.memory;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;

public final class MemoryHelpers {

    private MemoryHelpers() {
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, DataType dataType, long count, Number value) {
        Shape shape = Shape.flat(count);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        long byteSize = dataType.byteSizeFor(count);
        Memory<B> memory = allocator.allocateMemory(dataType, count);
        MemoryOperations<B> memoryOperations = context.memoryOperations();
        if (dataType == DataType.I8) {
            memoryOperations.fillByte(memory, 0, byteSize, value.byteValue());
        } else if (dataType == DataType.I16) {
            memoryOperations.fillShort(memory, 0, byteSize, value.shortValue());
        } else if (dataType == DataType.I32) {
            memoryOperations.fillInt(memory, 0, byteSize, value.intValue());
        } else if (dataType == DataType.I64) {
            memoryOperations.fillLong(memory, 0, byteSize, value.longValue());
        } else if (dataType == DataType.F32) {
            memoryOperations.fillFloat(memory, 0, byteSize, value.floatValue());
        } else if (dataType == DataType.F64) {
            memoryOperations.fillDouble(memory, 0, byteSize, value.doubleValue());
        } else if (dataType == DataType.F16) {
            memoryOperations.fillShort(memory, 0, byteSize, Float.floatToFloat16(value.floatValue()));
        } else if (dataType == DataType.BF16) {
            memoryOperations.fillShort(memory, 0, byteSize, BFloat16.fromFloat(value.byteValue()));
        } else {
            throw new IllegalArgumentException("unsupported value " + value);
        }
        return MemoryView.of(memory, dataType, Layout.rowMajor(shape));
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

    public static <B> MemoryView<B> arange(MemoryContext<B> context, DataType dataType, long end) {
        return arange(context, dataType, 0, end, 1);
    }

    public static <B> MemoryView<B> arange(MemoryContext<B> context, DataType dataType,
                                           long start, long end, long step) {
        if (step == 0) {
            throw new IllegalArgumentException("step cannot be 0");
        }

        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            throw new UnsupportedOperationException("Context does not support direct memory access");
        }

        long count = arangeCount(start, end, step);
        Shape shape = Shape.flat(count);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        Memory<B> memory = allocator.allocateMemory(dataType, count);
        MemoryView<B> view = MemoryView.of(memory, dataType, Layout.rowMajor(shape));

        for (long i = 0; i < count; i++) {
            double value = start + (double) i * step;
            long offset = i * dataType.byteSize();
            if (dataType == DataType.I8) {
                memoryAccess.writeByte(memory, offset, (byte) value);
            } else if (dataType == DataType.I16) {
                memoryAccess.writeShort(memory, offset, (short) value);
            } else if (dataType == DataType.I32) {
                memoryAccess.writeInt(memory, offset, (int) value);
            } else if (dataType == DataType.I64) {
                memoryAccess.writeLong(memory, offset, (long) value);
            } else if (dataType == DataType.F16) {
                memoryAccess.writeShort(memory, offset, (short) Float.floatToFloat16((float) value));
            } else if (dataType == DataType.BF16) {
                memoryAccess.writeShort(memory, offset, BFloat16.fromFloat((float) value));
            } else if (dataType == DataType.F32) {
                memoryAccess.writeFloat(memory, offset, (float) value);
            } else if (dataType == DataType.F64) {
                memoryAccess.writeDouble(memory, offset, value);
            } else {
                throw new IllegalArgumentException("Unsupported data type for arange: " + dataType);
            }
        }
        return view;
    }

    private static long arangeCount(long start, long end, long step) {
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
}

