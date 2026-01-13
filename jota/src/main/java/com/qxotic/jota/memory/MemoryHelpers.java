
package com.qxotic.jota.memory;

import com.qxotic.jota.BFloat16;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;

public final class MemoryHelpers {

    private MemoryHelpers() {
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, DataType dataType, long count, double value) {
        Shape shape = Shape.flat(count);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        Memory<B> memory = allocator.allocateMemory(dataType, count);
        MemoryView<B> view = MemoryView.of(memory, 0L, dataType, Layout.rowMajor(shape));
        fillView(context, view, value);
        return view;
    }

    public static <B> MemoryView<B> full(MemoryContext<B> context, Shape shape, DataType dataType, double value) {
        MemoryView<B> base = full(context, dataType, shape.size(), value);
        return base.view(shape);
    }

    public static <B> MemoryView<B> ones(MemoryContext<B> context, DataType dataType, long count) {
        return full(context, dataType, count, 1.0);
    }

    public static <B> MemoryView<B> ones(MemoryContext<B> context, Shape shape, DataType dataType) {
        return full(context, shape, dataType, 1.0);
    }

    public static <B> MemoryView<B> zeros(MemoryContext<B> context, DataType dataType, long count) {
        return full(context, dataType, count, 0.0);
    }

    public static <B> MemoryView<B> zeros(MemoryContext<B> context, Shape shape, DataType dataType) {
        return full(context, shape, dataType, 0.0);
    }

    public static <B> MemoryView<B> arange(MemoryContext<B> context, DataType dataType, long end) {
        return arange(context, dataType, 0, end, 1);
    }

    public static <B> MemoryView<B> arange(MemoryContext<B> context, DataType dataType,
                                           long start, long end, long step) {
        if (step == 0) {
            throw new IllegalArgumentException("step cannot be 0");
        }
        long count = arangeCount(start, end, step);
        Shape shape = Shape.flat(count);
        MemoryAllocator<B> allocator = context.memoryAllocator();
        Memory<B> memory = allocator.allocateMemory(dataType, count);
        MemoryView<B> view = MemoryView.of(memory, 0L, dataType, Layout.rowMajor(shape));

        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess == null) {
            throw new UnsupportedOperationException("Context does not support direct memory access");
        }

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

    private static <B> void fillView(MemoryContext<B> context, MemoryView<B> view, double value) {
        DataType dataType = view.dataType();
        MemoryAccess<B> memoryAccess = context.memoryAccess();
        if (memoryAccess != null) {
            long count = view.shape().size();
            long baseOffset = view.byteOffset();
            long stride = dataType.byteSize();
            for (long i = 0; i < count; i++) {
                long offset = baseOffset + i * stride;
                if (dataType == DataType.I8) {
                    memoryAccess.writeByte(view.memory(), offset, (byte) value);
                } else if (dataType == DataType.I16) {
                    memoryAccess.writeShort(view.memory(), offset, (short) value);
                } else if (dataType == DataType.I32) {
                    memoryAccess.writeInt(view.memory(), offset, (int) value);
                } else if (dataType == DataType.I64) {
                    memoryAccess.writeLong(view.memory(), offset, (long) value);
                } else if (dataType == DataType.F16) {
                    memoryAccess.writeShort(view.memory(), offset,
                            (short) Float.floatToFloat16((float) value));
                } else if (dataType == DataType.BF16) {
                    memoryAccess.writeShort(view.memory(), offset, BFloat16.fromFloat((float) value));
                } else if (dataType == DataType.F32) {
                    memoryAccess.writeFloat(view.memory(), offset, (float) value);
                } else if (dataType == DataType.F64) {
                    memoryAccess.writeDouble(view.memory(), offset, value);
                } else {
                    throw new IllegalArgumentException("Unsupported data type for full: " + dataType);
                }
            }
            return;
        }

        MemoryOperations<B> operations = context.memoryOperations();
        long byteSize = dataType.byteSizeFor(view.shape());
        if (dataType == DataType.I8) {
            operations.fillByte(view.memory(), view.byteOffset(), byteSize, (byte) value);
        } else if (dataType == DataType.I16) {
            operations.fillShort(view.memory(), view.byteOffset(), byteSize, (short) value);
        } else if (dataType == DataType.I32) {
            operations.fillInt(view.memory(), view.byteOffset(), byteSize, (int) value);
        } else if (dataType == DataType.I64) {
            operations.fillLong(view.memory(), view.byteOffset(), byteSize, (long) value);
        } else if (dataType == DataType.F16) {
            operations.fillShort(view.memory(), view.byteOffset(), byteSize,
                    (short) Float.floatToFloat16((float) value));
        } else if (dataType == DataType.BF16) {
            operations.fillShort(view.memory(), view.byteOffset(), byteSize, BFloat16.fromFloat((float) value));
        } else if (dataType == DataType.F32) {
            operations.fillFloat(view.memory(), view.byteOffset(), byteSize, (float) value);
        } else if (dataType == DataType.F64) {
            operations.fillDouble(view.memory(), view.byteOffset(), byteSize, value);
        } else {
            throw new IllegalArgumentException("Unsupported data type for full: " + dataType);
        }
    }
}

