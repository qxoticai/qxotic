package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

final class StridedCopy {

    private StridedCopy() {}

    static <B> void copy(MemoryContext<B> context, MemoryView<B> src, MemoryView<B> dst) {
        if (src.dataType() != dst.dataType()) {
            throw new IllegalArgumentException(
                    "Data type mismatch: " + src.dataType() + " vs " + dst.dataType());
        }
        if (!src.shape().equals(dst.shape())) {
            throw new IllegalArgumentException(
                    "Shape mismatch: " + src.shape() + " vs " + dst.shape());
        }
        if (!src.memory().device().equals(dst.memory().device())) {
            throw new IllegalArgumentException("Source and destination devices must match");
        }

        if (src.isContiguous() && dst.isContiguous()) {
            long bytes = src.shape().size() * src.dataType().byteSize();
            if (bytes == 0) {
                return;
            }
            context.memoryOperations()
                    .copy(src.memory(), src.byteOffset(), dst.memory(), dst.byteOffset(), bytes);
            return;
        }

        MemoryAccess<B> access = context.memoryAccess();
        if (access == null) {
            throw new IllegalStateException("MemoryAccess is required for strided copies");
        }

        if (context.device() == ai.qxotic.jota.Device.PANAMA
                && src.memory().base() instanceof MemorySegment
                && dst.memory().base() instanceof MemorySegment) {
            copyWithMemorySegment(
                    (MemorySegment) src.memory().base(),
                    src,
                    (MemorySegment) dst.memory().base(),
                    dst,
                    src.dataType());
            return;
        }

        copyWithAccess(access, src, dst, src.dataType());
    }

    private static <B> void copyWithAccess(
            MemoryAccess<B> access, MemoryView<B> src, MemoryView<B> dst, DataType dataType) {
        long size = src.shape().size();
        for (long index = 0; index < size; index++) {
            long srcOffset = Indexing.linearToOffset(src, index);
            long dstOffset = Indexing.linearToOffset(dst, index);
            if (dataType == DataType.BOOL || dataType == DataType.I8) {
                byte value = access.readByte(src.memory(), srcOffset);
                access.writeByte(dst.memory(), dstOffset, value);
            } else if (dataType == DataType.I16
                    || dataType == DataType.FP16
                    || dataType == DataType.BF16) {
                short value = access.readShort(src.memory(), srcOffset);
                access.writeShort(dst.memory(), dstOffset, value);
            } else if (dataType == DataType.I32) {
                int value = access.readInt(src.memory(), srcOffset);
                access.writeInt(dst.memory(), dstOffset, value);
            } else if (dataType == DataType.I64) {
                long value = access.readLong(src.memory(), srcOffset);
                access.writeLong(dst.memory(), dstOffset, value);
            } else if (dataType == DataType.FP32) {
                float value = access.readFloat(src.memory(), srcOffset);
                access.writeFloat(dst.memory(), dstOffset, value);
            } else if (dataType == DataType.FP64) {
                double value = access.readDouble(src.memory(), srcOffset);
                access.writeDouble(dst.memory(), dstOffset, value);
            } else {
                throw new IllegalStateException("Unsupported data type: " + dataType);
            }
        }
    }

    private static void copyWithMemorySegment(
            MemorySegment src,
            MemoryView<?> srcView,
            MemorySegment dst,
            MemoryView<?> dstView,
            DataType dataType) {
        long size = srcView.shape().size();
        for (long index = 0; index < size; index++) {
            long srcOffset = Indexing.linearToOffset(srcView, index);
            long dstOffset = Indexing.linearToOffset(dstView, index);
            if (dataType == DataType.BOOL || dataType == DataType.I8) {
                byte value = src.get(ValueLayout.JAVA_BYTE, srcOffset);
                dst.set(ValueLayout.JAVA_BYTE, dstOffset, value);
            } else if (dataType == DataType.I16
                    || dataType == DataType.FP16
                    || dataType == DataType.BF16) {
                short value = src.get(ValueLayout.JAVA_SHORT_UNALIGNED, srcOffset);
                dst.set(ValueLayout.JAVA_SHORT_UNALIGNED, dstOffset, value);
            } else if (dataType == DataType.I32) {
                int value = src.get(ValueLayout.JAVA_INT_UNALIGNED, srcOffset);
                dst.set(ValueLayout.JAVA_INT_UNALIGNED, dstOffset, value);
            } else if (dataType == DataType.I64) {
                long value = src.get(ValueLayout.JAVA_LONG_UNALIGNED, srcOffset);
                dst.set(ValueLayout.JAVA_LONG_UNALIGNED, dstOffset, value);
            } else if (dataType == DataType.FP32) {
                float value = src.get(ValueLayout.JAVA_FLOAT_UNALIGNED, srcOffset);
                dst.set(ValueLayout.JAVA_FLOAT_UNALIGNED, dstOffset, value);
            } else if (dataType == DataType.FP64) {
                double value = src.get(ValueLayout.JAVA_DOUBLE_UNALIGNED, srcOffset);
                dst.set(ValueLayout.JAVA_DOUBLE_UNALIGNED, dstOffset, value);
            } else {
                throw new IllegalStateException("Unsupported data type: " + dataType);
            }
        }
    }
}
