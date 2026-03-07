package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Indexing;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

final class StridedCopy {

    private StridedCopy() {}

    static <B> void copy(MemoryDomain<B> domain, MemoryView<B> src, MemoryView<B> dst) {
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

        if (src.isRowMajorContiguous() && dst.isRowMajorContiguous()) {
            long bytes = src.shape().size() * src.dataType().byteSize();
            if (bytes == 0) {
                return;
            }
            domain.memoryOperations()
                    .copy(src.memory(), src.byteOffset(), dst.memory(), dst.byteOffset(), bytes);
            return;
        }

        MemoryAccess<B> access = domain.directAccess();
        if (access == null) {
            copyViaHost(domain, src, dst);
            return;
        }

        if (src.memory().base() instanceof MemorySegment
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

    @SuppressWarnings("unchecked")
    private static <B> void copyViaHost(
            MemoryDomain<B> domain, MemoryView<B> src, MemoryView<B> dst) {
        DataType dtype = src.dataType();
        Shape shape = src.shape();
        MemoryDomain<MemorySegment> hostDomain = Environment.current().nativeMemoryDomain();
        MemoryOperations<B> deviceOps = domain.memoryOperations();
        MemoryAccess<MemorySegment> hostAccess = hostDomain.directAccess();

        try (Arena arena = Arena.ofConfined()) {
            // Mirror src on host, preserving strides so offsets remain valid.
            MemoryView<MemorySegment> hostSrc;
            if (src.isRowMajorContiguous()) {
                long bytes = shape.size() * dtype.byteSize();
                Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(arena.allocate(bytes));
                deviceOps.copyToNative(src.memory(), src.byteOffset(), mem, 0, bytes);
                hostSrc = MemoryView.rowMajor(mem, dtype, shape);
            } else {
                long bytes = src.memory().byteSize();
                Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(arena.allocate(bytes));
                deviceOps.copyToNative(src.memory(), 0, mem, 0, bytes);
                hostSrc = MemoryView.of(mem, src.byteOffset(), dtype, src.layout());
            }

            // Mirror dst on host and perform strided copy there.
            if (dst.isRowMajorContiguous()) {
                long bytes = shape.size() * dtype.byteSize();
                Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(arena.allocate(bytes));
                MemoryView<MemorySegment> hostDst = MemoryView.rowMajor(mem, dtype, shape);
                copyWithMemorySegment(
                        hostSrc.memory().base(), hostSrc, hostDst.memory().base(), hostDst, dtype);
                deviceOps.copyFromNative(mem, 0, dst.memory(), dst.byteOffset(), bytes);
            } else {
                long bytes = dst.memory().byteSize();
                Memory<MemorySegment> mem = MemoryFactory.ofMemorySegment(arena.allocate(bytes));
                deviceOps.copyToNative(dst.memory(), 0, mem, 0, bytes);
                MemoryView<MemorySegment> hostDst =
                        MemoryView.of(mem, dst.byteOffset(), dtype, dst.layout());
                copyWithMemorySegment(
                        hostSrc.memory().base(), hostSrc, hostDst.memory().base(), hostDst, dtype);
                deviceOps.copyFromNative(mem, 0, dst.memory(), 0, bytes);
            }
        }
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
