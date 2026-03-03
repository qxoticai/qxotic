package com.qxotic.jota.memory;

import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public interface MemoryOperations<B> {

    void copy(Memory<B> src, long srcByteOffset, Memory<B> dst, long dstByteOffset, long byteSize);

    void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<B> dst,
            long dstByteOffset,
            long byteSize);

    void copyToNative(
            Memory<B> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize);

    void fillByte(Memory<B> memory, long byteOffset, long byteSize, byte byteValue);

    void fillShort(Memory<B> memory, long byteOffset, long byteSize, short shortValue);

    void fillInt(Memory<B> memory, long byteOffset, long byteSize, int intValue);

    void fillLong(Memory<B> memory, long byteOffset, long byteSize, long longValue);

    default void fillFloat(Memory<B> memory, long byteOffset, long byteSize, float floatValue) {
        fillInt(memory, byteOffset, byteSize, Float.floatToRawIntBits(floatValue));
    }

    default void fillDouble(Memory<B> memory, long byteOffset, long byteSize, double doubleValue) {
        fillLong(memory, byteOffset, byteSize, Double.doubleToRawLongBits(doubleValue));
    }

    static <S, T> void copy(
            MemoryOperations<S> srcOps,
            Memory<S> src,
            long srcByteOffset,
            MemoryOperations<T> dstOps,
            Memory<T> dst,
            long dstByteOffset,
            long byteSize,
            Memory<MemorySegment> buffer) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        checkGranularity(src, srcByteOffset, dst, dstByteOffset, byteSize);
        long copiedBytes = 0;
        while (copiedBytes < byteSize) {
            long chunkBytes = Math.min(byteSize - copiedBytes, buffer.byteSize());
            srcOps.copyToNative(src, srcByteOffset + copiedBytes, buffer, 0, chunkBytes);
            dstOps.copyFromNative(buffer, 0, dst, dstByteOffset + copiedBytes, chunkBytes);
            copiedBytes += chunkBytes;
        }
    }

    static <S, T> void copy(
            MemoryOperations<S> srcOps,
            Memory<S> src,
            long srcByteOffset,
            MemoryOperations<T> dstOps,
            Memory<T> dst,
            long dstByteOffset,
            long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        checkGranularity(src, srcByteOffset, dst, dstByteOffset, byteSize);
        long bufferSize = computeSizeBufferSize(byteSize);
        try (var arena = Arena.ofConfined()) {
            MemorySegment memorySegment = arena.allocate(bufferSize, 4 << 10);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
            copy(srcOps, src, srcByteOffset, dstOps, dst, dstByteOffset, byteSize, memory);
        }
    }

    private static long computeSizeBufferSize(long byteSize) {
        long chunkSize = 4 << 10; // 4KB
        int log2 = 64 - Long.numberOfLeadingZeros(byteSize / chunkSize + 1);
        long bufferSize = Math.max(4 << 10, byteSize / Math.max(1, log2));
        return bufferSize;
    }

    private static <S, T> void checkGranularity(
            Memory<S> src, long srcByteOffset, Memory<T> dst, long dstByteOffset, long byteSize) {
        long srcGranularity = src.memoryGranularity();
        long dstGranularity = dst.memoryGranularity();

        if (srcByteOffset % srcGranularity != 0) {
            throw new IllegalArgumentException(
                    "Source offset "
                            + srcByteOffset
                            + " is not aligned to source granularity "
                            + srcGranularity);
        }
        if (dstByteOffset % dstGranularity != 0) {
            throw new IllegalArgumentException(
                    "Destination offset "
                            + dstByteOffset
                            + " is not aligned to destination granularity "
                            + dstGranularity);
        }
        if (byteSize % srcGranularity != 0) {
            throw new IllegalArgumentException(
                    "Byte size "
                            + byteSize
                            + " is not a multiple of source granularity "
                            + srcGranularity);
        }
        if (byteSize % dstGranularity != 0) {
            throw new IllegalArgumentException(
                    "Byte size "
                            + byteSize
                            + " is not a multiple of destination granularity "
                            + dstGranularity);
        }
    }
}
