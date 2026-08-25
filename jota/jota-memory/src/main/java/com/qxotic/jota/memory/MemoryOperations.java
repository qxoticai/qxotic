package com.qxotic.jota.memory;

import com.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public interface MemoryOperations<B> {

    void copy(Memory<B> src, long srcByteOffset, Memory<B> dst, long dstByteOffset, long byteSize);

    /**
     * Transfers from a {@link MemorySegment}, heap ({@code MemorySegment.ofArray}) or native, into
     * this backend. Implementations that hand addresses to a driver must check {@code isNative()}
     * and stage heap segments themselves.
     */
    void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<B> dst,
            long dstByteOffset,
            long byteSize);

    /** The mirror of {@link #copyFromNative}: this backend into a heap or native segment. */
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
        long copyGranularity = checkGranularity(src, srcByteOffset, dst, dstByteOffset, byteSize);
        if (buffer.byteSize() < copyGranularity || buffer.byteSize() % copyGranularity != 0) {
            throw new IllegalArgumentException(
                    "Staging buffer size must be a positive multiple of " + copyGranularity);
        }
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
        long copyGranularity = checkGranularity(src, srcByteOffset, dst, dstByteOffset, byteSize);
        // One side already a MemorySegment (heap or native): the backend primitive is the direct
        // transfer, so no staging buffer and no chunking.
        if (dst.base() instanceof MemorySegment) {
            @SuppressWarnings("unchecked")
            Memory<MemorySegment> segment = (Memory<MemorySegment>) dst;
            srcOps.copyToNative(src, srcByteOffset, segment, dstByteOffset, byteSize);
            return;
        }
        if (src.base() instanceof MemorySegment) {
            @SuppressWarnings("unchecked")
            Memory<MemorySegment> segment = (Memory<MemorySegment>) src;
            dstOps.copyFromNative(segment, srcByteOffset, dst, dstByteOffset, byteSize);
            return;
        }
        long bufferSize = computeBufferSize(byteSize, copyGranularity);
        try (var arena = Arena.ofConfined()) {
            MemorySegment memorySegment = arena.allocate(bufferSize, 4 << 10);
            Memory<MemorySegment> memory = MemoryFactory.ofMemorySegment(memorySegment);
            copy(srcOps, src, srcByteOffset, dstOps, dst, dstByteOffset, byteSize, memory);
        }
    }

    private static long computeBufferSize(long byteSize, long copyGranularity) {
        long chunkSize = 4 << 10; // 4KB
        long sizeAlignment = leastCommonMultiple(chunkSize, copyGranularity);
        int log2 = 64 - Long.numberOfLeadingZeros(byteSize / chunkSize + 1);
        long bufferSize = Math.max(chunkSize, byteSize / Math.max(1, log2));
        return Math.max(sizeAlignment, bufferSize - bufferSize % sizeAlignment);
    }

    private static long leastCommonMultiple(long left, long right) {
        long a = left;
        long b = right;
        while (b != 0) {
            long remainder = a % b;
            a = b;
            b = remainder;
        }
        return Math.multiplyExact(left / a, right);
    }

    private static <S, T> long checkGranularity(
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
        return leastCommonMultiple(srcGranularity, dstGranularity);
    }
}
