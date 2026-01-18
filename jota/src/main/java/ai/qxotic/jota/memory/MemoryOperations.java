package ai.qxotic.jota.memory;

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
        long copiedBytes = 0;
        while (copiedBytes < byteSize) {
            long chunkBytes = Math.min(byteSize - copiedBytes, buffer.byteSize());
            srcOps.copyToNative(src, srcByteOffset + copiedBytes, buffer, 0, chunkBytes);
            dstOps.copyFromNative(buffer, 0, dst, dstByteOffset + copiedBytes, chunkBytes);
            copiedBytes += chunkBytes;
        }
    }
}
