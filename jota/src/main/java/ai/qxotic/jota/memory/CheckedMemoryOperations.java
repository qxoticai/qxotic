package ai.qxotic.jota.memory;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

public final class CheckedMemoryOperations<B> implements MemoryOperations<B> {

    private final MemoryOperations<B> delegate;

    public CheckedMemoryOperations(MemoryOperations<B> delegate) {
        this.delegate = Objects.requireNonNull(delegate);
    }

    @Override
    public void copy(Memory<B> src, long srcByteOffset, Memory<B> dst, long dstByteOffset, long byteSize) {
        checkBounds(src, srcByteOffset, byteSize);
        checkBounds(dst, dstByteOffset, byteSize);
        delegate.copy(src, srcByteOffset, dst, dstByteOffset, byteSize);
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<B> dst, long dstByteOffset, long byteSize) {
        checkBounds(src, srcByteOffset, byteSize);
        checkBounds(dst, dstByteOffset, byteSize);
        delegate.copyFromNative(src, srcByteOffset, dst, dstByteOffset, byteSize);
    }

    @Override
    public void copyToNative(Memory<B> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        checkBounds(src, srcByteOffset, byteSize);
        checkBounds(dst, dstByteOffset, byteSize);
        delegate.copyToNative(src, srcByteOffset, dst, dstByteOffset, byteSize);
    }

    @Override
    public void fillByte(Memory<B> memory, long byteOffset, long byteSize, byte byteValue) {
        checkBounds(memory, byteOffset, byteSize);
        delegate.fillByte(memory, byteOffset, byteSize, byteValue);
    }

    @Override
    public void fillShort(Memory<B> memory, long byteOffset, long byteSize, short shortValue) {
        checkBounds(memory, byteOffset, byteSize);
        checkAligned(byteOffset, Short.BYTES);
        checkAligned(byteSize, Short.BYTES);
        delegate.fillShort(memory, byteOffset, byteSize, shortValue);
    }

    @Override
    public void fillInt(Memory<B> memory, long byteOffset, long byteSize, int intValue) {
        checkBounds(memory, byteOffset, byteSize);
        checkAligned(byteOffset, Integer.BYTES);
        checkAligned(byteSize, Integer.BYTES);
        delegate.fillInt(memory, byteOffset, byteSize, intValue);
    }

    @Override
    public void fillLong(Memory<B> memory, long byteOffset, long byteSize, long longValue) {
        checkBounds(memory, byteOffset, byteSize);
        checkAligned(byteOffset, Long.BYTES);
        checkAligned(byteSize, Long.BYTES);
        delegate.fillLong(memory, byteOffset, byteSize, longValue);
    }

    @Override
    public void fillFloat(Memory<B> memory, long byteOffset, long byteSize, float floatValue) {
        checkBounds(memory, byteOffset, byteSize);
        checkAligned(byteOffset, Float.BYTES);
        checkAligned(byteSize, Float.BYTES);
        delegate.fillFloat(memory, byteOffset, byteSize, floatValue);
    }

    @Override
    public void fillDouble(Memory<B> memory, long byteOffset, long byteSize, double doubleValue) {
        checkBounds(memory, byteOffset, byteSize);
        checkAligned(byteOffset, Double.BYTES);
        checkAligned(byteSize, Double.BYTES);
        delegate.fillDouble(memory, byteOffset, byteSize, doubleValue);
    }

    private void checkBounds(Memory<?> memory, long byteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(byteOffset >= 0, "negative byte offset");
        MemoryAccessChecks.checkBounds(byteSize >= 0, "negative byte size");
        MemoryAccessChecks.checkBounds(byteOffset + byteSize <= memory.byteSize(), "out of bounds access");
    }

    private void checkAligned(long value, long alignment) {
        if (alignment <= 1) {
            return;
        }
        MemoryAccessChecks.checkAlignment((value & (alignment - 1)) == 0, "unaligned access");
    }
}
