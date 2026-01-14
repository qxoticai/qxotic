package com.qxotic.jota.memory;

import java.util.Objects;

public final class CheckedMemoryAccess<B> implements MemoryAccess<B> {

    private final MemoryAccess<B> delegate;

    public CheckedMemoryAccess(MemoryAccess<B> delegate) {
        this.delegate = Objects.requireNonNull(delegate);
    }

    @Override
    public byte readByte(Memory<B> memory, long byteOffset) {
        checkBounds(memory, byteOffset, Byte.BYTES);
        return delegate.readByte(memory, byteOffset);
    }

    @Override
    public short readShort(Memory<B> memory, long byteOffset) {
        checkBounds(memory, byteOffset, Short.BYTES);
        checkAligned(byteOffset, Short.BYTES);
        return delegate.readShort(memory, byteOffset);
    }

    @Override
    public int readInt(Memory<B> memory, long byteOffset) {
        checkBounds(memory, byteOffset, Integer.BYTES);
        checkAligned(byteOffset, Integer.BYTES);
        return delegate.readInt(memory, byteOffset);
    }

    @Override
    public float readFloat(Memory<B> memory, long byteOffset) {
        checkBounds(memory, byteOffset, Float.BYTES);
        checkAligned(byteOffset, Float.BYTES);
        return delegate.readFloat(memory, byteOffset);
    }

    @Override
    public long readLong(Memory<B> memory, long byteOffset) {
        checkBounds(memory, byteOffset, Long.BYTES);
        checkAligned(byteOffset, Long.BYTES);
        return delegate.readLong(memory, byteOffset);
    }

    @Override
    public double readDouble(Memory<B> memory, long byteOffset) {
        checkBounds(memory, byteOffset, Double.BYTES);
        checkAligned(byteOffset, Double.BYTES);
        return delegate.readDouble(memory, byteOffset);
    }

    @Override
    public void writeByte(Memory<B> memory, long byteOffset, byte value) {
        checkWriteable(memory);
        checkBounds(memory, byteOffset, Byte.BYTES);
        delegate.writeByte(memory, byteOffset, value);
    }

    @Override
    public void writeShort(Memory<B> memory, long byteOffset, short value) {
        checkWriteable(memory);
        checkBounds(memory, byteOffset, Short.BYTES);
        checkAligned(byteOffset, Short.BYTES);
        delegate.writeShort(memory, byteOffset, value);
    }

    @Override
    public void writeInt(Memory<B> memory, long byteOffset, int value) {
        checkWriteable(memory);
        checkBounds(memory, byteOffset, Integer.BYTES);
        checkAligned(byteOffset, Integer.BYTES);
        delegate.writeInt(memory, byteOffset, value);
    }

    @Override
    public void writeFloat(Memory<B> memory, long byteOffset, float value) {
        checkWriteable(memory);
        checkBounds(memory, byteOffset, Float.BYTES);
        checkAligned(byteOffset, Float.BYTES);
        delegate.writeFloat(memory, byteOffset, value);
    }

    @Override
    public void writeLong(Memory<B> memory, long byteOffset, long value) {
        checkWriteable(memory);
        checkBounds(memory, byteOffset, Long.BYTES);
        checkAligned(byteOffset, Long.BYTES);
        delegate.writeLong(memory, byteOffset, value);
    }

    @Override
    public void writeDouble(Memory<B> memory, long byteOffset, double value) {
        checkWriteable(memory);
        checkBounds(memory, byteOffset, Double.BYTES);
        checkAligned(byteOffset, Double.BYTES);
        delegate.writeDouble(memory, byteOffset, value);
    }

    private void checkBounds(Memory<B> memory, long byteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(byteOffset >= 0, "negative byte offset");
        MemoryAccessChecks.checkBounds(byteSize >= 0, "negative byte size");
        MemoryAccessChecks.checkBounds(byteOffset + byteSize <= memory.byteSize(), "out of bounds access");
    }

    private void checkWriteable(Memory<B> memory) {
        MemoryAccessChecks.checkReadOnly(!memory.isReadOnly(), "memory is read-only");
    }

    private void checkAligned(long byteOffset, long byteSize) {
        if (byteSize <= 1) {
            return;
        }
        MemoryAccessChecks.checkAlignment((byteOffset & (byteSize - 1)) == 0, "unaligned access");
    }
}
