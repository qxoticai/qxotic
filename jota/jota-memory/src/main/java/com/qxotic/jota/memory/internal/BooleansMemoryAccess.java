package com.qxotic.jota.memory.internal;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAccessChecks;
import sun.misc.Unsafe;

/**
 * A {@code boolean[]} only holds BOOL (see {@link ArrayMemory#supportsDataType}), so only byte
 * access is defined: the JVM leaves a boolean holding a value other than 0/1 undefined, which is
 * what any wider Unsafe write would create.
 */
final class BooleansMemoryAccess implements MemoryAccess<boolean[]> {

    private static final Unsafe UNSAFE = UnsafeAccess.get();
    private static final BooleansMemoryAccess INSTANCE = new BooleansMemoryAccess();

    public static MemoryAccess<boolean[]> instance() {
        return INSTANCE;
    }

    private BooleansMemoryAccess() {}

    @Override
    public byte readByte(Memory<boolean[]> memory, long byteOffset) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, Byte.BYTES);
        boolean value =
                UNSAFE.getBoolean(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset);
        return (byte) (value ? 1 : 0);
    }

    @Override
    public void writeByte(Memory<boolean[]> memory, long byteOffset, byte value) {
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkBounds(memory, byteOffset, Byte.BYTES);
        UNSAFE.putBoolean(memory.base(), Unsafe.ARRAY_BOOLEAN_BASE_OFFSET + byteOffset, value != 0);
    }

    @Override
    public short readShort(Memory<boolean[]> memory, long byteOffset) {
        throw new UnsupportedOperationException("boolean[] memory is byte-addressable only");
    }

    @Override
    public int readInt(Memory<boolean[]> memory, long byteOffset) {
        throw new UnsupportedOperationException("boolean[] memory is byte-addressable only");
    }

    @Override
    public long readLong(Memory<boolean[]> memory, long byteOffset) {
        throw new UnsupportedOperationException("boolean[] memory is byte-addressable only");
    }

    @Override
    public void writeShort(Memory<boolean[]> memory, long byteOffset, short value) {
        throw new UnsupportedOperationException("boolean[] memory is byte-addressable only");
    }

    @Override
    public void writeInt(Memory<boolean[]> memory, long byteOffset, int value) {
        throw new UnsupportedOperationException("boolean[] memory is byte-addressable only");
    }

    @Override
    public void writeLong(Memory<boolean[]> memory, long byteOffset, long value) {
        throw new UnsupportedOperationException("boolean[] memory is byte-addressable only");
    }
}
