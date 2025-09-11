package com.llm4j.jota.memory.impl;

import com.llm4j.jota.memory.Memory;
import com.llm4j.jota.memory.MemoryOperations;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

class BytesMemoryOperations implements MemoryOperations<byte[]> {

    private static final BytesMemoryOperations INSTANCE = new BytesMemoryOperations();

    public static MemoryOperations<byte[]> instance() {
        return INSTANCE;
    }

    private BytesMemoryOperations() {
    }

    @Override
    public void copy(Memory<byte[]> src, long srcByteOffset, Memory<byte[]> dst, long dstByteOffset, long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        System.arraycopy(
                src.base(), Math.toIntExact(srcByteOffset),
                dst.base(), Math.toIntExact(dstByteOffset),
                Math.toIntExact(byteSize)
        );
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<byte[]> dst, long dstByteOffset, long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        MemorySegment.copy(
                src.base(), srcByteOffset,
                MemorySegment.ofArray(dst.base()), dstByteOffset,
                byteSize
        );
    }

    @Override
    public void copyToNative(Memory<byte[]> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        MemorySegment.copy(
                MemorySegment.ofArray(src.base()), srcByteOffset,
                dst.base(), dstByteOffset,
                byteSize
        );
    }

    @Override
    public void fillByte(Memory<byte[]> memory, long byteOffset, long byteSize, byte byteValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        Arrays.fill(memory.base(), intByteOffset, intByteOffset + intByteSize, byteValue);
    }

    @Override
    public void fillShort(Memory<byte[]> memory, long byteOffset, long byteSize, short shortValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteSize % Short.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }

        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        byte b0 = (byte) (shortValue & 0xFF);
        byte b1 = (byte) ((shortValue >> 8) & 0xFF);
        byte[] base = memory.base();
        for (int i = 0; i < intByteSize; i += Short.BYTES) {
            base[intByteOffset + i] = b0;
            base[intByteOffset + i + 1] = b1;
        }
    }

    @Override
    public void fillInt(Memory<byte[]> memory, long byteOffset, long byteSize, int intValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteSize % Integer.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }

        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        byte b0 = (byte) (intValue & 0xFF);
        byte b1 = (byte) ((intValue >> 8) & 0xFF);
        byte b2 = (byte) ((intValue >> 16) & 0xFF);
        byte b3 = (byte) ((intValue >> 24) & 0xFF);
        byte[] base = memory.base();
        for (int i = 0; i < intByteSize; i += Integer.BYTES) {
            base[intByteOffset + i] = b0;
            base[intByteOffset + i + 1] = b1;
            base[intByteOffset + i + 2] = b2;
            base[intByteOffset + i + 3] = b3;
        }
    }

    @Override
    public void fillLong(Memory<byte[]> memory, long byteOffset, long byteSize, long longValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteSize % Long.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }

        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        byte b0 = (byte) (longValue & 0xFF);
        byte b1 = (byte) ((longValue >> 8) & 0xFF);
        byte b2 = (byte) ((longValue >> 16) & 0xFF);
        byte b3 = (byte) ((longValue >> 24) & 0xFF);
        byte b4 = (byte) ((longValue >> 32) & 0xFF);
        byte b5 = (byte) ((longValue >> 40) & 0xFF);
        byte b6 = (byte) ((longValue >> 48) & 0xFF);
        byte b7 = (byte) ((longValue >> 56) & 0xFF);
        byte[] base = memory.base();
        for (int i = 0; i < intByteSize; i += Long.BYTES) {
            base[intByteOffset + i] = b0;
            base[intByteOffset + i + 1] = b1;
            base[intByteOffset + i + 2] = b2;
            base[intByteOffset + i + 3] = b3;
            base[intByteOffset + i + 4] = b4;
            base[intByteOffset + i + 5] = b5;
            base[intByteOffset + i + 6] = b6;
            base[intByteOffset + i + 7] = b7;
        }
    }
}
