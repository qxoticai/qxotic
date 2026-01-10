package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryOperations;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

final class FloatsMemoryOperations implements MemoryOperations<float[]> {

    private static final FloatsMemoryOperations INSTANCE = new FloatsMemoryOperations();

    public static MemoryOperations<float[]> instance() {
        return INSTANCE;
    }

    @Override
    public void copy(Memory<float[]> src, long srcByteOffset, Memory<float[]> dst, long dstByteOffset, long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (srcByteOffset < 0) {
            throw new IllegalArgumentException("negative src offset");
        }
        if (dstByteOffset < 0) {
            throw new IllegalArgumentException("negative dst offset");
        }
        if (srcByteOffset % Float.BYTES != 0 || dstByteOffset % Float.BYTES != 0 || byteSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }
        System.arraycopy(
                src.base(), Math.toIntExact(srcByteOffset / Float.BYTES),
                dst.base(), Math.toIntExact(dstByteOffset / Float.BYTES),
                Math.toIntExact(byteSize / Float.BYTES)
        );
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<float[]> dst, long dstByteOffset, long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (srcByteOffset < 0) {
            throw new IllegalArgumentException("negative src offset");
        }
        if (dstByteOffset < 0) {
            throw new IllegalArgumentException("negative dst offset");
        }
        if (dstByteOffset % Float.BYTES != 0 || byteSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }
        MemorySegment.copy(
                src.base(), srcByteOffset,
                MemorySegment.ofArray(dst.base()), dstByteOffset,
                byteSize
        );
    }

    @Override
    public void copyToNative(Memory<float[]> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (srcByteOffset < 0) {
            throw new IllegalArgumentException("negative src offset");
        }
        if (dstByteOffset < 0) {
            throw new IllegalArgumentException("negative dst offset");
        }
        if (srcByteOffset % Float.BYTES != 0 || byteSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }
        MemorySegment.copy(
                MemorySegment.ofArray(src.base()), srcByteOffset,
                dst.base(), dstByteOffset,
                byteSize
        );
    }

    @Override
    public void fillByte(Memory<float[]> memory, long byteOffset, long byteSize, byte byteValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteOffset % Float.BYTES != 0 || byteSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }
        int bits = 0x01010101 * Byte.toUnsignedInt(byteValue);
        Arrays.fill(
                memory.base(), Math.toIntExact(byteOffset / Float.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Float.BYTES),
                Float.intBitsToFloat(bits)
        );
    }

    @Override
    public void fillShort(Memory<float[]> memory, long byteOffset, long byteSize, short shortValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteOffset % Float.BYTES != 0 || byteSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }
        int bits = 0x00010001 * Short.toUnsignedInt(shortValue);
        Arrays.fill(
                memory.base(), Math.toIntExact(byteOffset / Float.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Float.BYTES),
                Float.intBitsToFloat(bits)
        );
    }

    @Override
    public void fillInt(Memory<float[]> memory, long byteOffset, long byteSize, int intValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteOffset % Float.BYTES != 0 || byteSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }
        Arrays.fill(
                memory.base(), Math.toIntExact(byteOffset / Float.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Float.BYTES),
                Float.intBitsToFloat(intValue)
        );
    }

    @Override
    public void fillLong(Memory<float[]> memory, long byteOffset, long byteSize, long longValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteOffset % Long.BYTES != 0 || byteSize % Long.BYTES != 0) {
            throw new IllegalArgumentException("unaligned access");
        }

        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        int floatsPerLong = Long.BYTES / Float.BYTES;
        int intFloatOffset = intByteOffset / Float.BYTES;
        int intFloatSize = intByteSize / Float.BYTES;

        float f0 = Float.intBitsToFloat((int) (longValue & 0xFFFFFF));
        float f1 = Float.intBitsToFloat((int) ((longValue >> 32) & 0xFFFFFF));
        float[] base = memory.base();
        for (int i = 0; i < intFloatSize; i += floatsPerLong) {
            base[intFloatOffset + i] = f0;
            base[intFloatOffset + i + 1] = f1;
        }
    }
}
