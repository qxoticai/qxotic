package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccessChecks;
import com.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

final class FloatsMemoryOperations implements MemoryOperations<float[]> {

    private static final FloatsMemoryOperations INSTANCE = new FloatsMemoryOperations();

    public static MemoryOperations<float[]> instance() {
        return INSTANCE;
    }

    @Override
    public void copy(
            Memory<float[]> src,
            long srcByteOffset,
            Memory<float[]> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Float.BYTES);
        if (byteSize == 0) {
            return;
        }
        System.arraycopy(
                src.base(),
                Math.toIntExact(srcByteOffset / Float.BYTES),
                dst.base(),
                Math.toIntExact(dstByteOffset / Float.BYTES),
                Math.toIntExact(byteSize / Float.BYTES));
    }

    @Override
    public void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<float[]> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Float.BYTES);
        if (byteSize == 0) {
            return;
        }
        MemorySegment.copy(
                src.base(),
                srcByteOffset,
                MemorySegment.ofArray(dst.base()),
                dstByteOffset,
                byteSize);
    }

    @Override
    public void copyToNative(
            Memory<float[]> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Float.BYTES);
        if (byteSize == 0) {
            return;
        }
        MemorySegment.copy(
                MemorySegment.ofArray(src.base()),
                srcByteOffset,
                dst.base(),
                dstByteOffset,
                byteSize);
    }

    @Override
    public void fillByte(Memory<float[]> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Float.BYTES);
        if (byteSize == 0) {
            return;
        }
        int bits = 0x01010101 * Byte.toUnsignedInt(byteValue);
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Float.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Float.BYTES),
                Float.intBitsToFloat(bits));
    }

    @Override
    public void fillShort(
            Memory<float[]> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Float.BYTES);
        if (byteSize == 0) {
            return;
        }
        int bits = 0x00010001 * Short.toUnsignedInt(shortValue);
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Float.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Float.BYTES),
                Float.intBitsToFloat(bits));
    }

    @Override
    public void fillInt(Memory<float[]> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Float.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Float.BYTES);
        if (byteSize == 0) {
            return;
        }
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Float.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Float.BYTES),
                Float.intBitsToFloat(intValue));
    }

    @Override
    public void fillLong(Memory<float[]> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
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
