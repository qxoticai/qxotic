package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccessChecks;
import com.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

final class DoublesMemoryOperations implements MemoryOperations<double[]> {

    private static final DoublesMemoryOperations INSTANCE = new DoublesMemoryOperations();

    public static MemoryOperations<double[]> instance() {
        return INSTANCE;
    }

    private DoublesMemoryOperations() {}

    @Override
    public void copy(
            Memory<double[]> src,
            long srcByteOffset,
            Memory<double[]> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Double.BYTES);
        if (byteSize == 0) {
            return;
        }
        System.arraycopy(
                src.base(),
                Math.toIntExact(srcByteOffset / Double.BYTES),
                dst.base(),
                Math.toIntExact(dstByteOffset / Double.BYTES),
                Math.toIntExact(byteSize / Double.BYTES));
    }

    @Override
    public void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<double[]> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Double.BYTES);
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
            Memory<double[]> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Double.BYTES);
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
    public void fillByte(Memory<double[]> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Double.BYTES);
        if (byteSize == 0) {
            return;
        }
        long bits = 0x0101010101010101L * Byte.toUnsignedInt(byteValue);
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Double.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Double.BYTES),
                Double.longBitsToDouble(bits));
    }

    @Override
    public void fillShort(
            Memory<double[]> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Double.BYTES);
        if (byteSize == 0) {
            return;
        }
        long bits = 0x0001000100010001L * Short.toUnsignedInt(shortValue);
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Double.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Double.BYTES),
                Double.longBitsToDouble(bits));
    }

    @Override
    public void fillInt(Memory<double[]> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Double.BYTES);
        if (byteSize == 0) {
            return;
        }
        long bits = (intValue & 0xFFFFFFFFL) | ((long) intValue << 32);
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Double.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Double.BYTES),
                Double.longBitsToDouble(bits));
    }

    @Override
    public void fillLong(Memory<double[]> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Double.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Double.BYTES);
        if (byteSize == 0) {
            return;
        }
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Double.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Double.BYTES),
                Double.longBitsToDouble(longValue));
    }
}
