package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccessChecks;
import ai.qxotic.jota.memory.MemoryOperations;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

final class IntsMemoryOperations implements MemoryOperations<int[]> {

    private static final IntsMemoryOperations INSTANCE = new IntsMemoryOperations();

    public static MemoryOperations<int[]> instance() {
        return INSTANCE;
    }

    private IntsMemoryOperations() {
    }

    @Override
    public void copy(Memory<int[]> src, long srcByteOffset, Memory<int[]> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        System.arraycopy(
                src.base(), Math.toIntExact(srcByteOffset / Integer.BYTES),
                dst.base(), Math.toIntExact(dstByteOffset / Integer.BYTES),
                Math.toIntExact(byteSize / Integer.BYTES)
        );
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<int[]> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        MemorySegment.copy(
                src.base(), srcByteOffset,
                MemorySegment.ofArray(dst.base()), dstByteOffset,
                byteSize
        );
    }

    @Override
    public void copyToNative(Memory<int[]> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        MemorySegment.copy(
                MemorySegment.ofArray(src.base()), srcByteOffset,
                dst.base(), dstByteOffset,
                byteSize
        );
    }

    @Override
    public void fillByte(Memory<int[]> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        int bits = 0x01010101 * Byte.toUnsignedInt(byteValue);
        Arrays.fill(
                memory.base(), Math.toIntExact(byteOffset / Integer.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Integer.BYTES),
                bits
        );
    }

    @Override
    public void fillShort(Memory<int[]> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        int bits = 0x00010001 * Short.toUnsignedInt(shortValue);
        Arrays.fill(
                memory.base(), Math.toIntExact(byteOffset / Integer.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Integer.BYTES),
                bits
        );
    }

    @Override
    public void fillInt(Memory<int[]> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        Arrays.fill(
                memory.base(), Math.toIntExact(byteOffset / Integer.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Integer.BYTES),
                intValue
        );
    }

    @Override
    public void fillLong(Memory<int[]> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }

        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        int intsPerLong = Long.BYTES / Integer.BYTES;
        int intOffset = intByteOffset / Integer.BYTES;
        int intSize = intByteSize / Integer.BYTES;

        int low = (int) longValue;
        int high = (int) (longValue >> 32);
        int[] base = memory.base();
        for (int i = 0; i < intSize; i += intsPerLong) {
            base[intOffset + i] = low;
            base[intOffset + i + 1] = high;
        }
    }
}
