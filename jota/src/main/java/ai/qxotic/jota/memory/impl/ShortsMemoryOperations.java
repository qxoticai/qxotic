package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccessChecks;
import ai.qxotic.jota.memory.MemoryOperations;

import java.lang.foreign.MemorySegment;
import java.util.Arrays;

final class ShortsMemoryOperations implements MemoryOperations<short[]> {

    private static final ShortsMemoryOperations INSTANCE = new ShortsMemoryOperations();

    public static MemoryOperations<short[]> instance() {
        return INSTANCE;
    }

    private ShortsMemoryOperations() {
    }

    @Override
    public void copy(Memory<short[]> src, long srcByteOffset, Memory<short[]> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Short.BYTES);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Short.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
        if (byteSize == 0) {
            return;
        }
        System.arraycopy(
                src.base(), Math.toIntExact(srcByteOffset / Short.BYTES),
                dst.base(), Math.toIntExact(dstByteOffset / Short.BYTES),
                Math.toIntExact(byteSize / Short.BYTES)
        );
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<short[]> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Short.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
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
    public void copyToNative(Memory<short[]> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Short.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
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
    public void fillByte(Memory<short[]> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Short.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
        if (byteSize == 0) {
            return;
        }
        short bits = (short) (0x0101 * Byte.toUnsignedInt(byteValue));
        Arrays.fill(
                memory.base(), Math.toIntExact(byteOffset / Short.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Short.BYTES),
                bits
        );
    }

    @Override
    public void fillShort(Memory<short[]> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Short.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
        if (byteSize == 0) {
            return;
        }
        Arrays.fill(
                memory.base(), Math.toIntExact(byteOffset / Short.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Short.BYTES),
                shortValue
        );
    }

    @Override
    public void fillInt(Memory<short[]> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Integer.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }

        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        int shortsPerInt = Integer.BYTES / Short.BYTES;
        int shortOffset = intByteOffset / Short.BYTES;
        int shortSize = intByteSize / Short.BYTES;

        short low = (short) intValue;
        short high = (short) (intValue >>> 16);
        short[] base = memory.base();
        for (int i = 0; i < shortSize; i += shortsPerInt) {
            base[shortOffset + i] = low;
            base[shortOffset + i + 1] = high;
        }
    }

    @Override
    public void fillLong(Memory<short[]> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }

        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        int shortsPerLong = Long.BYTES / Short.BYTES;
        int shortOffset = intByteOffset / Short.BYTES;
        int shortSize = intByteSize / Short.BYTES;

        short s0 = (short) longValue;
        short s1 = (short) (longValue >>> 16);
        short s2 = (short) (longValue >>> 32);
        short s3 = (short) (longValue >>> 48);
        short[] base = memory.base();
        for (int i = 0; i < shortSize; i += shortsPerLong) {
            base[shortOffset + i] = s0;
            base[shortOffset + i + 1] = s1;
            base[shortOffset + i + 2] = s2;
            base[shortOffset + i + 3] = s3;
        }
    }
}
