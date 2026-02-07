package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccessChecks;
import ai.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

final class LongsMemoryOperations implements MemoryOperations<long[]> {

    private static final LongsMemoryOperations INSTANCE = new LongsMemoryOperations();

    public static MemoryOperations<long[]> instance() {
        return INSTANCE;
    }

    private LongsMemoryOperations() {}

    @Override
    public void copy(
            Memory<long[]> src,
            long srcByteOffset,
            Memory<long[]> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }
        System.arraycopy(
                src.base(),
                Math.toIntExact(srcByteOffset / Long.BYTES),
                dst.base(),
                Math.toIntExact(dstByteOffset / Long.BYTES),
                Math.toIntExact(byteSize / Long.BYTES));
    }

    @Override
    public void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<long[]> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(dstByteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
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
            Memory<long[]> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        MemoryAccessChecks.checkAlignedValue(srcByteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
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
    public void fillByte(Memory<long[]> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }
        long bits = 0x0101010101010101L * Byte.toUnsignedInt(byteValue);
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Long.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Long.BYTES),
                bits);
    }

    @Override
    public void fillShort(Memory<long[]> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }
        long bits = 0x0001000100010001L * Short.toUnsignedInt(shortValue);
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Long.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Long.BYTES),
                bits);
    }

    @Override
    public void fillInt(Memory<long[]> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }
        long bits = (intValue & 0xFFFFFFFFL) | ((long) intValue << 32);
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Long.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Long.BYTES),
                bits);
    }

    @Override
    public void fillLong(Memory<long[]> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteOffset, Long.BYTES);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }
        Arrays.fill(
                memory.base(),
                Math.toIntExact(byteOffset / Long.BYTES),
                Math.toIntExact((byteOffset + byteSize) / Long.BYTES),
                longValue);
    }
}
