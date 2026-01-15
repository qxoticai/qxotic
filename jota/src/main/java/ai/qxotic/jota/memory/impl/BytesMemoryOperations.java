package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccessChecks;
import ai.qxotic.jota.memory.MemoryOperations;

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
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        if (byteSize == 0) {
            return;
        }
        System.arraycopy(
                src.base(), Math.toIntExact(srcByteOffset),
                dst.base(), Math.toIntExact(dstByteOffset),
                Math.toIntExact(byteSize)
        );
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<byte[]> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
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
    public void copyToNative(Memory<byte[]> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
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
    public void fillByte(Memory<byte[]> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        Arrays.fill(memory.base(), intByteOffset, intByteOffset + intByteSize, byteValue);
    }

    @Override
    public void fillShort(Memory<byte[]> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
        if (byteSize == 0) {
            return;
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
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
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
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
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
