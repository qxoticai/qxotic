package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccessChecks;
import ai.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;

final class ByteBufferMemoryOperations implements MemoryOperations<ByteBuffer> {

    private static final ByteBufferMemoryOperations INSTANCE = new ByteBufferMemoryOperations();

    public static MemoryOperations<ByteBuffer> instance() {
        return INSTANCE;
    }

    private ByteBufferMemoryOperations() {}

    @Override
    public void copy(
            Memory<ByteBuffer> src,
            long srcByteOffset,
            Memory<ByteBuffer> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        if (byteSize == 0) {
            return;
        }
        int intSrcByteOffset = Math.toIntExact(srcByteOffset);
        int intDstByteOffset = Math.toIntExact(dstByteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        dst.base().put(intDstByteOffset, src.base(), intSrcByteOffset, intByteSize);
    }

    @Override
    public void copyFromNative(
            Memory<MemorySegment> src,
            long srcByteOffset,
            Memory<ByteBuffer> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        if (byteSize == 0) {
            return;
        }
        int intDstByteOffset = Math.toIntExact(dstByteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; ++i) {
            byte byteValue = src.base().get(ValueLayout.JAVA_BYTE, srcByteOffset + i);
            dst.base().put(intDstByteOffset + i, byteValue);
        }
    }

    @Override
    public void copyToNative(
            Memory<ByteBuffer> src,
            long srcByteOffset,
            Memory<MemorySegment> dst,
            long dstByteOffset,
            long byteSize) {
        MemoryAccessChecks.checkBounds(src, srcByteOffset, byteSize);
        MemoryAccessChecks.checkBounds(dst, dstByteOffset, byteSize);
        if (byteSize == 0) {
            return;
        }
        int intSrcByteOffset = Math.toIntExact(srcByteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; ++i) {
            byte byteValue = src.base().get(intSrcByteOffset + i);
            dst.base().set(ValueLayout.JAVA_BYTE, dstByteOffset + i, byteValue);
        }
    }

    @Override
    public void fillByte(
            Memory<ByteBuffer> memory, long byteOffset, long byteSize, byte byteValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        if (byteSize == 0) {
            return;
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; ++i) {
            memory.base().put(intByteOffset + i, byteValue);
        }
    }

    @Override
    public void fillShort(
            Memory<ByteBuffer> memory, long byteOffset, long byteSize, short shortValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Short.BYTES);
        if (byteSize == 0) {
            return;
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Short.BYTES) {
            memory.base().putShort(intByteOffset + i, shortValue);
        }
    }

    @Override
    public void fillInt(Memory<ByteBuffer> memory, long byteOffset, long byteSize, int intValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Integer.BYTES);
        if (byteSize == 0) {
            return;
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Integer.BYTES) {
            memory.base().putInt(intByteOffset + i, intValue);
        }
    }

    @Override
    public void fillLong(
            Memory<ByteBuffer> memory, long byteOffset, long byteSize, long longValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Long.BYTES);
        if (byteSize == 0) {
            return;
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Long.BYTES) {
            memory.base().putLong(intByteOffset + i, longValue);
        }
    }

    @Override
    public void fillFloat(
            Memory<ByteBuffer> memory, long byteOffset, long byteSize, float floatValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Float.BYTES);
        if (byteSize == 0) {
            return;
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Float.BYTES) {
            memory.base().putFloat(intByteOffset + i, floatValue);
        }
    }

    @Override
    public void fillDouble(
            Memory<ByteBuffer> memory, long byteOffset, long byteSize, double doubleValue) {
        MemoryAccessChecks.checkBounds(memory, byteOffset, byteSize);
        MemoryAccessChecks.checkWriteable(memory);
        MemoryAccessChecks.checkAlignedValue(byteSize, Double.BYTES);
        if (byteSize == 0) {
            return;
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Double.BYTES) {
            memory.base().putDouble(intByteOffset + i, doubleValue);
        }
    }
}
