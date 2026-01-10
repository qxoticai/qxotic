package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryOperations;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;

final class ByteBufferMemoryOperations implements MemoryOperations<ByteBuffer> {

    private static final ByteBufferMemoryOperations INSTANCE = new ByteBufferMemoryOperations();

    public static MemoryOperations<ByteBuffer> instance() {
        return INSTANCE;
    }

    private ByteBufferMemoryOperations() {
    }

    @Override
    public void copy(Memory<ByteBuffer> src, long srcByteOffset, Memory<ByteBuffer> dst, long dstByteOffset, long byteSize) {
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
        int intSrcByteOffset = Math.toIntExact(srcByteOffset);
        int intDstByteOffset = Math.toIntExact(dstByteOffset);
        int intByteSize = Math.toIntExact(byteSize);

        dst.base().put(intDstByteOffset, src.base(), intSrcByteOffset, intByteSize);
    }

    @Override
    public void copyFromNative(Memory<MemorySegment> src, long srcByteOffset, Memory<ByteBuffer> dst, long dstByteOffset, long byteSize) {
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
        int intDstByteOffset = Math.toIntExact(dstByteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; ++i) {
            byte byteValue = src.base().get(ValueLayout.JAVA_BYTE, srcByteOffset + i);
            dst.base().put(intDstByteOffset + i, byteValue);
        }
    }

    @Override
    public void copyToNative(Memory<ByteBuffer> src, long srcByteOffset, Memory<MemorySegment> dst, long dstByteOffset, long byteSize) {
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
        int intSrcByteOffset = Math.toIntExact(srcByteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; ++i) {
            byte byteValue = src.base().get(intSrcByteOffset + i);
            dst.base().set(ValueLayout.JAVA_BYTE, dstByteOffset + i, byteValue);
        }
    }

    @Override
    public void fillByte(Memory<ByteBuffer> memory, long byteOffset, long byteSize, byte byteValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; ++i) {
            memory.base().put(intByteOffset + i, byteValue);
        }
    }

    @Override
    public void fillShort(Memory<ByteBuffer> memory, long byteOffset, long byteSize, short shortValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteSize % Short.BYTES != 0) {
            throw new IllegalArgumentException("unaligned byteSize");
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Short.BYTES) {
            memory.base().putShort(intByteOffset + i, shortValue);
        }
    }

    @Override
    public void fillInt(Memory<ByteBuffer> memory, long byteOffset, long byteSize, int intValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteSize % Integer.BYTES != 0) {
            throw new IllegalArgumentException("unaligned byteSize");
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Integer.BYTES) {
            memory.base().putInt(intByteOffset + i, intValue);
        }
    }

    @Override
    public void fillLong(Memory<ByteBuffer> memory, long byteOffset, long byteSize, long longValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteSize % Long.BYTES != 0) {
            throw new IllegalArgumentException("unaligned byteSize");
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Long.BYTES) {
            memory.base().putLong(intByteOffset + i, longValue);
        }
    }

    @Override
    public void fillFloat(Memory<ByteBuffer> memory, long byteOffset, long byteSize, float floatValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("unaligned byteSize");
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Float.BYTES) {
            memory.base().putFloat(intByteOffset + i, floatValue);
        }
    }

    @Override
    public void fillDouble(Memory<ByteBuffer> memory, long byteOffset, long byteSize, double doubleValue) {
        if (byteSize == 0) {
            return;
        }
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative size");
        }
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        if (byteSize % Double.BYTES != 0) {
            throw new IllegalArgumentException("unaligned byteSize");
        }
        int intByteOffset = Math.toIntExact(byteOffset);
        int intByteSize = Math.toIntExact(byteSize);
        for (int i = 0; i < intByteSize; i += Double.BYTES) {
            memory.base().putDouble(intByteOffset + i, doubleValue);
        }
    }

}
