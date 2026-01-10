package com.qxotic.jota.memory;

import com.qxotic.jota.memory.impl.ContextFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import com.qxotic.jota.memory.Memory;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Test;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;

class MemoryOperationsCrossContextTest {

    @AutoClose
    Context<MemorySegment> nativeContext = ContextFactory.ofMemorySegment();

    @AutoClose
    Context<float[]> floatContext = ContextFactory.ofFloats();

    @AutoClose
    Context<ByteBuffer> bufferContext = ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false));

    @Test
    void copyShouldTransferFloatBetweenDifferentContexts() {
        Memory<float[]> src = floatContext.memoryAllocator().allocateMemory(4);
        Memory<ByteBuffer> dst = bufferContext.memoryAllocator().allocateMemory(4);
        Memory<MemorySegment> buffer = nativeContext.memoryAllocator().allocateMemory(4);

        floatContext.memoryAccess().writeFloat(src, 0, 3.14f);

        MemoryOperations.copy(
                floatContext.memoryOperations(), src, 0,
                bufferContext.memoryOperations(), dst, 0,
                4, buffer);

        assertEquals(3.14f, bufferContext.memoryAccess().readFloat(dst, 0));
    }

    @Test
    void copyShouldTransferIntWithOffsetBetweenContexts() {
        Memory<ByteBuffer> src = bufferContext.memoryAllocator().allocateMemory(8);
        Memory<float[]> dst = floatContext.memoryAllocator().allocateMemory(8);
        Memory<MemorySegment> buffer = nativeContext.memoryAllocator().allocateMemory(4);

        bufferContext.memoryAccess().writeInt(src, 4, 0x12345678);

        MemoryOperations.copy(
                bufferContext.memoryOperations(), src, 4,
                floatContext.memoryOperations(), dst, 0,
                4, buffer
        );

        assertEquals(0x12345678, floatContext.memoryAccess().readInt(dst, 0));
    }

    @Test
    void copyShouldHandleChunkedTransferBetweenContexts() {
        Memory<ByteBuffer> src = bufferContext.memoryAllocator().allocateMemory(12);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(12);
        Memory<MemorySegment> buffer = nativeContext.memoryAllocator().allocateMemory(4);

        bufferContext.memoryAccess().writeLong(src, 0, 0x1122334455667788L);

        MemoryOperations.copy(
                bufferContext.memoryOperations(), src, 0,
                nativeContext.memoryOperations(), dst, 0,
                8, buffer
        );

        assertEquals(0x1122334455667788L, nativeContext.memoryAccess().readLong(dst, 0));
    }
}
