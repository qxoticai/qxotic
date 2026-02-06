package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.hip.HipDevicePtr;
import ai.qxotic.jota.hip.HipMemoryContext;
import ai.qxotic.jota.hip.HipRuntime;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Test;

class MemoryOperationsCrossContextTest {

    @AutoClose MemoryContext<MemorySegment> nativeContext = ContextFactory.ofMemorySegment();

    @AutoClose MemoryContext<float[]> floatContext = ContextFactory.ofFloats();

    @AutoClose
    MemoryContext<ByteBuffer> bufferContext =
            ContextFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false));

    @Test
    void copyShouldTransferFloatBetweenDifferentContexts() {
        Memory<float[]> src = floatContext.memoryAllocator().allocateMemory(4);
        Memory<ByteBuffer> dst = bufferContext.memoryAllocator().allocateMemory(4);
        Memory<MemorySegment> buffer = nativeContext.memoryAllocator().allocateMemory(4);

        floatContext.memoryAccess().writeFloat(src, 0, 3.14f);

        MemoryOperations.copy(
                floatContext.memoryOperations(),
                src,
                0,
                bufferContext.memoryOperations(),
                dst,
                0,
                4,
                buffer);

        assertEquals(3.14f, bufferContext.memoryAccess().readFloat(dst, 0));
    }

    @Test
    void copyShouldTransferIntWithOffsetBetweenContexts() {
        Memory<ByteBuffer> src = bufferContext.memoryAllocator().allocateMemory(8);
        Memory<float[]> dst = floatContext.memoryAllocator().allocateMemory(8);
        Memory<MemorySegment> buffer = nativeContext.memoryAllocator().allocateMemory(4);

        bufferContext.memoryAccess().writeInt(src, 4, 0x12345678);

        MemoryOperations.copy(
                bufferContext.memoryOperations(),
                src,
                4,
                floatContext.memoryOperations(),
                dst,
                0,
                4,
                buffer);

        assertEquals(0x12345678, floatContext.memoryAccess().readInt(dst, 0));
    }

    @Test
    void copyShouldHandleChunkedTransferBetweenContexts() {
        Memory<ByteBuffer> src = bufferContext.memoryAllocator().allocateMemory(12);
        Memory<MemorySegment> dst = nativeContext.memoryAllocator().allocateMemory(12);
        Memory<MemorySegment> buffer = nativeContext.memoryAllocator().allocateMemory(4);

        bufferContext.memoryAccess().writeLong(src, 0, 0x1122334455667788L);

        MemoryOperations.copy(
                bufferContext.memoryOperations(),
                src,
                0,
                nativeContext.memoryOperations(),
                dst,
                0,
                8,
                buffer);

        assertEquals(0x1122334455667788L, nativeContext.memoryAccess().readLong(dst, 0));
    }

    @Test
    void copyShouldRoundTripBetweenNativeAndHip() {
        Assumptions.assumeTrue(HipRuntime.isAvailable(), "HIP runtime not available");
        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryOperations<HipDevicePtr> hipOps = hipContext.memoryOperations();

        Memory<MemorySegment> hostSrc = nativeContext.memoryAllocator().allocateMemory(4);
        Memory<MemorySegment> hostDst = nativeContext.memoryAllocator().allocateMemory(4);
        Memory<HipDevicePtr> device = hipContext.memoryAllocator().allocateMemory(4);

        nativeContext.memoryAccess().writeFloat(hostSrc, 0, 6.25f);

        hipOps.copyFromNative(hostSrc, 0, device, 0, 4);
        hipOps.copyToNative(device, 0, hostDst, 0, 4);

        assertEquals(6.25f, nativeContext.memoryAccess().readFloat(hostDst, 0));
    }

    @Test
    void copyShouldTransferBetweenHipBuffers() {
        Assumptions.assumeTrue(HipRuntime.isAvailable(), "HIP runtime not available");
        HipMemoryContext hipContext = HipMemoryContext.instance();
        MemoryOperations<HipDevicePtr> hipOps = hipContext.memoryOperations();

        Memory<MemorySegment> hostSrc = nativeContext.memoryAllocator().allocateMemory(4);
        Memory<MemorySegment> hostDst = nativeContext.memoryAllocator().allocateMemory(4);
        Memory<HipDevicePtr> deviceSrc = hipContext.memoryAllocator().allocateMemory(4);
        Memory<HipDevicePtr> deviceDst = hipContext.memoryAllocator().allocateMemory(4);

        nativeContext.memoryAccess().writeFloat(hostSrc, 0, -2.5f);

        hipOps.copyFromNative(hostSrc, 0, deviceSrc, 0, 4);
        hipOps.copy(deviceSrc, 0, deviceDst, 0, 4);
        hipOps.copyToNative(deviceDst, 0, hostDst, 0, 4);

        assertEquals(-2.5f, nativeContext.memoryAccess().readFloat(hostDst, 0));
    }
}
