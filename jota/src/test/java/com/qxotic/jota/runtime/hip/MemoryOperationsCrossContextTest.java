package com.qxotic.jota.runtime.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryAllocatorFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.AutoClose;
import org.junit.jupiter.api.Test;

class MemoryOperationsCrossDomainTest {

    @AutoClose MemoryDomain<MemorySegment> nativeDomain = DomainFactory.ofMemorySegment();

    @AutoClose MemoryDomain<float[]> floatDomain = DomainFactory.ofFloats();

    @AutoClose
    MemoryDomain<ByteBuffer> bufferDomain =
            DomainFactory.ofByteBuffer(MemoryAllocatorFactory.ofByteBuffer(false));

    @Test
    void copyShouldTransferFloatBetweenDifferentDomains() {
        Memory<float[]> src = floatDomain.memoryAllocator().allocateMemory(4);
        Memory<ByteBuffer> dst = bufferDomain.memoryAllocator().allocateMemory(4);
        Memory<MemorySegment> buffer = nativeDomain.memoryAllocator().allocateMemory(4);

        floatDomain.directAccess().writeFloat(src, 0, 3.14f);

        MemoryOperations.copy(
                floatDomain.memoryOperations(),
                src,
                0,
                bufferDomain.memoryOperations(),
                dst,
                0,
                4,
                buffer);

        assertEquals(3.14f, bufferDomain.directAccess().readFloat(dst, 0));
    }

    @Test
    void copyShouldTransferIntWithOffsetBetweenDomains() {
        Memory<ByteBuffer> src = bufferDomain.memoryAllocator().allocateMemory(8);
        Memory<float[]> dst = floatDomain.memoryAllocator().allocateMemory(8);
        Memory<MemorySegment> buffer = nativeDomain.memoryAllocator().allocateMemory(4);

        bufferDomain.directAccess().writeInt(src, 4, 0x12345678);

        MemoryOperations.copy(
                bufferDomain.memoryOperations(),
                src,
                4,
                floatDomain.memoryOperations(),
                dst,
                0,
                4,
                buffer);

        assertEquals(0x12345678, floatDomain.directAccess().readInt(dst, 0));
    }

    @Test
    void copyShouldHandleChunkedTransferBetweenDomains() {
        Memory<ByteBuffer> src = bufferDomain.memoryAllocator().allocateMemory(12);
        Memory<MemorySegment> dst = nativeDomain.memoryAllocator().allocateMemory(12);
        Memory<MemorySegment> buffer = nativeDomain.memoryAllocator().allocateMemory(4);

        bufferDomain.directAccess().writeLong(src, 0, 0x1122334455667788L);

        MemoryOperations.copy(
                bufferDomain.memoryOperations(),
                src,
                0,
                nativeDomain.memoryOperations(),
                dst,
                0,
                8,
                buffer);

        assertEquals(0x1122334455667788L, nativeDomain.directAccess().readLong(dst, 0));
    }

    @Test
    void copyShouldRoundTripBetweenNativeAndHip() {
        Assumptions.assumeTrue(HipRuntime.isAvailable(), "HIP runtime not available");
        HipMemoryDomain hipDomain = HipMemoryDomain.instance();
        MemoryOperations<HipDevicePtr> hipOps = hipDomain.memoryOperations();

        Memory<MemorySegment> hostSrc = nativeDomain.memoryAllocator().allocateMemory(4);
        Memory<MemorySegment> hostDst = nativeDomain.memoryAllocator().allocateMemory(4);
        Memory<HipDevicePtr> device = hipDomain.memoryAllocator().allocateMemory(4);

        nativeDomain.directAccess().writeFloat(hostSrc, 0, 6.25f);

        hipOps.copyFromNative(hostSrc, 0, device, 0, 4);
        hipOps.copyToNative(device, 0, hostDst, 0, 4);

        assertEquals(6.25f, nativeDomain.directAccess().readFloat(hostDst, 0));
    }

    @Test
    void copyShouldTransferBetweenHipBuffers() {
        Assumptions.assumeTrue(HipRuntime.isAvailable(), "HIP runtime not available");
        HipMemoryDomain hipDomain = HipMemoryDomain.instance();
        MemoryOperations<HipDevicePtr> hipOps = hipDomain.memoryOperations();

        Memory<MemorySegment> hostSrc = nativeDomain.memoryAllocator().allocateMemory(4);
        Memory<MemorySegment> hostDst = nativeDomain.memoryAllocator().allocateMemory(4);
        Memory<HipDevicePtr> deviceSrc = hipDomain.memoryAllocator().allocateMemory(4);
        Memory<HipDevicePtr> deviceDst = hipDomain.memoryAllocator().allocateMemory(4);

        nativeDomain.directAccess().writeFloat(hostSrc, 0, -2.5f);

        hipOps.copyFromNative(hostSrc, 0, deviceSrc, 0, 4);
        hipOps.copy(deviceSrc, 0, deviceDst, 0, 4);
        hipOps.copyToNative(deviceDst, 0, hostDst, 0, 4);

        assertEquals(-2.5f, nativeDomain.directAccess().readFloat(hostDst, 0));
    }
}
