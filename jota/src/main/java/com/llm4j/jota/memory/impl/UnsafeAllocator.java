package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.ScopedMemory;
import com.llm4j.jota.memory.ScopedMemoryAllocator;
import sun.misc.Unsafe;

import java.lang.foreign.MemorySegment;

class UnsafeAllocator implements ScopedMemoryAllocator<MemorySegment> {

    private static final Unsafe UNSAFE = UnsafeAccess.get();

    public static UnsafeAllocator INSTANCE = new UnsafeAllocator();

    public static ScopedMemoryAllocator<MemorySegment> instance() {
        return INSTANCE;
    }

    private UnsafeAllocator() {
    }

    @Override
    public Device device() {
        return Device.CPU;
    }

    @Override
    public ScopedMemory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        if (!Util.isPowerOf2(byteAlignment)) {
            throw new IllegalArgumentException("invalid byteAlignment, not a power of 2");
        }
        long mallocAddress = UNSAFE.allocateMemory(byteSize + byteAlignment - 1);
        long alignedAddress = mallocAddress;
        if (alignedAddress % byteAlignment != 0) {
            alignedAddress += byteAlignment - (alignedAddress % byteAlignment);
        }
        assert alignedAddress % byteAlignment == 0;
        return new RawScopedMemory(mallocAddress, alignedAddress, byteSize);
    }

    private static final class RawScopedMemory implements ScopedMemory<MemorySegment> {
        private final long mallocAddress;
        private final MemorySegment memorySegment;

        public RawScopedMemory(long mallocAddress, long alignedAddress, long byteSize) {
            this.mallocAddress = mallocAddress;
            this.memorySegment = MemorySegment.ofAddress(alignedAddress).reinterpret(byteSize);
        }

        @Override
        public void close() {
            UNSAFE.freeMemory(mallocAddress);
        }

        @Override
        public long byteSize() {
            return memorySegment.byteSize();
        }

        @Override
        public boolean isReadOnly() {
            return memorySegment.isReadOnly();
        }

        @Override
        public Device device() {
            return Device.CPU;
        }

        @Override
        public MemorySegment base() {
            return memorySegment;
        }
    }
}
