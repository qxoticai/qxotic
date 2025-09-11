package com.llm4j.jota.memory.impl;

import com.llm4j.jota.memory.MemoryAllocator;
import com.llm4j.jota.memory.ScopedMemoryAllocator;
import com.llm4j.jota.memory.ScopedMemoryAllocatorArena;

import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class MemoryAllocatorFactory {

    private MemoryAllocatorFactory() {
        // no instances
    }

    public static MemoryAllocator<byte[]> ofBytes() {
        return ByteArrayAllocator.instance();
    }

    public static MemoryAllocator<float[]> ofFloats() {
        return FloatArrayAllocator.instance();
    }

    public static MemoryAllocator<ByteBuffer> ofByteBuffer(boolean direct, ByteOrder byteOrder) {
        return ByteBufferAllocator.create(direct, byteOrder);
    }

    // Native order.
    public static MemoryAllocator<ByteBuffer> ofByteBuffer(boolean direct) {
        return ByteBufferAllocator.create(direct, ByteOrder.nativeOrder());
    }

    public static ScopedMemoryAllocator<MemorySegment> ofPanama() {
        return UnsafeAllocator.instance();
    }

    public static ScopedMemoryAllocatorArena<MemorySegment> newPanamaArena() {
        return UnsafeAllocatorArena.create();
    }

    public static MemoryAllocator<MemorySegment> newPanamaAuto() {
        return new ManagedPanamaAllocator();
    }
}
