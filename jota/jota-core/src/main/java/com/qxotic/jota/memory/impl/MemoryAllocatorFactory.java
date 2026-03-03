package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.ScopedMemoryAllocator;
import com.qxotic.jota.memory.ScopedMemoryAllocatorArena;
import com.qxotic.jota.runtime.panama.PanamaFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class MemoryAllocatorFactory {

    private MemoryAllocatorFactory() {
        // no instances
    }

    public static MemoryAllocator<boolean[]> ofBooleans() {
        return BooleansMemoryAllocator.instance();
    }

    public static MemoryAllocator<byte[]> ofBytes() {
        return BytesMemoryAllocator.instance();
    }

    public static MemoryAllocator<float[]> ofFloats() {
        return FloatsMemoryAllocator.instance();
    }

    public static MemoryAllocator<int[]> ofInts() {
        return IntsMemoryAllocator.instance();
    }

    public static MemoryAllocator<short[]> ofShorts() {
        return ShortsMemoryAllocator.instance();
    }

    public static MemoryAllocator<long[]> ofLongs() {
        return LongsMemoryAllocator.instance();
    }

    public static MemoryAllocator<double[]> ofDoubles() {
        return DoublesMemoryAllocator.instance();
    }

    public static MemoryAllocator<ByteBuffer> ofByteBuffer(boolean direct, ByteOrder byteOrder) {
        return ByteBufferAllocator.create(direct, byteOrder);
    }

    // Native order.
    public static MemoryAllocator<ByteBuffer> ofByteBuffer(boolean direct) {
        return ByteBufferAllocator.create(direct, ByteOrder.nativeOrder());
    }

    public static ScopedMemoryAllocator<MemorySegment> ofPanama() {
        return PanamaFactory.scopedAllocator();
    }

    public static ScopedMemoryAllocatorArena<MemorySegment> newPanamaArena() {
        return PanamaFactory.createArena();
    }

    public static MemoryAllocator<MemorySegment> newPanamaAuto() {
        return PanamaFactory.createManagedArena();
    }

    public static MemoryAllocator<MemorySegment> newPanamaOnHeap() {
        return PanamaFactory.onHeapAllocator();
    }
}
