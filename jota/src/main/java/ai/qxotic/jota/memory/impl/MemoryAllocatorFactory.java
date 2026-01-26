package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.ScopedMemoryAllocator;
import ai.qxotic.jota.memory.ScopedMemoryAllocatorArena;
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
        return UnsafeAllocator.instance();
    }

    public static ScopedMemoryAllocatorArena<MemorySegment> newPanamaArena() {
        return UnsafeAllocatorArena.create();
    }

    public static MemoryAllocator<MemorySegment> newPanamaAuto() {
        return new PanamaAutoAllocator();
    }

    public static MemoryAllocator<MemorySegment> newPanamaOnHeap() {
        return PanamaBytesAllocator.instance();
    }
}
