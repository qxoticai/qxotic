package com.qxotic.jota.memory.internal;

import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.ScopedMemoryAllocator;
import com.qxotic.jota.memory.ScopedMemoryAllocatorArena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class MemoryAllocatorFactory {

    private static final MemoryAllocator<boolean[]> BOOLEANS =
            new ArrayMemoryAllocator<>(boolean[].class, boolean[]::new);
    private static final MemoryAllocator<byte[]> BYTES =
            new ArrayMemoryAllocator<>(byte[].class, byte[]::new);
    private static final MemoryAllocator<short[]> SHORTS =
            new ArrayMemoryAllocator<>(short[].class, short[]::new);
    private static final MemoryAllocator<int[]> INTS =
            new ArrayMemoryAllocator<>(int[].class, int[]::new);
    private static final MemoryAllocator<long[]> LONGS =
            new ArrayMemoryAllocator<>(long[].class, long[]::new);
    private static final MemoryAllocator<float[]> FLOATS =
            new ArrayMemoryAllocator<>(float[].class, float[]::new);
    private static final MemoryAllocator<double[]> DOUBLES =
            new ArrayMemoryAllocator<>(double[].class, double[]::new);

    private MemoryAllocatorFactory() {
        // no instances
    }

    public static MemoryAllocator<boolean[]> ofBooleans() {
        return BOOLEANS;
    }

    public static MemoryAllocator<byte[]> ofBytes() {
        return BYTES;
    }

    public static MemoryAllocator<short[]> ofShorts() {
        return SHORTS;
    }

    public static MemoryAllocator<int[]> ofInts() {
        return INTS;
    }

    public static MemoryAllocator<long[]> ofLongs() {
        return LONGS;
    }

    public static MemoryAllocator<float[]> ofFloats() {
        return FLOATS;
    }

    public static MemoryAllocator<double[]> ofDoubles() {
        return DOUBLES;
    }

    public static MemoryAllocator<ByteBuffer> ofByteBuffer(boolean direct, ByteOrder byteOrder) {
        return ByteBufferAllocator.create(direct, byteOrder);
    }

    // Native order.
    public static MemoryAllocator<ByteBuffer> ofByteBuffer(boolean direct) {
        return ByteBufferAllocator.create(direct, ByteOrder.nativeOrder());
    }

    public static ScopedMemoryAllocator<MemorySegment> ofPanama() {
        return NativeMemoryFactory.scopedAllocator();
    }

    public static ScopedMemoryAllocatorArena<MemorySegment> newPanamaArena() {
        return NativeMemoryFactory.createArena();
    }

    public static MemoryAllocator<MemorySegment> newPanamaAuto() {
        return NativeMemoryFactory.createManagedArena();
    }

    public static MemoryAllocator<MemorySegment> newPanamaOnHeap() {
        return NativeMemoryFactory.onHeapAllocator();
    }
}
