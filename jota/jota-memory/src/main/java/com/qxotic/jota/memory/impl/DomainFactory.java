package com.qxotic.jota.memory.impl;

import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

public final class DomainFactory {

    private static final MemoryDomain<boolean[]> BOOLEANS =
            new ArrayMemoryDomain<>(
                    boolean[].class,
                    MemoryAllocatorFactory.ofBooleans(),
                    BooleansMemoryAccess.instance(),
                    BooleansMemoryOperations.instance());
    private static final MemoryDomain<byte[]> BYTES =
            new ArrayMemoryDomain<>(
                    byte[].class,
                    MemoryAllocatorFactory.ofBytes(),
                    BytesMemoryAccess.instance(),
                    BytesMemoryOperations.instance());
    private static final MemoryDomain<short[]> SHORTS =
            new ArrayMemoryDomain<>(
                    short[].class,
                    MemoryAllocatorFactory.ofShorts(),
                    ShortsMemoryAccess.instance(),
                    ShortsMemoryOperations.instance());
    private static final MemoryDomain<int[]> INTS =
            new ArrayMemoryDomain<>(
                    int[].class,
                    MemoryAllocatorFactory.ofInts(),
                    IntsMemoryAccess.instance(),
                    IntsMemoryOperations.instance());
    private static final MemoryDomain<long[]> LONGS =
            new ArrayMemoryDomain<>(
                    long[].class,
                    MemoryAllocatorFactory.ofLongs(),
                    LongsMemoryAccess.instance(),
                    LongsMemoryOperations.instance());
    private static final MemoryDomain<float[]> FLOATS =
            new ArrayMemoryDomain<>(
                    float[].class,
                    MemoryAllocatorFactory.ofFloats(),
                    FloatsMemoryAccess.instance(),
                    FloatsMemoryOperations.instance());
    private static final MemoryDomain<double[]> DOUBLES =
            new ArrayMemoryDomain<>(
                    double[].class,
                    MemoryAllocatorFactory.ofDoubles(),
                    DoublesMemoryAccess.instance(),
                    DoublesMemoryOperations.instance());

    private DomainFactory() {
        // no instances
    }

    public static MemoryDomain<boolean[]> ofBooleans() {
        return BOOLEANS;
    }

    public static MemoryDomain<byte[]> ofBytes() {
        return BYTES;
    }

    public static MemoryDomain<short[]> ofShorts() {
        return SHORTS;
    }

    public static MemoryDomain<int[]> ofInts() {
        return INTS;
    }

    public static MemoryDomain<long[]> ofLongs() {
        return LONGS;
    }

    public static MemoryDomain<float[]> ofFloats() {
        return FLOATS;
    }

    public static MemoryDomain<double[]> ofDoubles() {
        return DOUBLES;
    }

    public static MemoryDomain<MemorySegment> ofMemorySegment(
            MemoryAllocator<MemorySegment> memoryAllocator) {
        return NativeMemoryFactory.createDomain(memoryAllocator);
    }

    public static MemoryDomain<MemorySegment> ofMemorySegment() {
        return NativeMemoryFactory.createDomain();
    }

    public static MemoryDomain<ByteBuffer> ofByteBuffer(
            MemoryAllocator<ByteBuffer> memoryAllocator) {
        return new ByteBufferDomain(memoryAllocator);
    }
}
