package com.qxotic.jota.memory;

import com.qxotic.jota.memory.internal.DomainFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

/**
 * {@link MemoryDomain}s. The array domains are shared, stateless singletons (their storage is
 * GC-managed). A native domain is always built {@code of} an allocator you name.
 */
public final class MemoryDomains {

    private MemoryDomains() {}

    public static MemoryDomain<MemorySegment> of(MemoryAllocator<MemorySegment> allocator) {
        return DomainFactory.ofMemorySegment(allocator);
    }

    public static MemoryDomain<ByteBuffer> ofByteBuffer(MemoryAllocator<ByteBuffer> allocator) {
        return DomainFactory.ofByteBuffer(allocator);
    }

    public static MemoryDomain<boolean[]> booleans() {
        return DomainFactory.ofBooleans();
    }

    public static MemoryDomain<byte[]> bytes() {
        return DomainFactory.ofBytes();
    }

    public static MemoryDomain<short[]> shorts() {
        return DomainFactory.ofShorts();
    }

    public static MemoryDomain<int[]> ints() {
        return DomainFactory.ofInts();
    }

    public static MemoryDomain<long[]> longs() {
        return DomainFactory.ofLongs();
    }

    public static MemoryDomain<float[]> floats() {
        return DomainFactory.ofFloats();
    }

    public static MemoryDomain<double[]> doubles() {
        return DomainFactory.ofDoubles();
    }
}
