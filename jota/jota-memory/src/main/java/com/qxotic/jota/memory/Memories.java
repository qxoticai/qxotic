package com.qxotic.jota.memory;

import com.qxotic.jota.memory.internal.MemoryFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

/**
 * {@link Memory} over storage you already have. Every method wraps; nothing is allocated. Arrays
 * only, no varargs: {@code of(42)} must not silently pick a primitive type.
 */
public final class Memories {

    private Memories() {}

    public static Memory<boolean[]> of(boolean[] booleans) {
        return MemoryFactory.ofBooleans(booleans);
    }

    public static Memory<byte[]> of(byte[] bytes) {
        return MemoryFactory.ofBytes(bytes);
    }

    public static Memory<short[]> of(short[] shorts) {
        return MemoryFactory.ofShorts(shorts);
    }

    public static Memory<int[]> of(int[] ints) {
        return MemoryFactory.ofInts(ints);
    }

    public static Memory<long[]> of(long[] longs) {
        return MemoryFactory.ofLongs(longs);
    }

    public static Memory<float[]> of(float[] floats) {
        return MemoryFactory.ofFloats(floats);
    }

    public static Memory<double[]> of(double[] doubles) {
        return MemoryFactory.ofDoubles(doubles);
    }

    /**
     * The whole buffer as memory: {@code [0, capacity)}, addressed absolutely.
     *
     * <ul>
     *   <li><b>Cursor.</b> Position, limit and mark are {@link ByteBuffer} bookkeeping, not part of
     *       the memory; they are ignored and never moved. To wrap a window, pass {@code
     *       buffer.slice()} (a slice shares the storage and starts at the old position).
     *   <li><b>Byte order.</b> Typed reads and writes go through the buffer, so they follow the
     *       buffer's own {@link ByteBuffer#order()}: a big-endian buffer holds big-endian values,
     *       exactly as {@code buffer.getInt} would see them. Bulk copies move bytes as they are.
     *       Every other jota backend, and therefore every byte that crosses into one, is
     *       native-order; when the bytes will meet a segment, an array or a device, use a
     *       native-order buffer ({@link MemoryAllocators#newByteBuffer(boolean)}, or {@code
     *       buffer.order(ByteOrder.nativeOrder())}). Note that {@code ByteBuffer.allocate}, {@code
     *       allocateDirect} and {@code slice()} all produce big-endian buffers.
     *   <li><b>Sharing.</b> Zero-copy: {@code base()} is the buffer you passed, writes through jota
     *       are visible in it and vice versa, and a read-only buffer yields read-only memory.
     * </ul>
     */
    public static Memory<ByteBuffer> of(ByteBuffer byteBuffer) {
        return MemoryFactory.ofByteBuffer(byteBuffer);
    }

    public static Memory<MemorySegment> of(MemorySegment memorySegment) {
        return MemoryFactory.ofMemorySegment(memorySegment);
    }
}
