package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class StridedCopyTest {

    private static final MemoryDomain<byte[]> OPAQUE = new OpaqueByteDomain();

    @Test
    void stagesAStridedSourceForAnOpaqueDomain() {
        MemoryView<byte[]> src = view(new byte[] {9, 1, 2, 9, 3, 4, 9}, 1, Stride.flat(3, 1));
        byte[] result = filled(8, (byte) 9);
        MemoryView<byte[]> dst =
                MemoryView.of(
                        Memories.of(result), 2, DataType.I8, Layout.rowMajor(Shape.flat(2, 2)));

        OPAQUE.copy(src, dst);

        assertArrayEquals(new byte[] {9, 9, 1, 2, 3, 4, 9, 9}, result);
    }

    @Test
    void stagesAStridedDestinationWithoutOverwritingItsGaps() {
        MemoryView<byte[]> src =
                MemoryView.of(
                        Memories.of(new byte[] {9, 1, 2, 3, 4, 9}),
                        1,
                        DataType.I8,
                        Layout.rowMajor(Shape.flat(2, 2)));
        byte[] result = filled(8, (byte) 9);
        MemoryView<byte[]> dst = view(result, 1, Stride.flat(3, 1));

        OPAQUE.copy(src, dst);

        assertArrayEquals(new byte[] {9, 1, 2, 9, 3, 4, 9, 9}, result);
    }

    @Test
    void rejectsIncompatibleViews() {
        MemoryView<byte[]> bytes =
                MemoryView.rowMajor(Memories.of(new byte[4]), DataType.I8, Shape.flat(4));
        MemoryView<byte[]> bools =
                MemoryView.rowMajor(Memories.of(new byte[4]), DataType.BOOL, Shape.flat(4));
        MemoryView<byte[]> shorter =
                MemoryView.rowMajor(Memories.of(new byte[3]), DataType.I8, Shape.flat(3));

        assertThrows(IllegalArgumentException.class, () -> OPAQUE.copy(bytes, bools));
        assertThrows(IllegalArgumentException.class, () -> OPAQUE.copy(bytes, shorter));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        OPAQUE.copy(
                                bytes,
                                MemoryView.rowMajor(
                                        otherDeviceMemory(4), DataType.I8, Shape.flat(4))));
    }

    @Test
    void emptyCopyIsANoOp() {
        MemoryView<byte[]> empty =
                MemoryView.rowMajor(Memories.of(new byte[0]), DataType.I8, Shape.flat(0));
        assertDoesNotThrow(() -> OPAQUE.copy(empty, empty));
    }

    static Stream<DataType> blockTypes() {
        return Stream.of(
                DataType.Q4_0,
                DataType.Q4_1,
                DataType.Q5_1,
                DataType.Q8_0,
                DataType.Q4_K,
                DataType.Q5_K,
                DataType.Q6_K,
                DataType.MXFP4,
                DataType.NVFP4,
                DataType.Q1_0,
                DataType.TQ1_0,
                DataType.TQ2_0);
    }

    /**
     * Blocks are opaque byte runs (17..210 bytes): 2 rows x 3 blocks, every other block of each.
     */
    @ParameterizedTest
    @MethodSource("blockTypes")
    void copiesBlockQuantizedElementsRaw(DataType type) {
        int blk = (int) type.byteSize();
        byte[] data = new byte[2 * 3 * blk];
        for (int i = 0; i < data.length; i++) data[i] = (byte) i;
        MemoryView<byte[]> src =
                MemoryView.rowMajor(Memories.of(data), type, Shape.flat(2, 3)).slice(1, 0, 3, 2);
        byte[] result = new byte[2 * 2 * blk];
        MemoryView<byte[]> dst = MemoryView.rowMajor(Memories.of(result), type, Shape.flat(2, 2));

        MemoryDomains.bytes().copy(src, dst);

        byte[] expected = new byte[result.length];
        System.arraycopy(data, 0, expected, 0, blk);
        System.arraycopy(data, 2 * blk, expected, blk, blk);
        System.arraycopy(data, 3 * blk, expected, 2 * blk, blk);
        System.arraycopy(data, 5 * blk, expected, 3 * blk, blk);
        assertArrayEquals(expected, result);
    }

    @Test
    void overlappingSelfCopyReadsBeforeItWrites() {
        byte[] data = {0, 1, 2, 3, 4, 5, 6, 7};
        MemoryView<byte[]> all = MemoryView.rowMajor(Memories.of(data), DataType.I8, Shape.flat(8));
        // elements {0,2,4} -> {2,4,6}: a forward element loop would read the clobbered 2.
        MemoryDomains.bytes().copy(all.slice(0, 0, 6, 2), all.slice(0, 2, 8, 2));

        assertArrayEquals(new byte[] {0, 1, 0, 3, 2, 5, 4, 7}, data);
    }

    @Test
    void stagesOnlyTheViewSpanForOpaqueDomains() {
        byte[] big = new byte[1 << 20];
        MemoryView<byte[]> src =
                MemoryView.rowMajor(Memories.of(big), DataType.I8, Shape.flat(1 << 20))
                        .slice(0, 1000, 1008, 2);
        MemoryView<byte[]> dst =
                MemoryView.rowMajor(Memories.of(new byte[4]), DataType.I8, Shape.flat(4));
        long[] staged = new long[1];
        MemoryDomain<byte[]> counting =
                new OpaqueByteDomain() {
                    @Override
                    public MemoryOperations<byte[]> memoryOperations() {
                        MemoryOperations<byte[]> d = super.memoryOperations();
                        return new MemoryOperations<>() {
                            public void copy(
                                    Memory<byte[]> s, long so, Memory<byte[]> t, long to, long n) {
                                d.copy(s, so, t, to, n);
                            }

                            public void copyFromNative(
                                    Memory<MemorySegment> s,
                                    long so,
                                    Memory<byte[]> t,
                                    long to,
                                    long n) {
                                d.copyFromNative(s, so, t, to, n);
                            }

                            public void copyToNative(
                                    Memory<byte[]> s,
                                    long so,
                                    Memory<MemorySegment> t,
                                    long to,
                                    long n) {
                                staged[0] += n;
                                d.copyToNative(s, so, t, to, n);
                            }

                            public void fillByte(Memory<byte[]> m, long o, long n, byte v) {
                                d.fillByte(m, o, n, v);
                            }

                            public void fillShort(Memory<byte[]> m, long o, long n, short v) {
                                d.fillShort(m, o, n, v);
                            }

                            public void fillInt(Memory<byte[]> m, long o, long n, int v) {
                                d.fillInt(m, o, n, v);
                            }

                            public void fillLong(Memory<byte[]> m, long o, long n, long v) {
                                d.fillLong(m, o, n, v);
                            }
                        };
                    }
                };

        counting.copy(src, dst);

        assertEquals(7, staged[0]); // bytes 1000..1006 inclusive, not the 1 MB allocation
    }

    private static MemoryView<byte[]> view(byte[] bytes, long byteOffset, Stride stride) {
        return MemoryView.of(
                Memories.of(bytes), byteOffset, DataType.I8, Layout.of(Shape.flat(2, 2), stride));
    }

    private static byte[] filled(int size, byte value) {
        byte[] bytes = new byte[size];
        Arrays.fill(bytes, value);
        return bytes;
    }

    private static Memory<byte[]> otherDeviceMemory(int size) {
        return new Memory<>() {
            private final byte[] bytes = new byte[size];

            @Override
            public long byteSize() {
                return bytes.length;
            }

            @Override
            public boolean isReadOnly() {
                return false;
            }

            @Override
            public Device device() {
                return DeviceType.PANAMA.deviceIndex(0);
            }

            @Override
            public byte[] base() {
                return bytes;
            }

            @Override
            public long memoryGranularity() {
                return 1;
            }
        };
    }

    private static class OpaqueByteDomain implements MemoryDomain<byte[]> {

        private final MemoryDomain<byte[]> delegate = MemoryDomains.bytes();

        @Override
        public Device device() {
            return delegate.device();
        }

        @Override
        public MemoryAllocator<byte[]> memoryAllocator() {
            return delegate.memoryAllocator();
        }

        @Override
        public MemoryAccess<byte[]> directAccess() {
            return null;
        }

        @Override
        public MemoryOperations<byte[]> memoryOperations() {
            return delegate.memoryOperations();
        }

        @Override
        public void close() {}
    }
}
