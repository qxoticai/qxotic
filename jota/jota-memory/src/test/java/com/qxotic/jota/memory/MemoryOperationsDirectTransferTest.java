package com.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.internal.DomainFactory;
import com.qxotic.jota.memory.internal.MemoryFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** A segment on either side of the static copy is one direct transfer, not a chunked bounce. */
class MemoryOperationsDirectTransferTest {

    private static final int SIZE =
            1 << 20; // 1 MB: the staged path would split this into ~10 chunks

    /** byte[] backend that records every transfer it is asked to do. */
    private static final class Recording implements MemoryOperations<byte[]> {
        final MemoryOperations<byte[]> delegate = DomainFactory.ofBytes().memoryOperations();
        final List<Long> toNative = new ArrayList<>();
        final List<Long> fromNative = new ArrayList<>();

        public void copy(Memory<byte[]> s, long so, Memory<byte[]> d, long o, long n) {
            delegate.copy(s, so, d, o, n);
        }

        public void copyFromNative(
                Memory<MemorySegment> s, long so, Memory<byte[]> d, long o, long n) {
            fromNative.add(n);
            delegate.copyFromNative(s, so, d, o, n);
        }

        public void copyToNative(
                Memory<byte[]> s, long so, Memory<MemorySegment> d, long o, long n) {
            toNative.add(n);
            delegate.copyToNative(s, so, d, o, n);
        }

        public void fillByte(Memory<byte[]> m, long o, long n, byte v) {}

        public void fillShort(Memory<byte[]> m, long o, long n, short v) {}

        public void fillInt(Memory<byte[]> m, long o, long n, int v) {}

        public void fillLong(Memory<byte[]> m, long o, long n, long v) {}
    }

    @Test
    void segmentDestinationIsOneTransfer() {
        byte[] data = new byte[SIZE];
        for (int i = 0; i < SIZE; i++) data[i] = (byte) i;
        Recording ops = new Recording();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment dst = arena.allocate(SIZE);
            MemoryOperations.copy(
                    ops,
                    MemoryFactory.ofBytes(data),
                    0,
                    DomainFactory.ofMemorySegment().memoryOperations(),
                    MemoryFactory.ofMemorySegment(dst),
                    0,
                    SIZE);
            assertEquals(List.of((long) SIZE), ops.toNative);
            assertArrayEquals(data, dst.toArray(ValueLayout.JAVA_BYTE));
        }
    }

    @Test
    void segmentSourceIsOneTransfer() {
        byte[] result = new byte[SIZE];
        Recording ops = new Recording();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment src = arena.allocate(SIZE);
            for (int i = 0; i < SIZE; i++) src.set(ValueLayout.JAVA_BYTE, i, (byte) (i * 7));
            MemoryOperations.copy(
                    DomainFactory.ofMemorySegment().memoryOperations(),
                    MemoryFactory.ofMemorySegment(src),
                    0,
                    ops,
                    MemoryFactory.ofBytes(result),
                    0,
                    SIZE);
            assertEquals(List.of((long) SIZE), ops.fromNative);
            assertArrayEquals(src.toArray(ValueLayout.JAVA_BYTE), result);
        }
    }

    /**
     * Heap segments (MemorySegment.ofArray) take the direct branch too; the backend must not treat
     * them as native.
     */
    @Test
    void heapSegmentOnEitherSideIsOneTransfer() {
        byte[] data = new byte[SIZE];
        for (int i = 0; i < SIZE; i++) data[i] = (byte) (i * 3);
        byte[] heap = new byte[SIZE];
        Recording ops = new Recording();
        MemoryOperations<MemorySegment> segOps = DomainFactory.ofMemorySegment().memoryOperations();

        MemoryOperations.copy(
                ops,
                MemoryFactory.ofBytes(data),
                0,
                segOps,
                MemoryFactory.ofMemorySegment(MemorySegment.ofArray(heap)),
                0,
                SIZE);
        assertEquals(List.of((long) SIZE), ops.toNative);
        assertArrayEquals(data, heap);

        byte[] back = new byte[SIZE];
        MemoryOperations.copy(
                segOps,
                MemoryFactory.ofMemorySegment(MemorySegment.ofArray(heap)),
                0,
                ops,
                MemoryFactory.ofBytes(back),
                0,
                SIZE);
        assertEquals(List.of((long) SIZE), ops.fromNative);
        assertArrayEquals(data, back);
    }

    @Test
    void directBranchesHonorOffsets() {
        byte[] data = new byte[64];
        for (int i = 0; i < data.length; i++) data[i] = (byte) i;
        Recording ops = new Recording();
        MemoryOperations<MemorySegment> segOps = DomainFactory.ofMemorySegment().memoryOperations();
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(64);
            // bytes 8..24 of the array land at 32..48 of the segment
            MemoryOperations.copy(
                    ops,
                    MemoryFactory.ofBytes(data),
                    8,
                    segOps,
                    MemoryFactory.ofMemorySegment(seg),
                    32,
                    16);
            byte[] expected = new byte[64];
            System.arraycopy(data, 8, expected, 32, 16);
            assertArrayEquals(expected, seg.toArray(ValueLayout.JAVA_BYTE));

            byte[] back = new byte[64];
            MemoryOperations.copy(
                    segOps,
                    MemoryFactory.ofMemorySegment(seg),
                    32,
                    ops,
                    MemoryFactory.ofBytes(back),
                    48,
                    16);
            byte[] expectedBack = new byte[64];
            System.arraycopy(data, 8, expectedBack, 48, 16);
            assertArrayEquals(expectedBack, back);
        }
    }

    @Test
    void directBranchesStillEnforceGranularity() {
        // int[] memory has 4-byte granularity: an odd offset or size must be rejected before any
        // transfer
        Memory<int[]> ints = MemoryFactory.ofInts(new int[16]);
        MemoryOperations<int[]> intOps = DomainFactory.ofInts().memoryOperations();
        MemoryOperations<MemorySegment> segOps = DomainFactory.ofMemorySegment().memoryOperations();
        try (Arena arena = Arena.ofConfined()) {
            Memory<MemorySegment> seg = MemoryFactory.ofMemorySegment(arena.allocate(64));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MemoryOperations.copy(intOps, ints, 2, segOps, seg, 0, 8));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MemoryOperations.copy(segOps, seg, 0, intOps, ints, 0, 6));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MemoryOperations.copy(segOps, seg, 0, intOps, ints, 0, -4));
        }
    }

    @Test
    void zeroBytesIsANoOp() {
        Recording ops = new Recording();
        MemoryOperations<MemorySegment> segOps = DomainFactory.ofMemorySegment().memoryOperations();
        try (Arena arena = Arena.ofConfined()) {
            Memory<MemorySegment> seg = MemoryFactory.ofMemorySegment(arena.allocate(8));
            MemoryOperations.copy(ops, MemoryFactory.ofBytes(new byte[8]), 0, segOps, seg, 0, 0);
            MemoryOperations.copy(segOps, seg, 0, ops, MemoryFactory.ofBytes(new byte[8]), 0, 0);
        }
        assertEquals(List.of(), ops.toNative);
        assertEquals(List.of(), ops.fromNative);
    }

    @Test
    void noSegmentOnEitherSideStillStages() {
        byte[] data = new byte[SIZE];
        for (int i = 0; i < SIZE; i++) data[i] = (byte) (i ^ 0x55);
        byte[] result = new byte[SIZE];
        Recording srcOps = new Recording();
        Recording dstOps = new Recording();

        MemoryOperations.copy(
                srcOps,
                MemoryFactory.ofBytes(data),
                0,
                dstOps,
                MemoryFactory.ofBytes(result),
                0,
                SIZE);

        assertArrayEquals(data, result);
        // chunked through the staging buffer: several transfers per side, summing to SIZE
        assertTrue(srcOps.toNative.size() > 1, "expected chunking, got " + srcOps.toNative);
        assertEquals(srcOps.toNative, dstOps.fromNative);
        assertEquals(SIZE, srcOps.toNative.stream().mapToLong(Long::longValue).sum());
    }
}
