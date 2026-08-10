package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential oracle: x KvTransfer (FP16 view side) vs jinfer-core KvTransfer (F16 tensor side) on
 * identical ring/blob content — straight transfers and ring walks in both directions.
 */
class KvTransferTest {

    private final Arena arena = Arena.ofAuto();

    private MemorySegment randomF16(int n, long seed) {
        MemorySegment seg = arena.allocate(2L * n, 64);
        Random rng = new Random(seed);
        for (int i = 0; i < n; i++) {
            seg.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    2L * i,
                    Float.floatToFloat16(rng.nextFloat() * 4 - 2));
        }
        return seg;
    }

    private void assertBlobEqual(MemorySegment a, MemorySegment b, long bytes, String what) {
        for (long i = 0; i < bytes; i++) {
            assertEquals(
                    a.get(ValueLayout.JAVA_BYTE, i),
                    b.get(ValueLayout.JAVA_BYTE, i),
                    what + " at byte " + i);
        }
    }

    @Test
    void transferBothDirections() {
        int n = 1000;
        MemorySegment ring = randomF16(n, 1);
        FloatTensor oldRing = FloatTensor.create(GGMLType.F16, n, ring);
        MemoryView<MemorySegment> newRing = Views.wrap(ring, DataType.FP16, Shape.flat((long) n));
        MemorySegment blobOld = arena.allocate(2L * n, 64);
        MemorySegment blobNew = arena.allocate(2L * n, 64);

        long movedOld =
                com.qxotic.jinfer.cache.KvTransfer.transfer(oldRing, 3, blobOld, 10, 500, true);
        long movedNew = KvTransfer.transfer(newRing, 3, blobNew, 10, 500, true);
        assertEquals(movedOld, movedNew);
        assertBlobEqual(blobOld, blobNew, movedOld, "transfer out");

        // restore direction: scribble on one blob, copy back through both paths
        MemorySegment src = arena.allocate(2L * n, 64);
        new Random(2).nextBytes(src.asSlice(0, 64).toArray(ValueLayout.JAVA_BYTE));
        MemorySegment.copy(src, 0, blobOld, 0, 64);
        MemorySegment.copy(src, 0, blobNew, 0, 64);
        MemorySegment ringOld2 = randomF16(n, 3), ringNew2 = arena.allocate(2L * n, 64);
        MemorySegment.copy(ringOld2, 0, ringNew2, 0, 2L * n);
        com.qxotic.jinfer.cache.KvTransfer.transfer(
                FloatTensor.create(GGMLType.F16, n, ringOld2), 0, blobOld, 0, 32, false);
        KvTransfer.transfer(
                Views.wrap(ringNew2, DataType.FP16, Shape.flat((long) n)),
                0,
                blobNew,
                0,
                32,
                false);
        assertBlobEqual(ringOld2, ringNew2, 2L * n, "transfer in");
    }

    @Test
    void ringSpanParity() {
        int w = 16, rowElems = 8; // ring of 16 slots x 8 elems
        int n = w * rowElems;
        MemorySegment ring = randomF16(n, 4);
        FloatTensor oldRing = FloatTensor.create(GGMLType.F16, n, ring);
        MemoryView<MemorySegment> newRing = Views.wrap(ring, DataType.FP16, Shape.flat((long) n));
        int from = 3, to = 41; // crosses the ring edge multiple times
        long bytes = (long) (to - from) * rowElems * 2;
        MemorySegment blobOld = arena.allocate(bytes, 64);
        MemorySegment blobNew = arena.allocate(bytes, 64);

        long coveredOld =
                com.qxotic.jinfer.cache.KvTransfer.ringSpan(
                        oldRing, from, to, w, rowElems, blobOld, 0, true);
        long coveredNew = KvTransfer.ringSpan(newRing, from, to, w, rowElems, blobNew, 0, true);
        assertEquals(coveredOld, coveredNew);
        assertEquals(bytes, coveredNew);
        assertBlobEqual(blobOld, blobNew, bytes, "ringSpan out");

        // restore: blobs -> fresh rings, compare the rings
        MemorySegment ringOld2 = randomF16(n, 5), ringNew2 = arena.allocate(2L * n, 64);
        MemorySegment.copy(ringOld2, 0, ringNew2, 0, 2L * n);
        com.qxotic.jinfer.cache.KvTransfer.ringSpan(
                FloatTensor.create(GGMLType.F16, n, ringOld2),
                from,
                to,
                w,
                rowElems,
                blobOld,
                0,
                false);
        KvTransfer.ringSpan(
                Views.wrap(ringNew2, DataType.FP16, Shape.flat((long) n)),
                from,
                to,
                w,
                rowElems,
                blobNew,
                0,
                false);
        assertBlobEqual(ringOld2, ringNew2, 2L * n, "ringSpan in");
    }

    @Test
    void floatArrayTransferParity() {
        float[] a = new float[300];
        Random rng = new Random(6);
        for (int i = 0; i < a.length; i++) a[i] = rng.nextFloat();
        MemorySegment blob = arena.allocate(a.length * 4L + 8, 64);
        assertEquals(
                com.qxotic.jinfer.cache.KvTransfer.transfer(a, blob, 4, true),
                KvTransfer.transfer(a, blob, 4, true));
        float[] b = new float[a.length];
        KvTransfer.transfer(b, blob, 4, false);
        org.junit.jupiter.api.Assertions.assertArrayEquals(a, b);
    }
}
