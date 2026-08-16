package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/** Saving and restoring ring spans must reconstruct the live window at every chain boundary. */
class KvTransferContractTest {

    private static final int WIDTH = 8;
    private static final int ROW = 4;

    @Test
    void chainOfWrappingAndAliasingSpansRebuildsTheLiveWindow() {
        int[][] blocks = {{0, 5}, {5, 9}, {9, 30}};
        try (Arena arena = Arena.ofConfined()) {
            PanamaMemoryArena memory = new PanamaMemoryArena(arena);
            MemoryView<MemorySegment> live = Views.allocateF16(memory, WIDTH * ROW);
            MemorySegment[] blobs = new MemorySegment[blocks.length];
            for (int i = 0; i < blocks.length; i++) {
                int from = blocks[i][0], to = blocks[i][1];
                ingest(live, from, to);
                blobs[i] = arena.allocate((long) (to - from) * ROW * Short.BYTES, 8);
                assertEquals(
                        (long) (to - from) * ROW * Short.BYTES,
                        KvTransfer.ringSpan(live, from, to, WIDTH, ROW, blobs[i], 0, true));
            }

            MemoryView<MemorySegment> restored = Views.allocateF16(memory, WIDTH * ROW);
            for (int i = 0; i < blocks.length; i++) {
                KvTransfer.ringSpan(
                        restored, blocks[i][0], blocks[i][1], WIDTH, ROW, blobs[i], 0, false);
            }
            assertWindow(restored, 30);
        }
    }

    @Test
    void aMidChainBoundaryRestoresItsOwnWindow() {
        try (Arena arena = Arena.ofConfined()) {
            PanamaMemoryArena memory = new PanamaMemoryArena(arena);
            MemoryView<MemorySegment> live = Views.allocateF16(memory, WIDTH * ROW);
            ingest(live, 0, 5);
            MemorySegment first = arena.allocate(5L * ROW * Short.BYTES, 8);
            KvTransfer.ringSpan(live, 0, 5, WIDTH, ROW, first, 0, true);
            ingest(live, 5, 9);
            MemorySegment second = arena.allocate(4L * ROW * Short.BYTES, 8);
            KvTransfer.ringSpan(live, 5, 9, WIDTH, ROW, second, 0, true);

            MemoryView<MemorySegment> restored = Views.allocateF16(memory, WIDTH * ROW);
            KvTransfer.ringSpan(restored, 0, 5, WIDTH, ROW, first, 0, false);
            KvTransfer.ringSpan(restored, 5, 9, WIDTH, ROW, second, 0, false);
            assertWindow(restored, 9);
        }
    }

    private static void ingest(MemoryView<MemorySegment> ring, int from, int to) {
        for (int position = from; position < to; position++) {
            int slot = position & (WIDTH - 1);
            for (int element = 0; element < ROW; element++) {
                ring.memory()
                        .base()
                        .set(
                                ValueLayout.JAVA_SHORT_UNALIGNED,
                                ring.byteOffset() + ((long) slot * ROW + element) * Short.BYTES,
                                Float.floatToFloat16(position * 100f + element));
            }
        }
    }

    private static void assertWindow(MemoryView<MemorySegment> restored, int position) {
        for (int row = position - WIDTH; row < position; row++) {
            int slot = row & (WIDTH - 1);
            for (int element = 0; element < ROW; element++) {
                assertEquals(
                        Float.float16ToFloat(Float.floatToFloat16(row * 100f + element)),
                        Float.float16ToFloat(
                                restored.memory()
                                        .base()
                                        .get(
                                                ValueLayout.JAVA_SHORT_UNALIGNED,
                                                restored.byteOffset()
                                                        + ((long) slot * ROW + element)
                                                                * Short.BYTES)),
                        0f,
                        "row " + row + " element " + element);
            }
        }
    }
}
