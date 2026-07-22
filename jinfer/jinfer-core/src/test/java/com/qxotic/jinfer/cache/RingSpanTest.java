package com.qxotic.jinfer.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.FloatTensor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * The ring-row aliasing proof, executable: rows of a sliding-window layer saved per-block through
 * their ring slots and restored in chain order rebuild the exact live window - including spans that
 * wrap the ring edge and spans LONGER than the window (whose dead positions read newer aliased rows
 * at save time; ascending restore makes newest-wins hold).
 */
public final class RingSpanTest {

    static final int W = 8; // ring width (power of 2)
    static final long ROW = 4; // elements per row

    static float rowValue(int p, int e) {
        return p * 100f + e;
    }

    /** Simulate ingestion: write row p into its ring slot. */
    static void ingest(FloatTensor ring, int from, int to) {
        for (int p = from; p < to; p++) {
            int slot = p & (W - 1);
            for (int e = 0; e < ROW; e++) ring.setFloat(slot * ROW + e, rowValue(p, e));
        }
    }

    @Test
    void chainOfSpansRebuildsTheLiveWindow() {
        // blocks: [0,5) [5,9) wraps the edge, [9,30) longer than W (aliases 21 > 8 slots)
        int[][] blocks = {{0, 5}, {5, 9}, {9, 30}};
        try (Arena arena = Arena.ofConfined()) {
            FloatTensor live = FloatTensor.allocateF32((int) (W * ROW));
            MemorySegment[] blobs = new MemorySegment[blocks.length];
            for (int i = 0; i < blocks.length; i++) {
                int from = blocks[i][0], to = blocks[i][1];
                ingest(live, from, to); // state advances to `to`...
                blobs[i] = arena.allocate((to - from) * ROW * 4L, 8); // F32 test ring
                long bytes = KvTransfer.ringSpan(live, from, to, W, ROW, blobs[i], 0, true);
                assertEquals((to - from) * ROW * 4L, bytes, "span bytes (F32 test ring)");
            }

            // restore the whole chain ascending into a cold ring
            FloatTensor cold = FloatTensor.allocateF32((int) (W * ROW));
            for (int i = 0; i < blocks.length; i++) {
                KvTransfer.ringSpan(cold, blocks[i][0], blocks[i][1], W, ROW, blobs[i], 0, false);
            }

            // the live window [30-W, 30) must be exact, slot for slot
            int to = blocks[blocks.length - 1][1];
            for (int p = to - W; p < to; p++) {
                int slot = p & (W - 1);
                for (int e = 0; e < ROW; e++) {
                    assertEquals(
                            rowValue(p, e),
                            cold.getFloat(slot * ROW + e),
                            0f, // F32 test ring: exact
                            "row " + p + " elem " + e);
                }
            }
        }
    }

    @Test
    void midChainBoundaryRestoresItsOwnWindow() {
        // restoring only the first two blocks must leave the window as of position 9
        try (Arena arena = Arena.ofConfined()) {
            FloatTensor live = FloatTensor.allocateF32((int) (W * ROW));
            ingest(live, 0, 5);
            MemorySegment b1 = arena.allocate(5 * ROW * 4L, 8);
            KvTransfer.ringSpan(live, 0, 5, W, ROW, b1, 0, true);
            ingest(live, 5, 9);
            MemorySegment b2 = arena.allocate(4 * ROW * 4L, 8);
            KvTransfer.ringSpan(live, 5, 9, W, ROW, b2, 0, true);

            FloatTensor cold = FloatTensor.allocateF32((int) (W * ROW));
            KvTransfer.ringSpan(cold, 0, 5, W, ROW, b1, 0, false);
            KvTransfer.ringSpan(cold, 5, 9, W, ROW, b2, 0, false);
            for (int p = 1; p < 9; p++) { // window at 9 = rows [1,9)
                int slot = p & (W - 1);
                assertEquals(rowValue(p, 0), cold.getFloat(slot * ROW), 0f, "row " + p);
            }
        }
    }
}
