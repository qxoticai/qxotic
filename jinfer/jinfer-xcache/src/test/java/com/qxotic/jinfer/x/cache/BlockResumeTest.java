package com.qxotic.jinfer.x.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.boundary.CheckpointCodec;
import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jinfer.x.boundary.ContextState;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/**
 * The self-contained-block law: EVERY block boundary is a resume point. A synthetic codec with
 * verifiable per-position rows and a position-dependent residue trailer proves that resume restores
 * content exactly at any boundary, that a mid-block divergence lands on the previous boundary
 * (never zero), and that the residue restored is the one saved AT that boundary.
 */
public final class BlockResumeTest {

    /** Rows are a visible array; the residue is a value that only exists "at" a position. */
    static final class FakeState extends ContextState {
        final long[] rows = new long[1 << 10];
        long residue; // simulated recurrent state: must equal residueAt(position) when live

        FakeState() {
            super(1 << 10, 512, new PanamaMemoryArena(Arena.ofAuto()), false);
        }

        static long rowAt(int p) {
            return 0xC0FFEE_0000L + p;
        }

        static long residueAt(int p) {
            return 0xBADD_0000L + p;
        }

        void ingestTo(int to) { // simulate ingestion: rows appear, residue mutates in place
            for (int p = position(); p < to; p++) rows[p] = rowAt(p);
            resumeAt(to);
            residue = residueAt(to);
        }

        @Override
        protected void clearHistory() {}
    }

    static final class FakeCodec extends CheckpointCodec<FakeState> {
        @Override
        protected long sizeOf(int positions) {
            return positions * 8L + 8; // rows + residue trailer
        }

        @Override
        protected void transfer(
                FakeState state, int from, int to, MemorySegment memory, boolean capture) {
            for (int position = from; position < to; position++) {
                long index = position - from;
                if (capture) {
                    memory.setAtIndex(ValueLayout.JAVA_LONG, index, state.rows[position]);
                } else {
                    state.rows[position] = memory.getAtIndex(ValueLayout.JAVA_LONG, index);
                }
            }
            long residueOffset = (long) (to - from) * Long.BYTES;
            if (capture) {
                memory.set(ValueLayout.JAVA_LONG, residueOffset, state.residue);
            } else {
                state.residue = memory.get(ValueLayout.JAVA_LONG, residueOffset);
            }
        }
    }

    @Test
    void everyBlockBoundaryIsAResumePoint() {
        BlockTree<FakeState> cache =
                new BlockTree<>(
                        new FakeCodec(),
                        CacheStore.inMemory(),
                        1 << 20,
                        ContentKey.sha256(new byte[] {7}));

        // build a chain of three blocks: [0,10) [10,17) [17,22)
        int[] bounds = {10, 17, 22};
        long[] fp = new long[22];
        for (int i = 0; i < fp.length; i++) fp[i] = 100 + i;
        FakeState w = new FakeState();
        BlockTree<FakeState>.Block tip = cache.resume(new long[0], 0, w);
        int prev = 0;
        for (int b : bounds) {
            w.ingestTo(b);
            tip = cache.commit(tip, fp, prev, b - prev, w);
            prev = b;
        }

        // resume at EVERY boundary: exact position, exact rows, the residue saved AT the boundary
        for (int b : bounds) {
            FakeState r = new FakeState();
            cache.resume(fp, b, r);
            assertEquals(b, r.position(), "resume lands exactly on boundary " + b);
            for (int p = 0; p < b; p++) assertEquals(FakeState.rowAt(p), r.rows[p], "row " + p);
            assertEquals(FakeState.residueAt(b), r.residue, "residue as of " + b);
        }

        // mid-block divergence: lands on the PREVIOUS boundary, never zero
        long[] diverged = fp.clone();
        diverged[13] ^= 0x5DEECE66DL; // inside block [10,17)
        FakeState d = new FakeState();
        cache.resume(diverged, diverged.length, d);
        assertEquals(10, d.position(), "mid-block divergence resumes at the previous boundary");
        assertEquals(FakeState.residueAt(10), d.residue, "residue of the landing boundary");

        // first-block divergence: cold start
        long[] cold = fp.clone();
        cold[0] ^= 1;
        FakeState c = new FakeState();
        cache.resume(cold, cold.length, c);
        assertEquals(0, c.position(), "divergence in the first block is a cold start");

        assertTrue(cache.stats().contains("blocks=3"), cache.stats());
    }
}
