package com.qxotic.jinfer.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.RuntimeState;
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
    static final class FakeState implements RuntimeState {
        int position;
        final long[] rows = new long[1 << 10];
        long residue; // simulated recurrent state: must equal residueAt(position) when live

        static long rowAt(int p) {
            return 0xC0FFEE_0000L + p;
        }

        static long residueAt(int p) {
            return 0xBADD_0000L + p;
        }

        void ingestTo(int to) { // simulate ingestion: rows appear, residue mutates in place
            for (int p = position; p < to; p++) rows[p] = rowAt(p);
            position = to;
            residue = residueAt(to);
        }

        @Override
        public int contextCapacity() {
            return rows.length;
        }

        @Override
        public int batchCapacity() {
            return 512;
        }

        @Override
        public int position() {
            return position;
        }

        @Override
        public int outputCount() {
            return 1;
        }

        @Override
        public void resumeAt(int p) {
            position = p;
        }
    }

    static final class FakeCodec implements StateCodec<FakeState> {
        @Override
        public long blockBytes(int positions) {
            return positions * 8L + 8; // rows + residue trailer
        }

        @Override
        public void save(FakeState s, int from, int to, MemorySegment dst) {
            assertEquals(to, s.position, "save is only valid at state.position() == to");
            for (int p = from; p < to; p++)
                dst.setAtIndex(ValueLayout.JAVA_LONG, p - from, s.rows[p]);
            dst.set(ValueLayout.JAVA_LONG, (long) (to - from) * 8, s.residue);
        }

        @Override
        public void restore(FakeState s, int from, int to, MemorySegment src) {
            for (int p = from; p < to; p++)
                s.rows[p] = src.getAtIndex(ValueLayout.JAVA_LONG, p - from);
            s.residue = src.get(ValueLayout.JAVA_LONG, (long) (to - from) * 8);
        }
    }

    @Test
    void everyBlockBoundaryIsAResumePoint() {
        PromptCache<FakeState> cache =
                new PromptCache<>(new FakeCodec(), CacheStore.inMemory(), 1 << 20, new byte[] {7});

        // build a chain of three blocks: [0,10) [10,17) [17,22)
        int[] bounds = {10, 17, 22};
        long[] fp = new long[22];
        for (int i = 0; i < fp.length; i++) fp[i] = 100 + i;
        FakeState w = new FakeState();
        PromptCache<FakeState>.Block tip = cache.resume(new long[0], 0, w);
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
            assertEquals(b, r.position, "resume lands exactly on boundary " + b);
            for (int p = 0; p < b; p++) assertEquals(FakeState.rowAt(p), r.rows[p], "row " + p);
            assertEquals(FakeState.residueAt(b), r.residue, "residue as of " + b);
        }

        // mid-block divergence: lands on the PREVIOUS boundary, never zero
        long[] diverged = fp.clone();
        diverged[13] ^= 0x5DEECE66DL; // inside block [10,17)
        FakeState d = new FakeState();
        cache.resume(diverged, diverged.length, d);
        assertEquals(10, d.position, "mid-block divergence resumes at the previous boundary");
        assertEquals(FakeState.residueAt(10), d.residue, "residue of the landing boundary");

        // first-block divergence: cold start
        long[] cold = fp.clone();
        cold[0] ^= 1;
        FakeState c = new FakeState();
        cache.resume(cold, cold.length, c);
        assertEquals(0, c.position, "divergence in the first block is a cold start");

        assertTrue(cache.stats().contains("blocks=3"), cache.stats());
    }
}
