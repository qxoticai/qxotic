package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.RuntimeState;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

/**
 * Sparse-checkpoint gate: multi-token blocks carry checkpoints, single-token decode runs one every
 * stride; a resume lands at the deepest checkpoint and re-ingesting the tail reproduces the
 * never-cached state exactly. The fake's recurrent state is order-dependent (like a conv history),
 * so any checkpoint-currency mistake changes the final value.
 */
public final class SparseCheckpointTest {

    static int failures;

    /** rows[p] = fingerprint ingested at p; recurrent = order-dependent running mix. */
    static final class FakeState implements RuntimeState {
        final long[] rows = new long[256];
        long recurrent;
        int position;

        void ingest(long[] fp, int from, int to) {
            for (int p = from; p < to; p++) {
                rows[p] = fp[p];
                recurrent = recurrent * 31 + fp[p] * (p + 1); // order- and position-dependent
            }
            position = to;
        }

        @Override
        public int contextCapacity() {
            return rows.length;
        }

        @Override
        public int batchCapacity() {
            return rows.length;
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
        public long rowBytes(int positions) {
            return positions * 8L;
        }

        @Override
        public long checkpointBytes() {
            return 8;
        }

        @Override
        public void saveRows(FakeState s, int from, int to, MemorySegment dst) {
            for (int p = from; p < to; p++)
                dst.setAtIndex(ValueLayout.JAVA_LONG, p - from, s.rows[p]);
        }

        @Override
        public void restoreRows(FakeState s, int from, int to, MemorySegment src) {
            for (int p = from; p < to; p++)
                s.rows[p] = src.getAtIndex(ValueLayout.JAVA_LONG, p - from);
        }

        @Override
        public void saveCheckpoint(FakeState s, int to, MemorySegment dst) {
            dst.set(ValueLayout.JAVA_LONG, 0, s.recurrent);
        }

        @Override
        public void restoreCheckpoint(FakeState s, int to, MemorySegment src) {
            s.recurrent = src.get(ValueLayout.JAVA_LONG, 0);
        }
    }

    public static void main(String[] args) {
        CacheStore store = CacheStore.inMemory();
        PromptCache<FakeState> cache =
                new PromptCache<>(
                        new FakeCodec(), store, 1 << 20, new byte[] {1}, 16); // stride via ctor

        long[] fp = new long[40];
        for (int i = 0; i < fp.length; i++) fp[i] = 100 + i;

        // Build: one 10-token turn block, then 30 single-token decode blocks.
        FakeState live = new FakeState();
        PromptCache<FakeState>.Cursor cur = cache.resume(fp, 0, live);
        live.ingest(fp, 0, 10);
        cur.commit(fp, 0, 10, live);
        for (int p = 10; p < 40; p++) {
            live.ingest(fp, p, p + 1);
            cur.commit(fp[p], live);
        }
        // checkpoints: the multi-token block (to=10) + stride at to=26 (26-10=16). 40-26 < 16.
        check(
                cache.stats().contains("checkpoints=2"),
                "sparse: 2 checkpoints across 31 blocks (" + cache.stats() + ")");
        long expectBytes =
                (10 * 8 + 8) + (29 * 8) + (8 + 8); // turn+ckpt, 29 bare singles, 1 ckpt single
        check(
                store.usedBytes() == expectBytes,
                "single-token blocks cost rows only ("
                        + store.usedBytes()
                        + " vs "
                        + expectBytes
                        + ")");

        // Resume: lands at the deepest checkpoint (26), not the deepest match (40).
        FakeState resumed = new FakeState();
        PromptCache<FakeState>.Cursor cur2 = cache.resume(fp, 40, resumed);
        check(
                cur2.position() == 26 && resumed.position() == 26,
                "resume lands at the last checkpoint (26, got " + cur2.position() + ")");

        // Tail re-ingest reproduces the never-cached state exactly (rows AND recurrent).
        resumed.ingest(fp, 26, 40);
        FakeState reference = new FakeState();
        reference.ingest(fp, 0, 40);
        check(
                Arrays.equals(resumed.rows, reference.rows)
                        && resumed.recurrent == reference.recurrent,
                "tail re-ingest reproduces the never-cached state byte-identically");

        // Divergence inside the un-checkpointed run still resumes at the checkpoint before it.
        long[] mutated = fp.clone();
        mutated[30] = -1;
        FakeState diverged = new FakeState();
        check(
                cache.resume(mutated, 40, diverged).position() == 26,
                "divergent tail resumes at the checkpoint");

        // Continuing to commit from the resume point dedups into the tree and re-checkpoints.
        cur2.commit(fp, 26, 14, resumed); // multi-token -> checkpointed at 40
        FakeState again = new FakeState();
        check(
                cache.resume(fp, 40, again).position() == 40,
                "re-committed tail becomes the new resume point");
        check(
                again.recurrent == reference.recurrent && Arrays.equals(again.rows, reference.rows),
                "full-depth resume after re-commit is exact");

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("SparseCheckpointTest: all passed");
    }

    static void check(boolean ok, String what) {
        if (ok) System.out.println("ok:   " + what);
        else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }
}
