package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * Eviction stress for PromptCache: a tiny budget forcing evictions during chain commits, dedup onto
 * shared prefixes, detached tips, and resumes - must never double-free or corrupt the tree (repro
 * harness for the 12B multimodal eviction crash).
 */
public final class BlockTreeEvictionTest {

    static final class FakeState extends ContextState {

        FakeState() {
            super(1 << 20, 512, MemoryAllocators.ofArena(Arena.ofAuto()), false);
        }

        @Override
        protected void clearHistory() {}
    }

    static final class FakeCodec extends CheckpointCodec<FakeState> {
        @Override
        protected long sizeOf(int positions) {
            return positions * 1024L + 4096; // rows + fixed residue
        }

        @Override
        protected void transfer(
                FakeState state, int from, int to, MemorySegment memory, boolean capture) {}
    }

    @Test
    void run() {
        // budget fits ~6 single-token blocks: every conversation evicts its own history
        BlockTree<FakeState> cache =
                new BlockTree<>(
                        new FakeCodec(),
                        CacheStore.inMemory(),
                        32 * 1024,
                        ContentKey.sha256(new byte[] {1}));

        for (int round = 0; round < 5; round++) {
            FakeState s = new FakeState();
            BlockTree<FakeState>.Block tip = cache.resume(new long[0], 0, s);
            long[] fp = new long[64];
            for (int i = 0; i < 64; i++) {
                fp[i] = round * 1000L + i;
                if (i < 8) fp[i] = i; // shared prefix across rounds (dedup)
                s.resumeAt(i + 1);
                tip = cache.commit(tip, fp, i, 1, s);
            }
            // resume against what survived (may be nothing - correctness never depends on it)
            FakeState r = new FakeState();
            cache.resume(fp, 64, r);
        }
        System.out.println(cache.stats());
    }

    @Test
    void anEvictedRetainedTipReattachesWhenItsBlockIsBackInTheTree() {
        // A commits one block and keeps its tip; B's traffic evicts it; C recommits the same
        // block; A's next commit must chain on the live twin, not no-op forever on the corpse
        BlockTree<FakeState> cache =
                new BlockTree<>(
                        new FakeCodec(),
                        CacheStore.inMemory(),
                        32 * 1024,
                        ContentKey.sha256(new byte[] {1}));
        long[] shared = {1, 2};
        FakeState a = new FakeState();
        BlockTree<FakeState>.Block tipA = cache.resume(new long[0], 0, a);
        a.resumeAt(1);
        tipA = cache.commit(tipA, shared, 0, 1, a);
        Assertions.assertTrue(tipA.live);

        FakeState b = new FakeState();
        BlockTree<FakeState>.Block tipB = cache.resume(new long[0], 0, b);
        long[] other = new long[8];
        for (int i = 0; i < other.length; i++) {
            other[i] = 100 + i;
            b.resumeAt(i + 1);
            tipB = cache.commit(tipB, other, i, 1, b);
        }
        Assertions.assertFalse(tipA.live, "B's traffic evicted A's tip");

        FakeState c = new FakeState();
        BlockTree<FakeState>.Block tipC = cache.resume(new long[0], 0, c);
        c.resumeAt(1);
        cache.commit(tipC, shared, 0, 1, c);

        a.resumeAt(2);
        BlockTree<FakeState>.Block next = cache.commit(tipA, shared, 1, 1, a);
        Assertions.assertTrue(next.live, "chained on the recommitted twin");
        Assertions.assertEquals(2, next.to);
    }
}
