package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.PanamaMemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * Eviction stress for PromptCache: a tiny budget forcing evictions during chain commits, dedup onto
 * shared prefixes, detached tips, and resumes - must never double-free or corrupt the tree (repro
 * harness for the 12B multimodal eviction crash).
 */
public final class BlockTreeEvictionTest {

    static final class FakeState extends ContextState {

        FakeState() {
            super(1 << 20, 512, new PanamaMemoryArena(Arena.ofAuto()), false);
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
}
