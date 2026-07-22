package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.RuntimeState;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * Eviction stress for PromptCache: a tiny budget forcing evictions during chain commits, dedup onto
 * shared prefixes, detached tips, and resumes - must never double-free or corrupt the tree (repro
 * harness for the 12B multimodal eviction crash).
 */
public final class PromptCacheEvictionTest {

    static final class FakeState implements RuntimeState {
        int position;

        @Override
        public int contextCapacity() {
            return 1 << 20;
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
            return positions * 1024L + 4096; // rows + fixed residue
        }

        @Override
        public void save(FakeState s, int from, int to, MemorySegment dst) {}

        @Override
        public void restore(FakeState s, int from, int to, MemorySegment src) {}
    }

    @Test
    void run() {
        // budget fits ~6 single-token blocks: every conversation evicts its own history
        PromptCache<FakeState> cache =
                new PromptCache<>(
                        new FakeCodec(), CacheStore.inMemory(), 32 * 1024, new byte[] {1});

        for (int round = 0; round < 5; round++) {
            FakeState s = new FakeState();
            PromptCache<FakeState>.Block tip = cache.resume(new long[0], 0, s);
            long[] fp = new long[64];
            for (int i = 0; i < 64; i++) {
                fp[i] = round * 1000L + i;
                if (i < 8) fp[i] = i; // shared prefix across rounds (dedup)
                s.position = i + 1;
                tip = cache.commit(tip, fp, i, 1, s);
            }
            // resume against what survived (may be nothing - correctness never depends on it)
            FakeState r = new FakeState();
            cache.resume(fp, 64, r);
        }
        System.out.println(cache.stats());
        System.out.println("PromptCacheEvictionTest: no crash");
    }
}
