package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.Model;
import com.qxotic.jinfer.RuntimeState;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * SessionPool: strict-prefix append-only matching, longest-match wins, mid-stream divergence is
 * never reused (recurrent state cannot rewind), LRU eviction past capacity, and adopt() keeps a
 * pooled session's stream in lockstep with decode-loop tokens ingested directly on the state.
 */
public final class SessionPoolTest {

    static int failures;

    static final class FakeState implements RuntimeState {
        int position;
        final List<Integer> ingested = new ArrayList<>();

        @Override
        public int contextCapacity() {
            return 64;
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

    static final class FakeModel implements Model<Config, Object, FakeState> {
        @Override
        public Config config() {
            return null;
        }

        @Override
        public Object weights() {
            return null;
        }

        @Override
        public FakeState newState(int contextCapacity, int batchCapacity) {
            return new FakeState();
        }

        @Override
        public void ingest(FakeState s, Batch batch) {
            for (int id : ((Batch.Input.Tokens) batch.input()).ids()) s.ingested.add(id);
            s.position += batch.count();
        }
    }

    static final class FakeCodec implements StateCodec<FakeState> {
        @Override
        public long rowBytes(int positions) {
            return positions * 8L;
        }

        @Override
        public void saveRows(FakeState s, int from, int to, MemorySegment dst) {}

        @Override
        public void restoreRows(FakeState s, int from, int to, MemorySegment src) {}
    }

    public static void main(String[] args) {
        FakeModel model = new FakeModel();
        PromptCache<FakeState> cache =
                new PromptCache<>(new FakeCodec(), CacheStore.inMemory(), 1 << 20, new byte[] {1});
        SessionPool<FakeState> pool = new SessionPool<>(2);

        // Conversation A: [1,2,3] ingested, then a decode-loop reply [7,8] adopted.
        CachedSession<FakeState> a =
                CachedSession.resume(model, cache, model.newState(0, 0), new long[0]);
        a.ingest(List.of(Batch.prefill(new int[] {1, 2, 3})));
        a.state().position += 2; // generator steps the state directly
        a.adopt(List.of(7, 8));
        check(a.length() == 5 && a.position() == 5, "adopt keeps stream and state in lockstep");
        pool.release(a);

        // Append-only follow-up: A's whole stream [1,2,3,7,8] prefixes the request -> tier 1.
        long[] followUp = {1, 2, 3, 7, 8, 9, 10};
        CachedSession<FakeState> hit = pool.acquire(followUp, followUp.length);
        check(hit == a, "strict-prefix stream reuses the pooled session");
        check(pool.size() == 0, "acquired session leaves the pool");
        pool.release(a);

        // Mid-stream divergence: request diverges at position 3 -> no reuse (cannot rewind).
        long[] diverged = {1, 2, 3, 99, 8, 9};
        check(
                pool.acquire(diverged, diverged.length) == null,
                "mid-stream divergence is never reused");

        // Identical conversation (no delta): not a STRICT prefix -> no reuse (logits would be
        // stale).
        long[] identical = {1, 2, 3, 7, 8};
        check(pool.acquire(identical, identical.length) == null, "identical stream is not reused");

        // Longest match wins: B = [1,2] also prefixes the follow-up; A (5) beats B (2).
        CachedSession<FakeState> b =
                CachedSession.resume(model, cache, model.newState(0, 0), new long[0]);
        b.ingest(List.of(Batch.prefill(new int[] {1, 2})));
        pool.release(b);
        check(pool.acquire(followUp, followUp.length) == a, "longest matching stream wins");
        pool.release(a);

        // Context bound: a request longer than the state's capacity is not placed on it.
        long[] tooLong = new long[65];
        for (int i = 0; i < 5; i++) tooLong[i] = new long[] {1, 2, 3, 7, 8}[i];
        for (int i = 5; i < 65; i++) tooLong[i] = 1000 + i;
        check(
                pool.acquire(tooLong, tooLong.length) == null,
                "request past contextCapacity is not pooled onto the state");

        // LRU eviction: capacity 2; releasing C evicts the least-recent (B).
        CachedSession<FakeState> c =
                CachedSession.resume(model, cache, model.newState(0, 0), new long[0]);
        c.ingest(List.of(Batch.prefill(new int[] {5})));
        pool.release(c); // pool: [b, a-released-last? order: b, a, c]
        check(pool.size() == 2, "capacity bounds the pool");
        check(
                pool.acquire(new long[] {1, 2, 9}, 3) == null,
                "evicted least-recent session (B) is gone");
        check(
                pool.acquire(followUp, followUp.length) == a,
                "most-recent sessions survive eviction");

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("SessionPoolTest: all passed");
    }

    static void check(boolean ok, String what) {
        if (ok) {
            System.out.println("ok:   " + what);
        } else {
            failures++;
            System.out.println("FAIL: " + what);
        }
    }
}
