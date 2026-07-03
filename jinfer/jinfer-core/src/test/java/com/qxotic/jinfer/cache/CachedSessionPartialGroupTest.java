package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CacheStore;
import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.Model;
import com.qxotic.jinfer.RuntimeState;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/** Regression: a cache hit ends on a BLOCK boundary, which need not be a GROUP boundary — the
 *  committed generation-prompt block is a byte-exact prefix of the echoed assistant turn, so a
 *  follow-up request resumes MID-group. {@link CachedSession#ingestGroups} must ingest only the
 *  un-restored tail of that group; re-ingesting the whole group duplicated its restored head in
 *  the context (the server bug this pins). */
public final class CachedSessionPartialGroupTest {

    static int failures;

    /** Records every token id the model actually ingests, in order. */
    static final class FakeState implements RuntimeState {
        int position;
        final List<Integer> ingested = new ArrayList<>();
        @Override public int contextCapacity() { return 1 << 20; }
        @Override public int batchCapacity() { return 512; }
        @Override public int position() { return position; }
        @Override public int outputCount() { return 1; }
        @Override public void resumeAt(int p) { position = p; }
    }

    static final class FakeModel implements Model<Config, Object, FakeState> {
        @Override public Config config() { return null; }
        @Override public Object weights() { return null; }
        @Override public FakeState newState(int contextCapacity, int batchCapacity) { return new FakeState(); }
        @Override public void ingest(FakeState s, Batch batch) {
            for (int id : ((Batch.Input.Tokens) batch.input()).ids()) s.ingested.add(id);
            s.position += batch.count();
        }
    }

    static final class FakeCodec implements KvCodec<FakeState> {
        @Override public long rowBytes(int positions) { return positions * 8L; }
        @Override public void saveRows(FakeState s, int from, int to, MemorySegment dst) { }
        @Override public void restoreRows(FakeState s, int from, int to, MemorySegment src) { }
    }

    public static void main(String[] args) {
        FakeModel model = new FakeModel();
        PromptCache<FakeState> cache = new PromptCache<>(new FakeCodec(), CacheStore.inMemory(), 1 << 20, new byte[]{1});

        // Request 1: [start 10,11] [user 20,21,22] [genPrompt 30,31] — three turn-aligned groups.
        List<List<Batch>> first = List.of(
                List.of(Batch.prefill(new int[]{10, 11})),
                List.of(Batch.prefill(new int[]{20, 21, 22})),
                List.of(Batch.prefill(new int[]{30, 31})));
        CachedSession<FakeState> s1 = CachedSession.resume(model, cache, model.newState(0, 0), new long[0]);
        s1.ingestGroups(first);
        check(s1.position() == 7, "request 1 ingested 7 positions");

        // Request 2 echoes the conversation: the assistant turn STARTS with the genPrompt tokens
        // (30,31) then continues (40,41) — its group is [30,31,40,41]. The cached genPrompt block
        // fingerprint-matches the group's head, so resume stops MID-group at position 7.
        List<List<Batch>> second = List.of(
                List.of(Batch.prefill(new int[]{10, 11})),
                List.of(Batch.prefill(new int[]{20, 21, 22})),
                List.of(Batch.prefill(new int[]{30, 31, 40, 41})),      // echoed assistant turn
                List.of(Batch.prefill(new int[]{50})));                  // new user turn
        long[] expected = {10, 11, 20, 21, 22, 30, 31, 40, 41, 50};
        FakeState state2 = model.newState(0, 0);
        CachedSession<FakeState> s2 = CachedSession.resume(model, cache, state2, expected);
        check(s2.position() == 7, "resume stops mid-group at the genPrompt block (got " + s2.position() + ")");

        s2.ingestGroups(second);
        check(s2.position() == 10, "session ends at the full conversation length (got " + s2.position() + ")");
        check(state2.ingested.equals(List.of(40, 41, 50)),
                "only the un-restored tail is ingested, no duplicated group head (got " + state2.ingested + ")");

        // And a request that resumes exactly ON a group boundary still skips whole groups.
        FakeState state3 = model.newState(0, 0);
        CachedSession<FakeState> s3 = CachedSession.resume(model, cache, state3, new long[]{10, 11, 20, 21, 22});
        check(s3.position() == 5, "boundary resume restores both whole groups");
        s3.ingestGroups(List.of(
                List.of(Batch.prefill(new int[]{10, 11})),
                List.of(Batch.prefill(new int[]{20, 21, 22})),
                List.of(Batch.prefill(new int[]{60, 61}))));
        check(state3.ingested.equals(List.of(60, 61)), "boundary resume ingests only the new group");

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            System.exit(1);
        }
        System.out.println("CachedSessionPartialGroupTest: all checks passed");
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
