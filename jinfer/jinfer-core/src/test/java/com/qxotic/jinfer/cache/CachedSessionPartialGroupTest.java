package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.cache.PromptCacheTest.FakeCodec;
import com.qxotic.jinfer.cache.PromptCacheTest.FakeModel;
import com.qxotic.jinfer.cache.PromptCacheTest.FakeState;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Regression: a cache hit ends on a BLOCK boundary, which need not be a GROUP boundary — the
 * committed generation-prompt block is a byte-exact prefix of the echoed assistant turn, so a
 * follow-up request resumes MID-group. {@link CachedSession#ingestGroups} must ingest only the
 * un-restored tail of that group; re-ingesting the whole group duplicated its restored head in the
 * context (the server bug this pins). Fakes are the package fixture ({@link PromptCacheTest}),
 * whose state records every ingested token id.
 */
public final class CachedSessionPartialGroupTest {

    static int failures;

    @Test
    void run() {
        FakeModel model = new FakeModel(new FakeCodec(false));
        BlockTree<FakeState> cache =
                new BlockTree<>(
                        new FakeCodec(false), CacheStore.inMemory(), 1 << 20, new byte[] {1});

        // Request 1: [start 10,11] [user 20,21,22] [genPrompt 30,31] — three turn-aligned groups.
        List<List<Batch>> first =
                List.of(
                        List.of(Batch.prefill(new int[] {10, 11})),
                        List.of(Batch.prefill(new int[] {20, 21, 22})),
                        List.of(Batch.prefill(new int[] {30, 31})));
        CachedSession<FakeState> s1 = CachedSession.start(model, cache, model.newState(0, 0));
        s1.ingestGroups(first);
        check(s1.position() == 7, "request 1 ingested 7 positions");

        // Request 2 echoes the conversation: the assistant turn STARTS with the genPrompt tokens
        // (30,31) then continues (40,41) — its group is [30,31,40,41]. The cached genPrompt block
        // fingerprint-matches the group's head, so resume stops MID-group at position 7.
        List<List<Batch>> second =
                List.of(
                        List.of(Batch.prefill(new int[] {10, 11})),
                        List.of(Batch.prefill(new int[] {20, 21, 22})),
                        List.of(Batch.prefill(new int[] {30, 31, 40, 41})), // echoed assistant turn
                        List.of(Batch.prefill(new int[] {50}))); // new user turn
        long[] expected = {10, 11, 20, 21, 22, 30, 31, 40, 41, 50};
        FakeState state2 = model.newState(0, 0);
        CachedSession<FakeState> s2 =
                CachedSession.resume(model, cache, state2, expected, expected.length, true);
        check(
                s2.position() == 7,
                "resume stops mid-group at the genPrompt block (got " + s2.position() + ")");

        s2.ingestGroups(second);
        check(
                s2.position() == 10,
                "session ends at the full conversation length (got " + s2.position() + ")");
        check(
                state2.ingested.equals(List.of(40, 41, 50)),
                "only the un-restored tail is ingested, no duplicated group head (got "
                        + state2.ingested
                        + ")");

        // And a request that resumes exactly ON a group boundary still skips whole groups.
        FakeState state3 = model.newState(0, 0);
        CachedSession<FakeState> s3 =
                CachedSession.resume(
                        model, cache, state3, new long[] {10, 11, 20, 21, 22}, 5, true);
        check(s3.position() == 5, "boundary resume restores both whole groups");
        s3.ingestGroups(
                List.of(
                        List.of(Batch.prefill(new int[] {10, 11})),
                        List.of(Batch.prefill(new int[] {20, 21, 22})),
                        List.of(Batch.prefill(new int[] {60, 61}))));
        check(
                state3.ingested.equals(List.of(60, 61)),
                "boundary resume ingests only the new group");

        if (failures > 0) {
            System.out.println(failures + " failure(s)");
            throw new AssertionError("failure(s) - see output above");
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
