package com.qxotic.jinfer.x.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jinfer.x.cache.PromptCacheTest.FakeCodec;
import com.qxotic.jinfer.x.cache.PromptCacheTest.FakeModel;
import com.qxotic.jinfer.x.cache.PromptCacheTest.FakeState;
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

    @Test
    void aMidGroupResumeIngestsOnlyTheUnrestoredTail() {
        FakeModel model = new FakeModel(new FakeCodec(0));
        BlockTree<FakeState> cache =
                new BlockTree<>(
                        new FakeCodec(0),
                        CacheStore.inMemory(),
                        1 << 20,
                        ContentKey.sha256(new byte[] {1}));

        // Request 1: [start 10,11] [user 20,21,22] [genPrompt 30,31] — three turn-aligned groups.
        List<List<Batch>> first =
                List.of(
                        List.of(Batch.prefill(new int[] {10, 11})),
                        List.of(Batch.prefill(new int[] {20, 21, 22})),
                        List.of(Batch.prefill(new int[] {30, 31})));
        CachedSession<FakeState> s1 = CachedSession.start(model, cache, model.newState(0, 0));
        s1.ingestGroups(first);
        assertEquals(7, s1.position(), "request 1 ingested 7 positions");

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
        assertEquals(7, s2.position(), "resume stops mid-group at the genPrompt block");

        s2.ingestGroups(second);
        assertEquals(10, s2.position(), "session ends at the full conversation length");
        assertEquals(
                List.of(40, 41, 50),
                state2.ingested,
                "only the un-restored tail is ingested, no duplicated group head");

        // And a request that resumes exactly ON a group boundary still skips whole groups.
        FakeState state3 = model.newState(0, 0);
        CachedSession<FakeState> s3 =
                CachedSession.resume(
                        model, cache, state3, new long[] {10, 11, 20, 21, 22}, 5, true);
        assertEquals(5, s3.position(), "boundary resume restores both whole groups");
        s3.ingestGroups(
                List.of(
                        List.of(Batch.prefill(new int[] {10, 11})),
                        List.of(Batch.prefill(new int[] {20, 21, 22})),
                        List.of(Batch.prefill(new int[] {60, 61}))));
        assertEquals(
                List.of(60, 61), state3.ingested, "boundary resume ingests only the new group");
    }
}
