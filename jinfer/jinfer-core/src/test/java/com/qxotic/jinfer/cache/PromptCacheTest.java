package com.qxotic.jinfer.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.BaseState;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.LanguageModel;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * The cache facade, model-agnostic: every way caching happens, driven through {@link
 * PromptCache#serve} exactly as the engine drives it - a fake model whose "generation" is the pass
 * bumping the state and reporting each token through {@link PromptCache.Serving#tail}.
 *
 * <p>The map of behaviors under test: HOT (live extension, LRU, recycling, the wiped spare, the
 * desync discard), BLOCKS (echo resume, the one-short law, the per-position tail, define +
 * full-hit, budget refusal), COARSE (define-only writes), HOT-ONLY (no codec / blocks disabled),
 * and the CATALOG (create, save, reopen, read-only, export).
 */
public final class PromptCacheTest {

    // ---- the fake family: verifiable, no model weights. The package fixture - other cache
    // tests (e.g. CachedSessionPartialGroupTest) reuse it rather than growing their own. ------

    static final int CONTEXT = 64;

    static final class FakeState extends BaseState {
        /** Every token id the model actually ingested, in order. */
        final List<Integer> ingested = new ArrayList<>();

        FakeState() {
            super(java.lang.foreign.Arena.ofAuto());
        }

        @Override
        public int contextCapacity() {
            return CONTEXT;
        }

        @Override
        public int batchCapacity() {
            return 512;
        }

        @Override
        public void reset() {
            resumeAt(0);
        }
    }

    static final Config CONFIG =
            new Config() {
                @Override
                public int vocabularySize() {
                    return 32;
                }

                @Override
                public int contextLength() {
                    return CONTEXT;
                }
            };

    static class FakeModel implements LanguageModel<Config, Object, FakeState> {
        final StateCodec<FakeState> codec;

        FakeModel(StateCodec<FakeState> codec) {
            this.codec = codec;
        }

        @Override
        public Optional<StateCodec<FakeState>> stateCodec() {
            return Optional.ofNullable(codec);
        }

        @Override
        public Config config() {
            return CONFIG;
        }

        @Override
        public Object weights() {
            return null;
        }

        @Override
        public FakeState newState(int ctx, int batch, java.lang.foreign.Arena arena) {
            return new FakeState();
        }

        @Override
        public void forward(FakeState s, Batch batch) {
            if (batch.input() instanceof Batch.Input.Tokens t) {
                for (int id : t.ids()) s.ingested.add(id);
            }
            s.position += batch.count();
        }

        @Override
        public FloatTensor head(FakeState s, int output) {
            throw new UnsupportedOperationException("no logits in a cache test");
        }
    }

    static final class FakeCodec implements StateCodec<FakeState> {
        final boolean coarse;

        FakeCodec(boolean coarse) {
            this.coarse = coarse;
        }

        @Override
        public boolean coarseBlocks() {
            return coarse;
        }

        @Override
        public long blockBytes(int positions) {
            return positions * 8L;
        }

        @Override
        public void save(FakeState s, int from, int to, MemorySegment dst) {}

        @Override
        public void restore(FakeState s, int from, int to, MemorySegment src) {}
    }

    static final byte[] SEED = {42};

    static FakeModel fine() {
        return new FakeModel(new FakeCodec(false));
    }

    static FakeModel coarse() {
        return new FakeModel(new FakeCodec(true));
    }

    static PromptCache<FakeState> cache(FakeModel model, int hot, long budget) {
        return PromptCache.of(model, SEED, new PromptCache.Options(hot, budget, null, false));
    }

    static PromptCache<FakeState> onCatalog(FakeModel model, Path catalog, boolean readOnly) {
        return PromptCache.of(model, SEED, new PromptCache.Options(0, 1 << 20, catalog, readOnly));
    }

    static List<Batch> prompt(int... ids) {
        return List.of(Batch.prefill(ids));
    }

    static List<Batch> turns(int[]... batches) {
        List<Batch> out = new ArrayList<>();
        for (int[] b : batches) out.add(Batch.prefill(b));
        return out;
    }

    /**
     * One "generation": decode {@code reply} tokens the way the engine does - the generator ingests
     * each on the state, then fires the after-ingest hook.
     */
    record Served(PromptCache.Tier tier, int restored) {}

    static Served generate(PromptCache<FakeState> cache, List<Batch> prompt, int... reply) {
        return cache.serve(
                prompt,
                (state, serving) -> {
                    for (int token : reply) {
                        state.position += 1; // model.ingest(step)
                        serving.tail(token); // afterIngest
                    }
                    return new Served(serving.tier(), serving.restored());
                });
    }

    // ---- HOT: live conversations ------------------------------------------------------------

    @Test
    void aStrictExtensionContinuesTheHotConversation() {
        try (var cache = cache(fine(), 2, 1 << 20)) {
            Served turn1 = generate(cache, turns(new int[] {1, 2, 3}), 7, 8);
            assertEquals(PromptCache.Tier.FRESH, turn1.tier());

            // turn 2 echoes turn 1's stream (prompt + reply) and adds a new turn
            Served turn2 = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8, 4}), 9);
            assertEquals(PromptCache.Tier.SESSION, turn2.tier(), "strict extension = hot");
            assertEquals(5, turn2.restored(), "everything the hot state held is reused");
        }
    }

    @Test
    void anIdenticalPromptIsNeverAHotMatch() {
        // identical = not a STRICT extension: the logits would be stale
        try (var cache = cache(fine(), 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}), 7);
            Served again = generate(cache, turns(new int[] {1, 2, 3}), 7);
            assertNotEquals(PromptCache.Tier.SESSION, again.tier());
        }
    }

    @Test
    void theLongestHotStreamWins() {
        try (var cache = cache(fine(), 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2})); // short conversation, no reply
            generate(cache, turns(new int[] {1, 2}, new int[] {3, 4})); // longer one
            Served hit = generate(cache, turns(new int[] {1, 2}, new int[] {3, 4}, new int[] {5}));
            assertEquals(PromptCache.Tier.SESSION, hit.tier());
            assertEquals(4, hit.restored(), "the deeper conversation served it");
        }
    }

    @Test
    void hotIsBoundedLruAndTheColdestFallsOut() {
        try (var cache = cache(fine(), 1, 0)) { // hot-only, cap 1
            generate(cache, turns(new int[] {1, 2}), 7); // conversation A
            generate(cache, turns(new int[] {5, 6}), 8); // conversation B evicts A (cap 1)
            Served a2 = generate(cache, turns(new int[] {1, 2}, new int[] {7, 3}));
            assertNotEquals(PromptCache.Tier.SESSION, a2.tier(), "A's live state is gone");
            assertEquals(1, cache.sample().hotSessions(), "the hot layer stays bounded");
        }
    }

    // ---- THE POOL IS THE ALLOCATOR ----------------------------------------------------------

    @Test
    void theStatelessDefaultAllocatesItsContextOnce() {
        try (var cache = cache(fine(), 0, 0)) {
            for (int i = 0; i < 5; i++) generate(cache, turns(new int[] {1, 2, 10 + i}), 7);
            PromptCache.Sample s = cache.sample();
            assertEquals(1, s.statesAllocated(), "five requests, ONE context: " + s);
            assertEquals(0, s.hotSessions(), "and no conversation is retained");
            assertEquals(0, s.hotHits(), "the wiped spare matches nothing");
        }
    }

    @Test
    void atCapacityTheColdestAllocationIsRecycledNotDropped() {
        try (var cache = cache(fine(), 1, 0)) {
            generate(cache, turns(new int[] {1, 2}), 7);
            generate(cache, turns(new int[] {5, 6}), 8); // unrelated: recycles A's state
            assertEquals(1, cache.sample().statesAllocated(), "a full hot layer never allocates");
        }
    }

    // ---- BLOCKS: content-keyed KV -----------------------------------------------------------

    @Test
    void anEchoedConversationResumesFromBlocksAfterTheHotStateIsGone() {
        try (var cache = cache(fine(), 1, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}), 7, 8); // A: committed as it went
            generate(cache, turns(new int[] {21, 22}), 9); // B recycles A's live state
            // A's turn-2 echo: hot is gone, blocks serve prompt+reply token-exact
            Served turn2 = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8, 4}));
            assertEquals(PromptCache.Tier.BLOCKS, turn2.tier());
            assertEquals(5, turn2.restored(), "prompt chunk + per-token reply all resume");
        }
    }

    @Test
    void aTruncatedEchoResumesAtItsExactCut() {
        // THE TAIL CONTRACT: the reply commits per position, so an echo cut mid-reply (a stop
        // string, an edited tail) resumes at its own divergence, not at a chunk boundary
        try (var cache = cache(fine(), 0, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}), 7, 8, 9);
            Served cut = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 99}));
            assertEquals(PromptCache.Tier.BLOCKS, cut.tier());
            assertEquals(4, cut.restored(), "resumes exactly at the cut: [1,2,3,7|99]");
        }
    }

    @Test
    void theOneShortLawLeavesTheFinalTokenToRecompute() {
        try (var cache = cache(fine(), 0, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}), 7, 8);
            // the identical stream again: everything is cached, yet one token must re-ingest
            Served again = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8}));
            assertEquals(4, again.restored(), "5 cached positions, one-short law restores 4");
        }
    }

    @Test
    void defineThenServeIsAFullHit() {
        // the CLI / withCachedPrompt shape: a defined single-batch prompt serves one-short
        try (var cache = cache(fine(), 0, 1 << 20)) {
            cache.define(prompt(1, 2, 3, 4, 5));
            Served hit = generate(cache, prompt(1, 2, 3, 4, 5), 7);
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier());
            assertEquals(4, hit.restored(), "define pins the final position as its own block");
        }
    }

    @Test
    void defineThatTheBudgetRefusesThrowsInsteadOfCachingNothing() {
        try (var cache = cache(fine(), 0, 8)) { // one position fits
            assertThrows(IllegalStateException.class, () -> cache.define(prompt(1, 2, 3, 4, 5)));
        }
    }

    @Test
    void aBudgetRefusalDuringServingKeepsTheConversationHot() {
        // budget admits the prompt chunk, refuses the reply singles: the tree DETACHES but the
        // session pools and tier-1 keeps working off the live KV
        try (var cache = cache(fine(), 1, 24)) { // 3 positions
            generate(cache, turns(new int[] {1, 2, 3}), 7, 8);
            assertTrue(cache.sample().refusals() > 0, "the refusal is counted, not silent");
            Served turn2 = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8, 4}));
            assertEquals(PromptCache.Tier.SESSION, turn2.tier(), "hot survives the detach");
        }
    }

    @Test
    void aPassThatBypassesTailIsDiscardedNotPooled() {
        // stream and state disagreeing would match a future prompt against DIFFERENT content
        try (var cache = cache(fine(), 2, 1 << 20)) {
            cache.serve(
                    turns(new int[] {1, 2, 3}),
                    (state, serving) -> {
                        state.position += 2; // decode tokens never reported through tail()
                        return null;
                    });
            assertEquals(0, cache.sample().hotSessions(), "a desynced session must never pool");
        }
    }

    @Test
    void aThrowingPassDiscardsTheSessionButKeepsItsBlocks() {
        try (var cache = cache(fine(), 2, 1 << 20)) {
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            cache.serve(
                                    turns(new int[] {1, 2, 3}),
                                    (state, serving) -> {
                                        throw new IllegalStateException("torn");
                                    }));
            assertEquals(0, cache.sample().hotSessions(), "a torn state never serves again");
            // the prompt chunk committed before the throw still serves an echo
            Served echo = generate(cache, turns(new int[] {1, 2, 3}, new int[] {4}));
            assertEquals(PromptCache.Tier.BLOCKS, echo.tier());
            assertEquals(3, echo.restored());
        }
    }

    // ---- COARSE: blocks written by define() alone -------------------------------------------

    @Test
    void coarseServingRestoresDefinedPrefixesAndCommitsNothing() {
        try (var cache = cache(coarse(), 0, 1 << 20)) {
            cache.define(turns(new int[] {1, 2, 3, 4}, new int[] {5})); // one block: [1,2,3,4]
            assertTrue(cache.treeStats().startsWith("blocks=1 "), cache.treeStats());

            Served hit = generate(cache, turns(new int[] {1, 2, 3, 4}, new int[] {9}), 7, 8);
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier());
            assertEquals(4, hit.restored(), "the defined block serves");
            assertTrue(
                    cache.treeStats().startsWith("blocks=1 "),
                    "a served turn must not write a ~90MB residue: " + cache.treeStats());
        }
    }

    @Test
    void aSingleBatchCoarseDefineStillServesOneShort() {
        // the CLI --cache shape: one token batch. Committed whole it would be a dead block
        // (a one-short serve can never match it); define must commit the prefix-only block
        try (var cache = cache(coarse(), 0, 1 << 20)) {
            cache.define(prompt(1, 2, 3, 4, 5));
            assertTrue(cache.treeStats().startsWith("blocks=1 "), cache.treeStats());
            Served hit = generate(cache, prompt(1, 2, 3, 4, 5), 7);
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier());
            assertEquals(4, hit.restored(), "all but the trailing position restores");
        }
    }

    // ---- HOT-ONLY: no codec, or blocks disabled ---------------------------------------------

    @Test
    void aCodecLessModelStillGetsHotConversations() {
        try (var cache = cache(new FakeModel(null), 2, 1 << 20)) {
            assertFalse(cache.blockCaching());
            generate(cache, turns(new int[] {1, 2, 3}), 7, 8);
            Served turn2 = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8, 4}));
            assertEquals(PromptCache.Tier.SESSION, turn2.tier(), "hot reuse needs no codec");
            assertEquals(5, turn2.restored());
            assertThrows(IllegalStateException.class, () -> cache.define(prompt(1, 2)));
        }
    }

    @Test
    void blocksDisabledBehavesHotOnly() {
        try (var cache = cache(fine(), 1, 0)) {
            assertFalse(cache.blockCaching(), "budget 0 = jinfer.promptCache=false");
            generate(cache, turns(new int[] {1, 2, 3}), 7);
            generate(cache, turns(new int[] {21}), 8); // recycles the live state
            Served echo = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 4}));
            assertEquals(PromptCache.Tier.FRESH, echo.tier(), "nothing was retained");
        }
    }

    // ---- THE CATALOG: the block layer on a file ---------------------------------------------

    @Test
    void theCatalogSurvivesARestart() throws Exception {
        Path catalog = Files.createTempDirectory("jinfer-pc").resolve("catalog.jkvf");
        FakeModel model = fine();
        try (var first = onCatalog(model, catalog, false)) {
            generate(first, turns(new int[] {1, 2, 3}), 7, 8);
            first.save();
        }
        assertTrue(Files.size(catalog) > 0, "save wrote the catalog");

        try (var second = onCatalog(model, catalog, false)) {
            Served echo = generate(second, turns(new int[] {1, 2, 3}, new int[] {7, 8, 4}));
            assertEquals(PromptCache.Tier.BLOCKS, echo.tier(), "yesterday's blocks serve today");
            assertEquals(5, echo.restored());
            generate(second, turns(new int[] {50, 51}), 9);
            second.save(); // append-only accumulation across boots
        }
        try (var third = onCatalog(model, catalog, false)) {
            assertEquals(
                    PromptCache.Tier.BLOCKS,
                    generate(third, turns(new int[] {50, 51}, new int[] {9, 1})).tier(),
                    "the second boot's traffic survived too");
        }
    }

    @Test
    void aReadOnlyCatalogServesButNeverGrows() throws Exception {
        Path catalog = Files.createTempDirectory("jinfer-pc").resolve("ro.jkvf");
        FakeModel model = fine();
        try (var writer = onCatalog(model, catalog, false)) {
            writer.define(prompt(1, 2, 3, 4));
            writer.save();
        }
        long size = Files.size(catalog);

        try (var ro = onCatalog(model, catalog, true)) {
            Served hit = generate(ro, turns(new int[] {1, 2, 3, 4}, new int[] {9}), 7);
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier(), "the artifact serves");
            ro.save(); // must be a no-op
        }
        assertEquals(size, Files.size(catalog), "read-only: served, never written");
    }

    @Test
    void aMissingReadOnlyCatalogDegradesToServingWithoutIt() {
        FakeModel model = fine();
        Path missing = Path.of("/nonexistent/jinfer/catalog.jkvf");
        try (var cache =
                PromptCache.of(model, SEED, new PromptCache.Options(1, 1 << 20, missing, true))) {
            assertTrue(cache.blockCaching(), "RAM blocks still work");
            generate(cache, turns(new int[] {1, 2}), 7);
            cache.save(); // no-op, must not try to create the file
        }
        assertFalse(Files.exists(missing));
    }

    @Test
    void exportRefusesItsOwnCatalogButWritesAFreshArtifact() throws Exception {
        Path dir = Files.createTempDirectory("jinfer-pc");
        Path catalog = dir.resolve("own.jkvf");
        FakeModel model = fine();
        try (var cache = onCatalog(model, catalog, false)) {
            cache.define(prompt(1, 2, 3));
            assertThrows(IllegalStateException.class, () -> cache.export(catalog));
            Path fresh = dir.resolve("export.jkvf");
            cache.export(fresh);
            try (var mounted = onCatalog(model, fresh, true)) {
                assertEquals(
                        PromptCache.Tier.BLOCKS,
                        generate(mounted, turns(new int[] {1, 2, 3}, new int[] {9})).tier());
            }
        }
    }

    // ---- review-pinned edges ----------------------------------------------------------------

    @Test
    void budgetZeroWithCatalogServesTheMountButRefusesGrowth() throws Exception {
        // the jinfer.promptCache=false + --cache combination: mounted blocks serve, RAM refuses
        Path catalog = Files.createTempDirectory("jinfer-pc").resolve("b0.jkvf");
        FakeModel model = fine();
        try (var writer = onCatalog(model, catalog, false)) {
            writer.define(prompt(1, 2, 3, 4));
            writer.save();
        }
        long size = Files.size(catalog);
        try (var frozen =
                PromptCache.of(model, SEED, new PromptCache.Options(0, 0, catalog, false))) {
            Served hit = generate(frozen, turns(new int[] {1, 2, 3, 4}, new int[] {9}), 7);
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier(), "the mount serves");
            assertTrue(frozen.sample().refusals() > 0, "growth is refused, and counted");
            assertThrows(IllegalStateException.class, () -> frozen.define(prompt(8, 9)));
            frozen.save();
        }
        assertEquals(size, Files.size(catalog), "nothing new ever reaches the file");
    }

    @Test
    void coarseRestoreEndingInsideABatchSlicesTheTail() {
        // the defined coarse block ends mid-batch of the request: the read-only session must
        // slice the group head it restored and ingest only the tail - still committing nothing
        try (var cache = cache(coarse(), 0, 1 << 20)) {
            cache.define(turns(new int[] {1, 2, 3, 4}, new int[] {5})); // one block: [1,2,3,4]
            Served hit = generate(cache, prompt(1, 2, 3, 4, 9), 7); // ONE batch, seam at 4
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier());
            assertEquals(4, hit.restored(), "restored to the block edge inside the batch");
            assertTrue(cache.treeStats().startsWith("blocks=1 "), cache.treeStats());
        }
    }

    @Test
    void defineDedupsAndDefineAfterServeStillFullHits() {
        try (var cache = cache(fine(), 0, 1 << 20)) {
            // traffic first: the prompt commits at chunk boundaries, no split-last single
            generate(cache, prompt(1, 2, 3, 4, 5), 7);
            // define AFTER the serve: the capped resume must still commit the final single -
            // an uncapped resume would dedup into the chunk and silently break the promise
            cache.define(prompt(1, 2, 3, 4, 5));
            Served hit = generate(cache, prompt(1, 2, 3, 4, 5), 8);
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier());
            assertEquals(4, hit.restored(), "define-after-serve still yields the full hit");

            // define twice: pure dedup - no new blocks, no bytes, no budget-refusal misread
            int blocksBefore = cache.sample().blocks();
            long bytesBefore = cache.sample().bytes();
            cache.define(prompt(1, 2, 3, 4, 5));
            assertEquals(blocksBefore, cache.sample().blocks(), "a re-define adds no blocks");
            assertEquals(bytesBefore, cache.sample().bytes(), "and no bytes");
        }
    }

    @Test
    void aHotHitWithAThrowingPassIsDiscardedButItsBlocksSurvive() {
        try (var cache = cache(fine(), 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}), 7, 8);
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            cache.serve(
                                    turns(new int[] {1, 2, 3}, new int[] {7, 8, 4}),
                                    (state, serving) -> {
                                        throw new IllegalStateException("torn mid-hit");
                                    }));
            assertEquals(0, cache.sample().hotSessions(), "the acquired session is gone");
            Served echo = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8, 5}));
            assertEquals(PromptCache.Tier.BLOCKS, echo.tier(), "its committed blocks survive");
            assertEquals(5, echo.restored());
        }
    }

    @Test
    void optionsRejectNonsenseInsteadOfBuildingAPathologicalCache() {
        // int widens silently into the long slot: a transposed (budget, hot) pair must not
        // become a million-session hot layer with a 4-byte budget
        assertThrows(
                IllegalArgumentException.class, () -> new PromptCache.Options(-1, 1, null, false));
        assertThrows(
                IllegalArgumentException.class, () -> new PromptCache.Options(1, -1, null, false));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        PromptCache.of(
                                new FakeModel(null),
                                null,
                                new PromptCache.Options(0, 0, null, false)));
    }

    @Test
    void theGuardsFailLoudly() {
        var cache = cache(fine(), 1, 1 << 20);
        // empty prompts are a caller bug, not a model crash
        assertThrows(IllegalArgumentException.class, () -> generate(cache, List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> generate(cache, prompt(new int[CONTEXT + 1])),
                "past the context, before any ingest");
        // a stashed Serving handle is dead once the pass returns
        PromptCache.Serving[] stashed = new PromptCache.Serving[1];
        cache.serve(
                turns(new int[] {1, 2}),
                (state, serving) -> {
                    stashed[0] = serving;
                    return null;
                });
        assertThrows(IllegalStateException.class, () -> stashed[0].tail(9));
        // and a closed cache refuses everything
        cache.close();
        cache.close(); // idempotent
        assertThrows(IllegalStateException.class, () -> generate(cache, prompt(1)));
        assertThrows(IllegalStateException.class, cache::sample);
        assertThrows(IllegalStateException.class, () -> cache.define(prompt(1, 2)));
        assertThrows(IllegalStateException.class, cache::save);
    }

    // ---- lifecycle --------------------------------------------------------------------------

    @Test
    void closeReachesEveryHotStateAndTheSpare() {
        FakeModel model = fine();
        var warm = cache(model, 1, 1 << 20);
        FakeState pooled = warm.serve(turns(new int[] {1, 2}), (state, serving) -> state);
        warm.close();
        assertTrue(pooled.isClosed(), "close frees the pooled state deterministically");

        var stateless = cache(model, 0, 0);
        FakeState spare = stateless.serve(turns(new int[] {1, 2}), (state, serving) -> state);
        stateless.close();
        assertTrue(spare.isClosed(), "close frees the spare too");
    }
}
