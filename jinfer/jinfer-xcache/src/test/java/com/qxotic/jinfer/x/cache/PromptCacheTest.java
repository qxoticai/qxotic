package com.qxotic.jinfer.x.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.boundary.BaseState;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Config;
import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jinfer.x.boundary.LanguageModel;
import com.qxotic.jinfer.x.boundary.StateCodec;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

/**
 * The cache facade, model-agnostic: every way caching happens, driven through {@link
 * PromptCache#serve} exactly as the engine drives it - a fake model whose "generation" is the pass
 * bumping the state and reporting each token through {@link PromptCache.Serving#tail}.
 *
 * <p>The map of behaviors under test: SESSIONS (live extension, LRU, recycling, desync discard),
 * BLOCKS (echo resume, the one-short law, the per-position tail, define + full-hit, budget
 * refusal), COARSE (define-only writes), SESSIONS-ONLY (no codec / blocks disabled), and the
 * CATALOG (create, save, reopen, read-only, export).
 */
public final class PromptCacheTest {

    // ---- the fake family: verifiable, no model weights. The package fixture - other cache
    // tests (e.g. CachedSessionPartialGroupTest) reuse it rather than growing their own. ------

    static final int CONTEXT = 64;

    /**
     * The cache allocates states of this size, and a prompt past it is refused. Matching the fake
     * model's own context length keeps the two bounds - trained-for and allocated - coincident,
     * which is what the guard tests below are written against.
     */
    private static final int CTX = CONTEXT;

    static final class FakeState extends BaseState {
        /** Every token id the model actually ingested, in order. */
        final List<Integer> ingested = new ArrayList<>();

        private final int batchCap;

        FakeState() {
            this(512);
        }

        FakeState(int batchCap) {
            super(Arena.ofAuto());
            this.batchCap = batchCap;
        }

        @Override
        public int contextCapacity() {
            return CONTEXT;
        }

        @Override
        public int batchCapacity() {
            return batchCap;
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

        /** Test knob: the batch capacity of allocated states (chunk granularity of prefill). */
        int stateBatch = 512;

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
        public FakeState newState(int ctx, int batch, Arena arena) {
            return new FakeState(stateBatch);
        }

        @Override
        public void forward(FakeState s, Batch batch) {
            if (batch.input() instanceof Batch.Input.Tokens t) {
                for (int id : t.ids()) s.ingested.add(id);
            }
            s.position += batch.count();
        }

        @Override
        public MemoryView<?> head(FakeState s, int output) {
            throw new UnsupportedOperationException("no logits in a cache test");
        }
    }

    static class FakeCodec implements StateCodec<FakeState> {
        final boolean coarse;

        FakeCodec(boolean coarse) {
            this.coarse = coarse;
        }

        @Override
        public boolean coarseCheckpoints() {
            return coarse;
        }

        @Override
        public long checkpointBytes(int positions) {
            return positions * 8L;
        }

        @Override
        public void saveCheckpoint(FakeState s, int from, int to, MemorySegment dst) {}

        @Override
        public void restoreCheckpoint(FakeState s, int from, int to, MemorySegment src) {}
    }

    static final ContentKey SEED = ContentKey.sha256(new byte[] {42});

    static FakeModel fine() {
        return new FakeModel(new FakeCodec(false));
    }

    /** A codec with a 100-byte residue trailer: the LFM2-class shape, scaled down. */
    static FakeModel residual() {
        return new FakeModel(
                new FakeCodec(false) {
                    @Override
                    public long checkpointBytes(int positions) {
                        return positions * 8L + 100;
                    }
                });
    }

    static FakeModel coarse() {
        return new FakeModel(new FakeCodec(true));
    }

    static PromptCache<FakeState> cache(FakeModel model, int retained, long budget) {
        return PromptCache.of(
                model, SEED, new PromptCache.Options(retained, CTX, budget, null, false));
    }

    static PromptCache<FakeState> onCatalog(FakeModel model, Path catalog, boolean readOnly) {
        return PromptCache.of(
                model, SEED, new PromptCache.Options(0, CTX, 1 << 20, catalog, readOnly));
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

    // ---- RETAINED SESSIONS: live conversations ----------------------------------------------

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
        try (var cache = cache(fine(), 1, 0)) { // sessions-only, cap 1
            generate(cache, turns(new int[] {1, 2}), 7); // conversation A
            generate(cache, turns(new int[] {5, 6}), 8); // conversation B evicts A (cap 1)
            Served a2 = generate(cache, turns(new int[] {1, 2}, new int[] {7, 3}));
            assertNotEquals(PromptCache.Tier.SESSION, a2.tier(), "A's live state is gone");
            assertEquals(1, cache.sample().retainedSessions(), "the session layer stays bounded");
        }
    }

    // ---- THE POOL IS THE ALLOCATOR ----------------------------------------------------------

    @Test
    void zeroClosesTheStateAfterEveryRequest() {
        try (var cache = cache(fine(), 0, 0)) {
            List<FakeState> states = new ArrayList<>();
            for (int i = 0; i < 3; i++) {
                states.add(cache.serve(turns(new int[] {1, 2, 10 + i}), (state, serving) -> state));
            }
            PromptCache.Sample s = cache.sample();
            assertEquals(3, s.stateAllocations());
            assertEquals(0, s.retainedSessions());
            assertEquals(0, s.sessionHits());
            assertEquals(3, states.stream().distinct().count());
            assertTrue(states.stream().allMatch(FakeState::isClosed));
        }
    }

    @Test
    void atCapacityTheColdestAllocationIsRecycledNotDropped() {
        try (var cache = cache(fine(), 1, 0)) {
            generate(cache, turns(new int[] {1, 2}), 7);
            generate(cache, turns(new int[] {5, 6}), 8); // unrelated: recycles A's state
            assertEquals(
                    1, cache.sample().stateAllocations(), "a full retained layer never allocates");
        }
    }

    @Test
    void retainedStatesStayBoundedAndRecycleLru() {
        try (var cache = cache(fine(), 2, 0)) {
            generate(cache, turns(new int[] {1, 2}), 7);
            generate(cache, turns(new int[] {3, 4}), 8);
            generate(cache, turns(new int[] {5, 6}), 9);
            assertEquals(2, cache.sample().stateAllocations());
            assertEquals(2, cache.sample().retainedSessions());
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
    void aResidueCodecCommitsTheReplyAsOneBlock() {
        // per-token singles would duplicate the residue per generated token (measured 300x the
        // row bytes on LFM2.5); a residue codec's tail commits ONCE per reply instead
        try (var cache = cache(residual(), 0, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}), 7, 8, 9);
            assertEquals(
                    2,
                    cache.sample().blocks(),
                    "one prompt chunk + ONE reply block, not one per token");
            // the echoed conversation still resumes through the whole reply (a turn boundary)
            Served turn2 = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8, 9, 4}));
            assertEquals(PromptCache.Tier.BLOCKS, turn2.tier());
            assertEquals(6, turn2.restored(), "prompt chunk + whole-reply block resume");
        }
    }

    @Test
    void aResidueCodecEchoCutMidReplyResumesAtTheTurnBoundary() {
        // the accepted trade: no per-token singles means a mid-reply cut re-prefills the reply
        // tail from the last turn boundary instead of resuming token-exact
        try (var cache = cache(residual(), 0, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}), 7, 8, 9);
            Served cut = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 99}));
            assertEquals(3, cut.restored(), "resumes at the prompt-chunk boundary, not the cut");
        }
    }

    // ---- COARSE TAIL SNAPSHOT: the rewind that saves thinking models from full re-prefill ----

    // the chat shape: the prompt's FINAL batch is the generation prompt (template convention);
    // 90 plays that role - the echoed next turn re-renders history differently past the seam
    private static final int GEN = 90;

    @Test
    void aStrippedEchoRewindsToTheCoarseTailSnapshot() {
        // the generated stream ends [.., GEN, 7, 8] (gen prompt + reply-with-thinking); the echo
        // renders the reply's turn as [9] - never a strict extension, so before the snapshot
        // this was a full re-prefill on every coarse turn
        try (var cache = cache(coarse(), 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}, new int[] {GEN}), 7, 8);
            Served turn2 = generate(cache, turns(new int[] {1, 2, 3, 9, 4}, new int[] {GEN}));
            assertEquals(PromptCache.Tier.SESSION, turn2.tier(), "snapshot rewind = hot tier");
            assertEquals(3, turn2.restored(), "rewound to the pre-gen-prompt seam");
        }
    }

    @Test
    void theSnapshotAdvancesToEachNewPromptBoundary() {
        try (var cache = cache(coarse(), 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}, new int[] {GEN}), 7, 8);
            generate(cache, turns(new int[] {1, 2, 3, 9, 4}, new int[] {GEN}), 8);
            // turn 3 echoes turn 2's stripped stream: rewinds to turn 2's seam (5), not 3
            Served turn3 = generate(cache, turns(new int[] {1, 2, 3, 9, 4, 6, 5}, new int[] {GEN}));
            assertEquals(PromptCache.Tier.SESSION, turn3.tier());
            assertEquals(5, turn3.restored(), "the snapshot follows the newest prompt boundary");
        }
    }

    @Test
    void aForkBeforeTheSnapshotStillRePrefills() {
        // the snapshot is the TAIL rewind point only: recurrent state cannot rewind further
        try (var cache = cache(coarse(), 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}, new int[] {GEN}), 7, 8);
            Served fork = generate(cache, turns(new int[] {1, 99, 3}, new int[] {GEN}));
            assertEquals(PromptCache.Tier.FRESH, fork.tier());
            assertEquals(0, fork.restored());
        }
    }

    @Test
    void fineCodecsTakeNoSnapshot() {
        try (var cache = cache(fine(), 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}, new int[] {GEN}), 7);
            assertEquals(
                    0, cache.sample().sessionSnapshotBytes(), "snapshots are the coarse-codec fix");
        }
        try (var cache = cache(coarse(), 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}, new int[] {GEN}), 7);
            assertTrue(
                    cache.sample().sessionSnapshotBytes() > 0,
                    "a served coarse turn holds a snapshot");
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
        // the withCachedPrompt shape: a defined single-batch prompt serves one-short
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
            assertTrue(cache.sample().blockRefusals() > 0, "the refusal is counted, not silent");
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
            assertEquals(
                    0, cache.sample().retainedSessions(), "a desynced session must never pool");
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
            assertEquals(0, cache.sample().retainedSessions(), "a torn state never serves again");
            // the prompt chunk committed before the throw still serves an echo
            Served echo = generate(cache, turns(new int[] {1, 2, 3}, new int[] {4}));
            assertEquals(PromptCache.Tier.BLOCKS, echo.tier());
            assertEquals(3, echo.restored());
        }
    }

    @Test
    void zeroClosesAStateWhenThePassThrows() {
        FakeState[] acquired = new FakeState[1];
        try (var cache = cache(fine(), 0, 0)) {
            assertThrows(
                    IllegalStateException.class,
                    () ->
                            cache.serve(
                                    prompt(1, 2),
                                    (state, serving) -> {
                                        acquired[0] = state;
                                        throw new IllegalStateException("boom");
                                    }));
            assertTrue(acquired[0].isClosed());
            assertEquals(0, cache.sample().retainedSessions());
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
        // one token batch. Committed whole it would be a dead block (a one-short serve can never
        // match it); define must commit the prefix-only block
        try (var cache = cache(coarse(), 0, 1 << 20)) {
            cache.define(prompt(1, 2, 3, 4, 5));
            assertTrue(cache.treeStats().startsWith("blocks=1 "), cache.treeStats());
            Served hit = generate(cache, prompt(1, 2, 3, 4, 5), 7);
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier());
            assertEquals(4, hit.restored(), "all but the trailing position restores");
        }
    }

    // ---- SESSIONS-ONLY: no codec, or blocks disabled ----------------------------------------

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
            assertFalse(cache.blockCaching(), "budget 0 = blocks off");
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
    void aMissingReadOnlyCatalogFailsLoudly() {
        FakeModel model = fine();
        Path missing = Path.of("/nonexistent/jinfer/catalog.jkvf");
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        PromptCache.of(
                                model,
                                SEED,
                                new PromptCache.Options(1, CTX, 1 << 20, missing, true)));
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
        // the blocks-off + catalog combination: mounted blocks serve, RAM refuses growth
        Path catalog = Files.createTempDirectory("jinfer-pc").resolve("b0.jkvf");
        FakeModel model = fine();
        try (var writer = onCatalog(model, catalog, false)) {
            writer.define(prompt(1, 2, 3, 4));
            writer.save();
        }
        long size = Files.size(catalog);
        try (var frozen =
                PromptCache.of(model, SEED, new PromptCache.Options(0, CTX, 0, catalog, false))) {
            Served hit = generate(frozen, turns(new int[] {1, 2, 3, 4}, new int[] {9}), 7);
            assertEquals(PromptCache.Tier.BLOCKS, hit.tier(), "the mount serves");
            assertTrue(frozen.sample().blockRefusals() > 0, "growth is refused, and counted");
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
            assertEquals(0, cache.sample().retainedSessions(), "the acquired session is gone");
            Served echo = generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8, 5}));
            assertEquals(PromptCache.Tier.BLOCKS, echo.tier(), "its committed blocks survive");
            assertEquals(5, echo.restored());
        }
    }

    @Test
    void optionsRejectNonsenseInsteadOfBuildingAPathologicalCache() {
        // int widens silently into the long slot: a transposed (budget, hot) pair must not
        // become a million-session retained layer with a 4-byte budget
        assertThrows(
                IllegalArgumentException.class,
                () -> new PromptCache.Options(-1, CTX, 1, null, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> new PromptCache.Options(1, CTX, -1, null, false));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        PromptCache.of(
                                new FakeModel(null),
                                null,
                                new PromptCache.Options(0, CTX, 0, null, false)));
    }

    @Test
    void aCatalogWithoutAStateCodecIsRefused() {
        // nothing could ever be written to the file: silently ignoring it would degrade the
        // configured cache into an unnoticed cold start
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        PromptCache.of(
                                new FakeModel(null),
                                SEED,
                                new PromptCache.Options(
                                        0, CTX, 0, Path.of("never-created.jcache"), false)));
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
        assertThrows(
                IllegalArgumentException.class,
                () -> cache.define(prompt(new int[CONTEXT + 1])),
                "an over-long define would append junk blocks no serve could match");
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

    // ---- cooperative prefill interrupt ------------------------------------------------------

    @Test
    void interruptedPrefillRunsThePassAndDiscardsTheSession() {
        FakeModel model = fine();
        model.stateBatch = 4; // a 12-token prompt ingests as 3 chunks
        try (var cache = cache(model, 2, 1 << 20)) {
            AtomicInteger calls = new AtomicInteger();
            Object marker =
                    cache.serve(
                            turns(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}),
                            (state, serving) -> {
                                assertFalse(serving.prefillComplete());
                                assertEquals(4, state.position); // chunk 0 completed, 1 never ran
                                assertEquals(PromptCache.Tier.FRESH, serving.tier());
                                return "stopped";
                            },
                            () -> calls.incrementAndGet() > 1);
            assertEquals("stopped", marker); // the pass ran and serve returned its result
            // the partially-served SESSION was discarded, but chunk 0's committed block survives:
            // the same prompt restores that prefix instead of prefilling from scratch
            Served again =
                    generate(cache, turns(new int[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}));
            assertEquals(PromptCache.Tier.BLOCKS, again.tier());
            assertEquals(4, again.restored());
        }
    }

    @Test
    void interruptBeforeTheFirstChunkIngestsNothing() {
        FakeModel model = fine();
        try (var cache = cache(model, 2, 1 << 20)) {
            Object marker =
                    cache.serve(
                            turns(new int[] {1, 2, 3}),
                            (state, serving) -> {
                                assertFalse(serving.prefillComplete());
                                assertEquals(0, state.position);
                                assertEquals(0, serving.restored());
                                assertTrue(serving.promptTime().toNanos() >= 0); // set even here
                                return "stopped";
                            },
                            () -> true);
            assertEquals("stopped", marker);
            Served again = generate(cache, turns(new int[] {1, 2, 3}));
            assertEquals(PromptCache.Tier.FRESH, again.tier()); // nothing committed or retained
        }
    }

    @Test
    void completedPrefillRetainsTheSession() {
        FakeModel model = fine();
        try (var cache = cache(model, 2, 1 << 20)) {
            generate(cache, turns(new int[] {1, 2, 3}), 7);
            AtomicInteger calls = new AtomicInteger();
            // fires only far past completion: a consulted-but-unfired interrupt changes nothing
            Served hit =
                    cache.serve(
                            turns(new int[] {1, 2, 3}, new int[] {7, 8}),
                            (state, serving) -> {
                                assertTrue(serving.prefillComplete());
                                return new Served(serving.tier(), serving.restored());
                            },
                            () -> calls.incrementAndGet() > 100);
            assertEquals(PromptCache.Tier.SESSION, hit.tier());
            Served next =
                    generate(cache, turns(new int[] {1, 2, 3}, new int[] {7, 8}, new int[] {9}));
            assertEquals(PromptCache.Tier.SESSION, next.tier()); // the session survived
        }
    }

    // ---- lifecycle --------------------------------------------------------------------------

    @Test
    void closeReachesEveryRetainedStateAndZeroAlreadyClosedItsState() {
        FakeModel model = fine();
        var warm = cache(model, 1, 1 << 20);
        FakeState pooled = warm.serve(turns(new int[] {1, 2}), (state, serving) -> state);
        warm.close();
        assertTrue(pooled.isClosed(), "close frees the pooled state deterministically");

        var stateless = cache(model, 0, 0);
        FakeState state = stateless.serve(turns(new int[] {1, 2}), (s, serving) -> s);
        assertTrue(state.isClosed(), "zero closes the state as soon as the request completes");
        stateless.close();
        assertTrue(state.isClosed(), "cache close stays idempotent");
    }
}
