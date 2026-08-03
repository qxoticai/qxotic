package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.RuntimeState;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * jinfer's KV cache - the one front door to every way caching happens. Two layers, one law:
 *
 * <ul>
 *   <li>HOT - the last N conversations stay live as ready-to-continue states; a prompt that
 *       strictly extends one continues in place (ANY model, codec or not).
 *   <li>BLOCKS - everything computed is kept as content-keyed KV blocks (RAM, budget-bounded,
 *       optionally backed by a catalog file that survives restarts). Interior content commits at
 *       turn boundaries; the decode tail commits per position, so a truncated or edited echo
 *       resumes token-exact.
 * </ul>
 *
 * <p>THE LAW, stated once: a resume always stops one position short, so the final token re-ingests
 * and the logits are always fresh - and a cached answer is byte-identical to a cold one.
 *
 * <p>{@link #of} reads the model's capabilities itself: no {@link StateCodec} = hot-only; a
 * coarse-residue codec = blocks written by {@link #define} alone (a served turn would cost a ~90MB
 * residue per block); otherwise the full picture. Callers make zero routing decisions.
 *
 * <p>Single-threaded by design: every method (including {@link #sample}) belongs to the one
 * generation thread, like the tree it fronts. The low-level layer ({@link CachedSession}, {@link
 * BlockTree}, {@link FrozenBlocks}) stays public for the testkit, benches and speculative decoding;
 * production callers should not need it.
 */
public final class PromptCache<S extends RuntimeState> implements AutoCloseable {

    /**
     * @param hotSessions live conversations retained; 0 = stateless between requests (the one
     *     allocation is still recycled as a wiped spare)
     * @param blockBudgetBytes RAM bound for the block layer; 0 = blocks disabled (hot-only) - an
     *     explicit {@code catalog} still mounts, read-only in spirit if the budget refuses growth
     * @param catalog the block layer's file, opened if present and CREATED OTHERWISE (so {@link
     *     #save} is always an append - never a rewrite of a mounted mapping); null = RAM only
     * @param readOnly the catalog is served, never written: {@link #save} is a no-op, and a missing
     *     or incompatible file degrades to serving without it instead of failing the boot
     */
    public record Options(int hotSessions, long blockBudgetBytes, Path catalog, boolean readOnly) {

        /** In-memory defaults: {@code hot} live sessions, the standard budget, no catalog. */
        public static Options inMemory(int hot, long budgetBytes) {
            return new Options(hot, budgetBytes, null, false);
        }
    }

    private final LanguageModel<?, ?, S> model;
    private final BlockTree<S> tree; // null = hot-only (no codec, or blocks disabled)
    private final boolean coarse; // blocks written by define() alone; serving restores read-only
    private final CacheStore store; // owned; freed at close
    private final Path catalog; // the effective mounted/created file (null = none / degraded)
    private final boolean readOnly;

    // ---- the HOT layer: THE POOL IS THE ALLOCATOR ------------------------------------------
    // A full context is the dominant per-pipeline allocation. At capacity the least-recent
    // session's state is recycled (reset() - every family has decided what a fresh sequence
    // must clear); with hotSessions=0 one WIPED bare allocation is retained as the spare, so
    // the stateless default still allocates its context once and keeps none of the content.
    private final int hotCapacity;
    private final ArrayDeque<CachedSession<S>> hot = new ArrayDeque<>();
    private S spare;
    private long hotHits;
    private long statesAllocated; // steady state: max(1, hotSessions)

    private PromptCache(
            LanguageModel<?, ?, S> model,
            BlockTree<S> tree,
            boolean coarse,
            CacheStore store,
            Path catalog,
            Options options) {
        this.model = model;
        this.tree = tree;
        this.coarse = coarse;
        this.store = store;
        this.catalog = catalog;
        this.readOnly = options.readOnly();
        this.hotCapacity = Math.max(0, options.hotSessions());
    }

    /**
     * Builds the cache the {@code model} can support - the only constructor. {@code seed} is the
     * model's cache identity (see {@link #modelSeed}); block keys and catalog files are rooted in
     * it, so two models can never match each other's blocks.
     */
    public static <S extends RuntimeState> PromptCache<S> of(
            LanguageModel<?, ?, S> model, byte[] seed, Options o) {
        StateCodec<S> codec = model.stateCodec().orElse(null);
        boolean wantBlocks = codec != null && (o.blockBudgetBytes() > 0 || o.catalog() != null);
        if (!wantBlocks) {
            return new PromptCache<>(model, null, false, CacheStore.inMemory(), null, o);
        }
        CacheStore store = CacheStore.inMemory();
        try {
            FrozenBlocks base = openCatalog(codec, store, seed, o);
            Path effective =
                    base != null || (o.catalog() != null && !o.readOnly()) ? o.catalog() : null;
            BlockTree<S> tree = new BlockTree<>(codec, store, o.blockBudgetBytes(), seed, base);
            return new PromptCache<>(model, tree, codec.coarseBlocks(), store, effective, o);
        } catch (RuntimeException | Error e) {
            store.close();
            throw e;
        }
    }

    /**
     * Opens the catalog, creating an EMPTY artifact first when a read-write file is missing - the
     * cache knows its file from birth, so {@link #save} is always an append against a mounted base.
     * Read-write problems fail the boot loudly (silently recreating would destroy the old catalog).
     * A MISSING read-only file degrades to serving without it; an EXISTING but incompatible file
     * fails loudly in BOTH modes - the caller pointed at a real artifact, and silently ignoring it
     * is worse than refusing to boot.
     */
    private static <S extends RuntimeState> FrozenBlocks openCatalog(
            StateCodec<S> codec, CacheStore store, byte[] seed, Options o) {
        if (o.catalog() == null) return null;
        try {
            if (!Files.exists(o.catalog())) {
                if (o.readOnly()) {
                    System.err.println(
                            "jinfer: read-only cache missing ("
                                    + o.catalog()
                                    + "): serving without it");
                    return null;
                }
                new BlockTree<>(codec, store, 0, seed).freeze(o.catalog());
            }
            return FrozenBlocks.open(o.catalog(), seed);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to open cache " + o.catalog(), e);
        }
    }

    /** Which source served a prompt; the difference worth tuning {@code jinfer.sessions} on. */
    public enum Tier {
        /** A hot conversation the prompt strictly extends: zero restore, only the delta. */
        SESSION,
        /** The block layer's longest cached prefix, restored into a state. */
        BLOCKS,
        /** Nothing matched: a recycled-or-fresh state prefilled the whole prompt. */
        FRESH
    }

    /** What a pass may do and know - nothing else escapes the cache. */
    public interface Serving {
        /**
         * Wire as the generator's {@code afterIngest}: records each decode token the moment the
         * state's frontier includes it (with blocks on, as its own per-position block - the tail
         * contract). A no-op lane when blocks are off, but ALWAYS keeps the hot stream in lockstep,
         * so wiring it is not optional.
         */
        void tail(int token);

        /** Positions served from cache instead of prefill. */
        int restored();

        Tier tier();
    }

    /**
     * The generation body: receives the prepared state (prompt already ingested) and the pass
     * context. Runs on the cache's single thread.
     */
    public interface Pass<S extends RuntimeState, R> {
        R run(S state, Serving serving) throws RuntimeException;
    }

    /**
     * THE serving protocol - every model kind, one path, cheapest source first: a hot session the
     * prompt strictly extends ({@link Tier#SESSION}), else the longest block prefix restored into a
     * recycled-or-fresh state ({@link Tier#BLOCKS}), else full prefill ({@link Tier#FRESH}). The
     * prompt is ingested (one block per batch, skip-restored) BEFORE the pass runs; the pass only
     * generates. On success the finished state returns to the hot layer; if the pass throws, the
     * session is discarded - a possibly-torn state must never serve again.
     */
    public <R> R serve(List<Batch> prompt, Pass<S, R> pass) {
        long[] fingerprints = CachedSession.fingerprints(prompt);
        CachedSession<S> session = hotAcquire(fingerprints);
        boolean tier1 = session != null;
        if (tier1) {
            hotHits++;
        } else {
            session = attach(fingerprints);
        }
        int restored = session.position();
        Tier tier = tier1 ? Tier.SESSION : restored > 0 ? Tier.BLOCKS : Tier.FRESH;
        // one group per batch: the codec's turn boundaries ARE the block boundaries, so a
        // follow-up diverging after turn k still reuses turns 0..k-1
        List<List<Batch>> groups = new ArrayList<>(prompt.size());
        for (Batch b : prompt) groups.add(List.of(b));
        CachedSession<S> bound = session;
        Serving serving =
                new Serving() {
                    @Override
                    public void tail(int token) {
                        bound.adopt(token);
                    }

                    @Override
                    public int restored() {
                        return restored;
                    }

                    @Override
                    public Tier tier() {
                        return tier;
                    }
                };
        R result;
        try {
            session.ingestGroups(groups);
            result = pass.run(session.state(), serving);
        } catch (RuntimeException | Error e) {
            closeState(session); // torn: never serves again; its committed blocks survive
            throw e;
        }
        release(session);
        return result;
    }

    /** A state for a miss, recycled where possible, attached to the right block mode. */
    private CachedSession<S> attach(long[] fingerprints) {
        int cap = fingerprints.length - 1; // THE LAW: resume stops one short
        S state = recycled(fingerprints.length);
        if (state == null) state = freshState();
        try {
            if (tree == null) return CachedSession.hot(model, state);
            return coarse
                    ? CachedSession.resumeReadOnly(model, tree, state, fingerprints, cap)
                    : CachedSession.resume(model, tree, state, fingerprints, cap);
        } catch (RuntimeException | Error e) {
            if (state instanceof com.qxotic.jinfer.BaseState base) base.close();
            throw e;
        }
    }

    /**
     * Pins a prefix into the block layer: interior batches as turn blocks, the FINAL position as
     * its own single (so a later serve of exactly this prompt - capped one short by the law -
     * restores everything but that last token); a coarse codec commits ONE block over everything
     * but the trailing batch (request-shaped scaffold: a block containing it would never match, yet
     * would still pay the residue). Dedups against what is already cached. Throws when the budget
     * refused it - a define exists only to cache, and returning quietly would let every later serve
     * re-prefill with no diagnostic.
     */
    public void define(List<Batch> prompt) {
        BlockTree<S> t = requireTree();
        int total = positions(prompt);
        if (total == 0) return;
        S state =
                coarse
                        ? model.newState(model.config().contextLength(), Math.max(total, 16))
                        : model.newState(
                                model.config().contextLength(),
                                Math.min(Math.max(total, 16), RuntimeFlags.BATCH_CAPACITY));
        statesAllocated++;
        try {
            CachedSession<S> s = CachedSession.resume(model, t, state, prompt);
            s.ingestGroups(defineGroups(prompt));
            if (s.detached()) {
                throw new IllegalStateException(
                        "cached prompt not fully retained: the cache budget refused it"
                                + " (-Djinfer.promptCacheMB, currently "
                                + (t.sample().budgetBytes() >> 20)
                                + " MB)");
            }
        } finally {
            if (state instanceof com.qxotic.jinfer.BaseState base) base.close();
        }
    }

    private List<List<Batch>> defineGroups(List<Batch> prompt) {
        int n = prompt.size();
        if (coarse) return List.of(prompt.subList(0, Math.max(1, n - 1)));
        List<List<Batch>> groups = new ArrayList<>(n + 1);
        for (int i = 0; i < n - 1; i++) groups.add(List.of(prompt.get(i)));
        Batch last = prompt.get(n - 1);
        if (last.input() instanceof Batch.Input.Tokens t && t.ids().length > 1) {
            int[] ids = t.ids();
            groups.add(List.of(Batch.prefill(java.util.Arrays.copyOf(ids, ids.length - 1))));
            groups.add(List.of(Batch.prefill(new int[] {ids[ids.length - 1]})));
        } else {
            groups.add(List.of(last));
        }
        return groups;
    }

    /** Appends the block layer's fresh blocks to the catalog; a no-op without one / read-only. */
    public void save() {
        if (tree == null || catalog == null || readOnly) return;
        try {
            tree.appendTo(catalog);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to save cache to " + catalog, e);
        }
    }

    /**
     * A NEW artifact from the whole block layer (mounted base + growth); refuses its own catalog -
     * {@link #save} is the write-back for that.
     */
    public void export(Path out) {
        try {
            requireTree().freeze(out);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to export cache to " + out, e);
        }
    }

    /** Whether the block layer exists (codec present and not disabled). */
    public boolean blockCaching() {
        return tree != null;
    }

    /** The whole cache in one reading - hot and blocks; block fields are zero when hot-only. */
    public record Sample(
            int hotSessions,
            long hotHits,
            long statesAllocated,
            int blocks,
            long bytes,
            long budgetBytes,
            long hits,
            long misses,
            long evictions,
            long refusals) {}

    public Sample sample() {
        BlockTree.Sample t = tree == null ? null : tree.sample();
        return new Sample(
                hot.size(),
                hotHits,
                statesAllocated,
                t == null ? 0 : t.blocks(),
                t == null ? 0 : t.bytes(),
                t == null ? 0 : t.budgetBytes(),
                t == null ? 0 : t.hits(),
                t == null ? 0 : t.misses(),
                t == null ? 0 : t.evictions(),
                t == null ? 0 : t.refusals());
    }

    /** Test seam: the block layer's stats line (see {@link BlockTree#stats}). */
    public String treeStats() {
        return requireTree().stats();
    }

    /** Frees every hot state, the spare, and the block blobs - deterministic, not GC-eventual. */
    @Override
    public void close() {
        while (!hot.isEmpty()) closeState(hot.removeFirst());
        if (spare instanceof com.qxotic.jinfer.BaseState base) base.close();
        spare = null;
        store.close();
    }

    // ---- hot-layer internals ----------------------------------------------------------------

    /**
     * The hot session with the LONGEST stream strictly prefixing the prompt, removed while in use;
     * null = no hot match.
     */
    private CachedSession<S> hotAcquire(long[] fingerprints) {
        CachedSession<S> best = null;
        for (CachedSession<S> s : hot) {
            if (s.streamIsStrictPrefixOf(fingerprints, fingerprints.length)
                    && fingerprints.length <= s.state().contextCapacity()
                    && (best == null || s.length() > best.length())) {
                best = s;
            }
        }
        if (best != null) hot.remove(best);
        return best;
    }

    /**
     * Returns a finished session to the hot layer as most-recent; past capacity the least-recent is
     * dropped. With hotSessions=0 the state is WIPED here, the moment the reply is done, and
     * retained as the bare spare: nothing of the conversation lingers between requests.
     */
    private void release(CachedSession<S> session) {
        if (session.length() != session.state().position()) {
            // stream and state disagree: pooling it would match a future prompt against DIFFERENT
            // content - silent poisoning. A caller bug (tail() not wired); free it, keep serving.
            System.err.println(
                    "jinfer: discarding desynced session (stream "
                            + session.length()
                            + " != state "
                            + session.state().position()
                            + ")");
            closeState(session);
            return;
        }
        if (hotCapacity == 0) {
            S state = session.state();
            if (spare == null) {
                if (state.position() != 0) state.reset();
                spare = state;
            } else {
                closeState(session);
            }
            return;
        }
        hot.addLast(session);
        while (hot.size() > hotCapacity) closeState(hot.removeFirst());
    }

    /**
     * A recycled allocation for a miss: the LRU hot state once at capacity (only a hot layer with
     * room left allocates), or the retained spare. Null = allocate.
     */
    private S recycled(int len) {
        if (hotCapacity > 0 && hot.size() >= hotCapacity) {
            CachedSession<S> oldest = hot.peekFirst();
            if (oldest != null && oldest.state().contextCapacity() >= len) {
                hot.removeFirst();
                S state = oldest.state();
                if (state.position() != 0) state.reset();
                return state;
            }
        }
        if (spare != null && spare.contextCapacity() >= len) {
            S state = spare;
            spare = null;
            return state;
        }
        return null;
    }

    /** Full batch capacity, not prompt-sized: this allocation serves every later request too. */
    private S freshState() {
        statesAllocated++;
        return model.newState(model.config().contextLength(), RuntimeFlags.BATCH_CAPACITY);
    }

    private void closeState(CachedSession<?> session) {
        if (session.state() instanceof com.qxotic.jinfer.BaseState base) base.close();
    }

    private static int positions(List<Batch> prompt) {
        int total = 0;
        for (Batch b : prompt) total += b.count();
        return total;
    }

    private BlockTree<S> requireTree() {
        if (tree == null) {
            throw new IllegalStateException(
                    model.stateCodec().isPresent()
                            ? "block caching is disabled (budget 0 / -Djinfer.promptCache=false)"
                            : model.getClass().getSimpleName()
                                    + " does not support block caching (no state codec)");
        }
        return tree;
    }

    /** A fast, stable model identity for the block key chain - see {@link BlockTree#modelSeed}. */
    public static byte[] modelSeed(Path gguf) {
        return BlockTree.modelSeed(gguf);
    }

    /** As {@link #modelSeed(Path)} over an open channel. */
    public static byte[] modelSeed(java.nio.channels.FileChannel ch) {
        return BlockTree.modelSeed(ch);
    }
}
