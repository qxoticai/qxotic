package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.telemetry.PerformanceCliff;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BooleanSupplier;

/**
 * jinfer's KV cache - the one front door to every way caching happens. Two layers, one law:
 *
 * <ul>
 *   <li>SESSIONS - the last N conversations stay live as ready-to-continue states; a prompt that
 *       strictly extends one continues in place (ANY model, codec or not).
 *   <li>BLOCKS - everything computed is kept as content-keyed KV blocks (RAM, budget-bounded,
 *       optionally backed by a catalog file that survives restarts). Interior content commits at
 *       batch boundaries (turns, when the codec encodes one batch per turn); the decode tail
 *       commits per position on residue-free codecs (a truncated or edited echo resumes
 *       token-exact, at no extra byte cost) and as one block per reply on residue-carrying codecs
 *       (every checkpoint is a turn boundary - duplicating the residue per token measured 300x the
 *       row bytes on LFM2.5).
 * </ul>
 *
 * <p>THE LAW, stated once: a resume always stops one position short, so the final token re-ingests
 * and the logits are always fresh - and a cached answer is byte-identical to a cold one.
 *
 * <p>{@link #of} reads the model's capabilities itself: no {@link CheckpointCodec} = sessions-only;
 * fixed checkpoint overhead above the configured limit = blocks written by {@link #define} alone;
 * otherwise the full picture.
 *
 * <p>Single-threaded by design: every method (including {@link #sample}) belongs to the one
 * generation thread, like the tree it fronts. The low-level layer ({@link CachedSession}, {@link
 * BlockTree}, {@link FrozenBlocks}) stays public for the testkit, benches and speculative decoding;
 * production callers should not need it.
 */
public final class PromptCache<S extends ContextState> implements AutoCloseable {

    private static final System.Logger LOG = System.getLogger("jinfer.cache");

    /** Immutable cache configuration; start from {@link #DEFAULTS} and name every change. */
    public static final class Options {

        private static final long DEFAULT_MAX_CHECKPOINT_OVERHEAD = 1L << 20;

        /**
         * The context capacity a caller with no opinion gets, bounded so a state allocates blind on
         * any machine (llama.cpp's default too), and never more than the model's own length.
         */
        public static final int DEFAULT_CONTEXT_CAPACITY = 4096;

        /** contextCapacity not chosen by the caller: resolved to min(default, model) at build. */
        private static final int AUTO = -1;

        /**
         * What a caller with no opinion gets: the bounded default context ({@link
         * #DEFAULT_CONTEXT_CAPACITY} or the model's length when that is smaller), 4 live
         * conversations, a 2 GB RAM-only block layer and a 1 MiB incremental-checkpoint overhead
         * limit. Turning the block layer off is {@code withBlockBudget(0)}.
         */
        public static final Options DEFAULTS =
                new Options(4, AUTO, 2048L << 20, null, false, DEFAULT_MAX_CHECKPOINT_OVERHEAD);

        private final int retainedSessions;
        private final int contextCapacity;
        private final long blockBudgetBytes;
        private final Path catalog;
        private final boolean readOnly;
        private final long maxCheckpointOverheadBytes;

        private Options(
                int retainedSessions,
                int contextCapacity,
                long blockBudgetBytes,
                Path catalog,
                boolean readOnly,
                long maxCheckpointOverheadBytes) {
            if (retainedSessions < 0)
                throw new IllegalArgumentException("retainedSessions " + retainedSessions);
            if (contextCapacity < 0 && contextCapacity != AUTO)
                throw new IllegalArgumentException(
                        "contextCapacity must be >= 0 (0 uses the model context length): "
                                + contextCapacity);
            if (blockBudgetBytes < 0)
                throw new IllegalArgumentException("blockBudgetBytes " + blockBudgetBytes);
            if (maxCheckpointOverheadBytes < 0)
                throw new IllegalArgumentException(
                        "maxCheckpointOverheadBytes " + maxCheckpointOverheadBytes);
            this.retainedSessions = retainedSessions;
            this.contextCapacity = contextCapacity;
            this.blockBudgetBytes = blockBudgetBytes;
            this.catalog = catalog;
            this.readOnly = readOnly;
            this.maxCheckpointOverheadBytes = maxCheckpointOverheadBytes;
        }

        /** These options over {@code catalog}, read-only or accumulating. */
        public Options withCatalog(Path catalog, boolean readOnly) {
            return new Options(
                    retainedSessions,
                    contextCapacity,
                    blockBudgetBytes,
                    catalog,
                    readOnly,
                    maxCheckpointOverheadBytes);
        }

        /** These options with a different block-layer budget; 0 disables the block layer. */
        public Options withBlockBudget(long blockBudgetBytes) {
            return new Options(
                    retainedSessions,
                    contextCapacity,
                    blockBudgetBytes,
                    catalog,
                    readOnly,
                    maxCheckpointOverheadBytes);
        }

        /** These options with a different number of resident conversations. */
        public Options withRetainedSessions(int retainedSessions) {
            return new Options(
                    retainedSessions,
                    contextCapacity,
                    blockBudgetBytes,
                    catalog,
                    readOnly,
                    maxCheckpointOverheadBytes);
        }

        /**
         * Returns these options with a different context-capacity upper bound. {@code 0} uses the
         * model's declared context length; otherwise the effective capacity is the smaller of this
         * value and that length.
         *
         * @throws IllegalArgumentException if {@code contextCapacity < 0}
         */
        public Options withContextCapacity(int contextCapacity) {
            if (contextCapacity < 0)
                throw new IllegalArgumentException(
                        "contextCapacity must be >= 0 (0 uses the model context length): "
                                + contextCapacity);
            return new Options(
                    retainedSessions,
                    contextCapacity,
                    blockBudgetBytes,
                    catalog,
                    readOnly,
                    maxCheckpointOverheadBytes);
        }

        /** These options with a different fixed checkpoint-overhead limit. */
        public Options withMaxCheckpointOverhead(long bytes) {
            return new Options(
                    retainedSessions, contextCapacity, blockBudgetBytes, catalog, readOnly, bytes);
        }
    }

    private final LanguageModel<?, ?, S> model;
    private final BlockTree<S> tree; // null = sessions-only (no codec, or blocks disabled)
    private final CheckpointCodec<S> codec; // null = sessions-only models
    private final boolean defineOnlyBlocks; // ordinary serving restores but does not grow the tree
    // The decode tail's granularity. Per-token singles keep EVERY reply position resumable, and
    // on a residue-free codec (dense rows, ring rows) they are free - a single stores just its
    // row, which any granularity stores anyway. A codec with a residue duplicates it into every
    // block, so per-token singles multiply it by the reply length. There the tail commits as ONE
    // block per reply instead: every checkpoint is a turn boundary, one residue per reply, and
    // an edited or stop-cut echo re-prefills at most one reply's tail.
    // MEASURED (LFM2.5-8B server, same mixed chat/tool battery, ~300KB conv residue): per-token
    // tail = 6.5k generated tokens -> 6,819 blocks / 2,107MB (~324KB/token, budget exhausted);
    // per-reply tail = 8.8k generated tokens -> 95 blocks / 111MB (~12.5KB/token = the pure row
    // rate - the duplication is fully gone). Hit pattern identical (22/8/2 vs 24/8/2).
    private final boolean tailPerToken;
    private final CacheStore store; // owned; freed at close
    private final Path writeBack; // save()'s append target; null = read-only or no catalog

    // ---- retained sessions: THE POOL IS THE ALLOCATOR --------------------------------------
    // A full context is the dominant per-pipeline allocation. At capacity the least-recent
    // session's state is recycled (reset() - every family has decided what a fresh sequence
    // must clear). With retainedSessions=0 no state survives its request.
    private final int retainedCapacity;
    // every state this cache allocates is this many positions; a prompt past it is refused
    private final int contextCapacity;
    private final ArrayDeque<CachedSession<S>> retained = new ArrayDeque<>();
    private boolean closed;
    private long sessionHits;
    private long statesAllocated;

    private PromptCache(
            LanguageModel<?, ?, S> model,
            CheckpointCodec<S> codec,
            BlockTree<S> tree,
            boolean defineOnlyBlocks,
            boolean tailPerToken,
            CacheStore store,
            Path writeBack,
            Options options) {
        this.model = model;
        this.codec = codec;
        this.tree = tree;
        this.defineOnlyBlocks = defineOnlyBlocks;
        this.tailPerToken = tailPerToken;
        this.store = store;
        this.writeBack = writeBack;
        this.retainedCapacity = options.retainedSessions;
        this.contextCapacity = options.contextCapacity;
    }

    /**
     * Builds the cache the {@code model} can support - the only constructor. {@code seed} is the
     * model's cache identity (a sha256 {@link ContentKey}, produced model-load-side): block keys
     * and catalog files are rooted in it, so two models can never match each other's blocks.
     */
    public static <S extends ContextState> PromptCache<S> of(
            LanguageModel<?, ?, S> model, ContentKey seed, Options o) {
        if (seed == null) throw new IllegalArgumentException("null seed");
        int modelCapacity = model.configuration().contextLength();
        int requested = o.contextCapacity;
        int capacity;
        if (requested == Options.AUTO) {
            capacity = Math.min(Options.DEFAULT_CONTEXT_CAPACITY, modelCapacity);
        } else if (requested == 0) {
            capacity = modelCapacity;
        } else if (requested > modelCapacity) {
            // an explicit capacity is a promise about what will fit: a state the model cannot
            // honor is refused here, at construction, not at the first long prompt
            throw new IllegalArgumentException(
                    "contextCapacity "
                            + requested
                            + " exceeds the model's context length "
                            + modelCapacity
                            + "; pass 0 for the model's maximum");
        } else {
            capacity = requested;
        }
        o = o.withContextCapacity(capacity);
        CheckpointCodec<S> codec = model.checkpointCodec().orElse(null);
        if (codec == null) PerformanceCliff.CACHE_SESSIONS_ONLY.report();
        if (codec == null && o.catalog != null) {
            // never silently ignore a file the caller pointed at: without a codec nothing can
            // ever be written to it, so the configured cache would degrade into an unnoticed
            // cold start - the exact failure openCatalog refuses for a missing artifact
            throw new IllegalArgumentException(
                    "catalog "
                            + o.catalog
                            + " configured but "
                            + model.getClass().getSimpleName()
                            + " has no state codec - drop the catalog or load a model that can"
                            + " freeze state");
        }
        boolean wantBlocks = codec != null && (o.blockBudgetBytes > 0 || o.catalog != null);
        if (!wantBlocks) {
            return new PromptCache<>(
                    model, codec, null, false, false, CacheStore.inMemory(), null, o);
        }
        CacheStore store = CacheStore.inMemory();
        try {
            FrozenBlocks base = openCatalog(seed, o);
            BlockTree<S> tree = new BlockTree<>(codec, store, o.blockBudgetBytes, seed, base);
            Path writeBack = o.readOnly ? null : o.catalog;
            long checkpointOverhead = codec.byteSize(0);
            boolean defineOnlyBlocks = checkpointOverhead > o.maxCheckpointOverheadBytes;
            return new PromptCache<>(
                    model,
                    codec,
                    tree,
                    defineOnlyBlocks,
                    checkpointOverhead == 0,
                    store,
                    writeBack,
                    o);
        } catch (RuntimeException | Error e) {
            store.close();
            throw e;
        }
    }

    /**
     * Opens the catalog, creating an EMPTY artifact first when a read-write file is missing - the
     * cache knows its file from birth, so {@link #save} is always an append against a mounted base.
     * Read-write problems fail the boot loudly (silently recreating would destroy the old catalog).
     * A missing or incompatible read-only artifact also fails: an explicitly configured cache must
     * never degrade into an unnoticed cold start.
     */
    private static FrozenBlocks openCatalog(ContentKey seed, Options o) {
        if (o.catalog == null) return null;
        try {
            if (!Files.exists(o.catalog)) {
                if (o.readOnly) {
                    throw new IllegalArgumentException("prompt cache does not exist: " + o.catalog);
                }
                FrozenBlocks.createEmpty(o.catalog, seed);
            }
            return FrozenBlocks.open(o.catalog, seed);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to open cache " + o.catalog, e);
        }
    }

    /** Which source served a prompt; the difference worth tuning retained sessions on. */
    public enum Tier {
        /** A retained conversation the prompt strictly extends: zero restore, only the delta. */
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
         * state's frontier includes it. With blocks on, a residue-free codec commits it as its own
         * per-position block (token-exact echo resume, free - a single stores just its row); a
         * residue-carrying codec buffers the reply and commits it as ONE block when the pass ends
         * (one residue per reply, checkpoints at turn boundaries). A no-op lane when blocks are
         * off, but ALWAYS keeps the retained stream in lockstep, so wiring it is not optional.
         */
        void tail(int token);

        /**
         * Bulk adoption of tokens a pass ingested directly on the state - the SPECULATIVE shape,
         * where verified tokens land in a batch and control returns once: pass exactly what the
         * state now holds beyond the prompt. One block at the frontier for a per-token codec; joins
         * the one-block reply commit for a residue codec. Same lifetime rule as {@link #tail}:
         * valid only until the pass returns.
         */
        void adopt(int[] tokens);

        /** Positions served from cache instead of prefill. */
        int restored();

        Tier tier();

        /** Time spent selecting/restoring a state and ingesting the uncached prompt suffix. */
        Duration promptTime();

        /**
         * False when the pass's interrupt stopped prompt ingestion at the last completed chunk: the
         * prompt is PARTIALLY served. The pass still runs - it produces its stopped outcome and
         * must not generate from the state - and the session is discarded after the pass returns: a
         * partially-served prompt never serves again. Committed blocks below the seam survive.
         */
        default boolean prefillComplete() {
            return true;
        }
    }

    /**
     * The generation body: receives the prepared state (prompt already ingested) and the pass
     * context. Runs on the cache's single thread.
     */
    public interface Pass<S extends ContextState, R> {
        R run(S state, Serving serving);
    }

    /**
     * THE serving protocol - every model kind, one path, cheapest source first: a retained session
     * the prompt strictly extends ({@link Tier#SESSION}), else the longest block prefix restored
     * into a recycled-or-fresh state ({@link Tier#BLOCKS}), else full prefill ({@link Tier#FRESH}).
     * The prompt is ingested (one block per batch, skip-restored) BEFORE the pass runs; the pass
     * only generates. On success the finished state returns to the retained layer; if the pass or
     * reply commit throws, the session is discarded - a possibly-torn state must never serve again.
     */
    public <R> R serve(List<Batch> prompt, Pass<S, R> pass) {
        return serve(prompt, pass, () -> false);
    }

    /**
     * As {@link #serve(List, Pass)} with a cooperative interrupt consulted BETWEEN prefill chunks
     * (never inside a forward - a chunk in flight always completes). When it first reports true,
     * ingestion stops at the last completed chunk and the pass runs with {@link
     * Serving#prefillComplete()} false; a completed prefill - including one fully restored from
     * cache, where the supplier is never consulted - retains the session as usual.
     */
    public <R> R serve(List<Batch> prompt, Pass<S, R> pass, BooleanSupplier prefillInterrupt) {
        checkOpen();
        long promptStarted = System.nanoTime();
        long[] fingerprints = CachedSession.fingerprints(prompt);
        if (fingerprints.length == 0) {
            throw new IllegalArgumentException("empty prompt: nothing to serve");
        }
        checkFitsContext(fingerprints.length);
        CachedSession<S> sessionHit = sessionAcquire(fingerprints);
        // COARSE REWIND: a thinking model whose template strips reasoning from echoed history
        // never strictly extends the generated stream, so the tail is unreachable by the plain
        // session match - but the session's TAIL SNAPSHOT (residue at the last prompt boundary) can
        // rewind the state to exactly where the echo diverges, and only the stripped reply +
        // new turn re-ingest. Fine codecs never need this: their block layer serves the echo.
        if (sessionHit == null && defineOnlyBlocks) sessionHit = snapshotAcquire(fingerprints);
        if (sessionHit != null) sessionHits++;
        CachedSession<S> session = sessionHit != null ? sessionHit : attach(fingerprints);
        int restored = session.position();
        Tier tier = sessionHit != null ? Tier.SESSION : restored > 0 ? Tier.BLOCKS : Tier.FRESH;
        // one group per batch: the codec's turn boundaries ARE the block boundaries, so a
        // follow-up diverging after turn k still reuses turns 0..k-1
        List<List<Batch>> groups = new ArrayList<>(prompt.size());
        for (Batch b : prompt) groups.add(List.of(b));
        Live serving = new Live(session, restored, tier);
        R result;
        try {
            boolean complete = true;
            if (defineOnlyBlocks && groups.size() > 1) {
                // the snapshot must sit BEFORE the final batch - the generation prompt, by the
                // chat-template convention that it is the prompt's last batch. The echoed next
                // turn shares everything before that seam and nothing after it (the echo renders
                // the reply's turn with history framing, the generation prompt with live framing:
                // <think>\n vs the truncated pair), so a later snapshot can never match.
                complete =
                        session.ingestGroups(
                                groups.subList(0, groups.size() - 1),
                                fingerprints,
                                prefillInterrupt);
                if (complete) session.snapshotTail(codec); // never snapshot a partial prompt
            }
            if (complete) complete = session.ingestGroups(groups, fingerprints, prefillInterrupt);
            serving.promptTime = Duration.ofNanos(System.nanoTime() - promptStarted);
            serving.prefillComplete = complete;
            result = pass.run(session.state(), serving);
            if (complete) serving.flushReply(); // one block, saved at the state's frontier
        } catch (RuntimeException | Error e) {
            closeSession(session); // torn: never serves again; its committed blocks survive
            throw e;
        } finally {
            serving.live = false;
        }
        if (serving.prefillComplete) {
            release(session);
        } else {
            closeSession(session); // a partially-served prompt never serves again
        }
        return result;
    }

    /** The pass's handle, expired the moment the pass returns (success or throw). */
    private final class Live implements Serving {
        private final CachedSession<S> session;
        private final int restored;
        private final Tier tier;
        private final List<Integer> reply = new ArrayList<>(); // buffered per-reply tail tokens
        private boolean live = true;
        private Duration promptTime;
        private boolean prefillComplete = true;

        Live(CachedSession<S> session, int restored, Tier tier) {
            this.session = session;
            this.restored = restored;
            this.tier = tier;
        }

        @Override
        public void tail(int token) {
            // a stashed handle used after the pass returned would poison the retained stream
            // (silently, in sessions-only mode) - fail loudly instead
            if (!live) {
                throw new IllegalStateException(
                        "the serving is over: tail() is valid only until the pass returns");
            }
            if (tailPerToken) {
                session.adopt(token);
            } else {
                reply.add(token); // committed as one block when the pass ends
            }
        }

        @Override
        public void adopt(int[] tokens) {
            if (!live) {
                throw new IllegalStateException(
                        "the serving is over: adopt() is valid only until the pass returns");
            }
            if (tailPerToken) {
                session.adopt(tokens); // one block at the frontier - the speculative shape
            } else {
                for (int token : tokens) reply.add(token); // joins the one-block reply commit
            }
        }

        /** Commits a buffered reply as ONE block - the residue-codec tail granularity. */
        void flushReply() {
            if (reply.isEmpty()) return;
            int[] tokens = new int[reply.size()];
            for (int i = 0; i < tokens.length; i++) tokens[i] = reply.get(i);
            session.adopt(tokens);
        }

        @Override
        public int restored() {
            return restored;
        }

        @Override
        public Tier tier() {
            return tier;
        }

        @Override
        public Duration promptTime() {
            if (promptTime == null) {
                throw new IllegalStateException("prompt has not been served yet");
            }
            return promptTime;
        }

        @Override
        public boolean prefillComplete() {
            return prefillComplete;
        }
    }

    /** A state for a miss, recycled where possible, attached to the right block mode. */
    private CachedSession<S> attach(long[] fingerprints) {
        int cap = fingerprints.length - 1; // THE LAW: resume stops one short
        S state = recycled(fingerprints.length);
        if (state == null) state = freshState();
        try {
            if (tree == null) return CachedSession.fresh(model, state);
            // Large fixed overhead makes ordinary block writes wasteful; define() still writes.
            return CachedSession.resume(model, tree, state, fingerprints, cap, !defineOnlyBlocks);
        } catch (RuntimeException | Error e) {
            closeState(state);
            throw e;
        }
    }

    /**
     * Pins a prefix into the block layer: interior batches as turn blocks, the FINAL position as
     * its own single (so a later serve of exactly this prompt - capped one short by the law -
     * restores everything but that last token; a single-token or media-final batch commits whole
     * instead, and cannot full-hit by construction); define-only mode commits one prefix block over
     * everything but the trailing batch, or, for a single-batch prompt, everything but the trailing
     * position. Dedups against what is already cached. Throws when the budget refused it: a define
     * exists only to cache, and returning quietly would let every later serve re-prefill with no
     * diagnostic.
     */
    public void define(List<Batch> prompt) {
        checkOpen();
        BlockTree<S> t = requireTree();
        long[] fingerprints = CachedSession.fingerprints(prompt);
        int total = fingerprints.length;
        if (total == 0) return;
        // the same guard serve() applies, BEFORE any state is sized or block committed: an
        // over-long define would append junk blocks no serve could ever match
        checkFitsContext(total);
        // Define-only mode commits everything-but-the-last-batch as one chunk, so its state must
        // hold the whole prompt per batch; fine codecs take the boundary's default batch width
        // (the old stateFor clamp's upper bound IS that default - identical chunking)
        S state =
                defineOnlyBlocks
                        ? model.newState(contextCapacity, Math.max(total, 16))
                        : model.newState(contextCapacity);
        try {
            // capped ONE SHORT like every resume: an uncapped resume would dedup into a chunk
            // boundary earlier traffic committed and silently skip the split-last single that
            // makes define-then-serve a full hit
            CachedSession<S> s =
                    CachedSession.resume(model, t, state, fingerprints, total - 1, true);
            s.ingestGroups(defineGroups(prompt), fingerprints);
            if (s.detached()) {
                throw new IllegalStateException(
                        "cached prompt not fully retained: the cache budget refused it"
                                + " (block budget currently "
                                + (t.sample().budgetBytes() >> 20)
                                + " MB)");
            }
        } finally {
            closeState(state);
        }
    }

    private List<List<Batch>> defineGroups(List<Batch> prompt) {
        int n = prompt.size();
        if (defineOnlyBlocks) {
            if (n > 1) return List.of(prompt.subList(0, n - 1));
            // a lone batch committed whole would be a block a one-short serve can never match:
            // one dead residue. Commit all but the trailing position instead (prefix-only);
            // a media batch cannot be sliced and still commits whole.
            if (prompt.get(0).input() instanceof Batch.Input.Tokens t && t.ids().length > 1) {
                return List.of(List.of(Batch.prefill(Arrays.copyOf(t.ids(), t.ids().length - 1))));
            }
            return List.of(prompt);
        }
        List<List<Batch>> groups = new ArrayList<>(n + 1);
        for (int i = 0; i < n - 1; i++) groups.add(List.of(prompt.get(i)));
        Batch last = prompt.get(n - 1);
        if (last.input() instanceof Batch.Input.Tokens t && t.ids().length > 1) {
            int[] ids = t.ids();
            groups.add(List.of(Batch.prefill(Arrays.copyOf(ids, ids.length - 1))));
            groups.add(List.of(Batch.prefill(new int[] {ids[ids.length - 1]})));
        } else {
            groups.add(List.of(last));
        }
        return groups;
    }

    /** Appends the block layer's fresh blocks to the catalog; a no-op without one / read-only. */
    public void save() {
        checkOpen();
        if (tree == null || writeBack == null) return;
        try {
            tree.appendTo(writeBack);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to save cache to " + writeBack, e);
        }
    }

    /**
     * A NEW artifact from the whole block layer (mounted base + growth); refuses its own catalog -
     * {@link #save} is the write-back for that.
     */
    public void export(Path out) {
        checkOpen();
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

    /** Maximum positions accepted by one served request. */
    public int contextCapacity() {
        return contextCapacity;
    }

    /** The whole cache in one reading; block fields are zero when only sessions are enabled. */
    public record Sample(
            int retainedSessions,
            int retainedSessionLimit,
            long sessionHits,
            long stateAllocations,
            long sessionSnapshotBytes,
            int blocks,
            long bytes,
            long budgetBytes,
            long blockHits,
            long blockMisses,
            long blockEvictions,
            long blockDiscards,
            long blockRefusals) {}

    public Sample sample() {
        checkOpen();
        BlockTree.Sample t = tree == null ? BlockTree.Sample.ZERO : tree.sample();
        long snapshotBytes = 0;
        for (CachedSession<S> s : retained) snapshotBytes += s.snapshotBytes();
        return new Sample(
                retained.size(),
                retainedCapacity,
                sessionHits,
                statesAllocated,
                snapshotBytes,
                t.blocks(),
                t.bytes(),
                t.budgetBytes(),
                t.hits(),
                t.misses(),
                t.evictions(),
                t.discards(),
                t.refusals());
    }

    /** Test seam: the block layer's stats line (see {@link BlockTree#stats}). */
    public String treeStats() {
        checkOpen();
        return requireTree().stats();
    }

    /** Frees every retained state and block blob - deterministic, not GC-eventual. */
    @Override
    public void close() {
        if (closed) return;
        closed = true;
        while (!retained.isEmpty()) closeSession(retained.removeFirst());
        store.close();
    }

    // ---- retained-session internals ---------------------------------------------------------

    /**
     * The retained session with the LONGEST stream strictly prefixing the prompt, removed while in
     * use; null = no match.
     */
    private CachedSession<S> sessionAcquire(long[] fingerprints) {
        CachedSession<S> best = null;
        for (CachedSession<S> s : retained) {
            if (s.streamIsStrictPrefixOf(fingerprints)
                    && fingerprints.length <= s.state().contextCapacity()
                    && (best == null || s.length() > best.length())) {
                best = s;
            }
        }
        if (best != null) retained.remove(best);
        return best;
    }

    /**
     * The retained session whose TAIL SNAPSHOT stream (the last prompt boundary) strictly prefixes
     * the prompt, rewound to that boundary and removed from the pool, the define-only echo path.
     * Longest snapshot wins; null = no match.
     */
    private CachedSession<S> snapshotAcquire(long[] fingerprints) {
        CachedSession<S> best = null;
        for (CachedSession<S> s : retained) {
            if (s.snapshotIsStrictPrefixOf(fingerprints)
                    && fingerprints.length <= s.state().contextCapacity()
                    && (best == null || s.length() > best.length())) {
                best = s;
            }
        }
        if (best == null) return null;
        retained.remove(best);
        best.rewindToSnapshot(codec);
        return best;
    }

    /**
     * Returns a finished session to the retained layer as most-recent; past capacity the
     * least-recent is dropped. With a zero capacity the state is closed here.
     */
    private void release(CachedSession<S> session) {
        if (session.length() != session.state().position()) {
            // stream and state disagree: pooling it would match a future prompt against DIFFERENT
            // content - silent poisoning. A caller bug (tail() not wired); free it, keep serving.
            LOG.log(
                    System.Logger.Level.WARNING,
                    "discarding desynced session (stream {0} != state {1})",
                    session.length(),
                    session.state().position());
            closeSession(session);
            return;
        }
        if (retainedCapacity == 0) {
            closeSession(session);
            return;
        }
        retained.addLast(session);
        while (retained.size() > retainedCapacity) closeSession(retained.removeFirst());
    }

    /**
     * A recycled allocation for a miss: the least-recent state once the retained layer is full.
     * Null means allocate.
     */
    private S recycled(int len) {
        if (retainedCapacity > 0 && retained.size() >= retainedCapacity) {
            CachedSession<S> oldest = retained.peekFirst();
            if (oldest != null && oldest.state().contextCapacity() >= len) {
                retained.removeFirst();
                oldest.dropSnapshot();
                S state = oldest.state();
                if (state.position() != 0) state.reset();
                return state;
            }
        }
        return null;
    }

    /** Full batch capacity, not prompt-sized: this allocation serves every later request too. */
    private S freshState() {
        statesAllocated++;
        return model.newState(contextCapacity);
    }

    private void closeSession(CachedSession<S> session) {
        session.dropSnapshot();
        closeState(session.state());
    }

    private static void closeState(ContextState state) {
        state.close();
    }

    private void checkOpen() {
        if (closed) throw new IllegalStateException("the cache is closed");
    }

    /** Whether the prompt fits the STATE this cache allocates, which is what it will run in. */
    private void checkFitsContext(int positions) {
        if (positions > contextCapacity) {
            throw new IllegalArgumentException(
                    "Prompt exceeds context capacity ("
                            + positions
                            + " tokens, "
                            + contextCapacity
                            + " available - raise the context capacity)");
        }
    }

    private BlockTree<S> requireTree() {
        if (tree == null) {
            throw new IllegalStateException(
                    codec != null
                            ? "block caching is disabled (block budget 0)"
                            : model.getClass().getSimpleName()
                                    + " does not support block caching (no state codec)");
        }
        return tree;
    }
}
