package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.cache.StateCodec;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.toknroll.IntSequence;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Supplier;

/**
 * The framework-neutral provider runtime shared by the langchain4j and Spring AI integrations: one
 * loaded model, the two-tier template stack (native codec first, hardened Jinja whole-render
 * fallback), the single-stream generation lock (a jinfer model runs one generation at a time;
 * concurrent calls queue), the cached-prompt block tree behind withCachedPrompt/save/load, and the
 * live-session pool behind cachedSessions(n). Integrations adapt only what is genuinely theirs:
 * message/tool mapping into {@link Conversation}s and framework exception types.
 *
 * <p>Everything here speaks jinfer types - no framework classes, no fingerprint/cache internals
 * (the cache package's content addressing stays its own law).
 */
public final class ChatEngine {

    private final LoadedModel<?> loaded;
    private final String modelName;
    private final JinjaChatTemplate jinja;
    private final ReentrantLock lock = new ReentrantLock(true);
    private final CacheStore promptStore; // owned: closed (freed) with the engine
    private final FrozenBlocks mounted; // the read-only artifact, shared with forks (null = none)
    private final PromptCache<?> prompts; // the cached-prompt block tree (null = unsupported)
    // cachedSessions(n): the last n live conversation states, reused append-only when a new
    // prompt's batch stream strictly extends one (all access under the generation lock)
    private final ArrayDeque<LiveSession> sessions = new ArrayDeque<>();
    private final int sessionCapacity;
    // the streaming driver: at most ONE lazy platform thread, reused while streams keep coming,
    // gone after an idle minute. One is enough - generations serialize on the engine lock anyway,
    // and a fresh thread per request would just park extras on that lock
    private final java.util.concurrent.ThreadPoolExecutor streamDriver =
            new java.util.concurrent.ThreadPoolExecutor(
                    0,
                    1,
                    60,
                    java.util.concurrent.TimeUnit.SECONDS,
                    new java.util.concurrent.LinkedBlockingQueue<>(),
                    r -> new Thread(r, "jinfer-stream"));
    private int sessionHits;
    private volatile boolean closed;

    /** A finished generation's state with the batch stream of everything ingested into it. */
    private record LiveSession(RuntimeState state, List<Batch> stream, int positions) {}

    public ChatEngine(
            Path modelPath,
            Path mediaProjector,
            int contextLength,
            Path cachedPrompts,
            int cachedSessions) {
        this.sessionCapacity = Math.max(0, cachedSessions);
        // the integrations' builder contract is "0 = the model's own maximum"; Models.load spells
        // that -1 - without this both integrations crashed in the port's tensor sizing on 0
        int ctx = contextLength <= 0 ? -1 : contextLength;
        try {
            this.loaded =
                    mediaProjector == null
                            ? Models.load(modelPath, ctx)
                            : Models.load(modelPath, mediaProjector, ctx);
            this.promptStore = CacheStore.inMemory();
            // built only when the model can support it (or a mount demands it): a codec-less
            // model must still load and chat - the codec throw belongs to the first
            // CACHED-feature use, not to plain construction
            this.mounted =
                    cachedPrompts == null ? null : FrozenBlocks.open(cachedPrompts, loaded.seed());
            if (mounted != null) {
                this.prompts = tree(loaded, promptStore, mounted);
            } else if (loaded.model().stateCodec().isPresent()) {
                this.prompts = tree(loaded, promptStore, null);
            } else {
                this.prompts = null;
            }
        } catch (IOException e) {
            throw new UncheckedIOException("failed to load " + modelPath, e);
        }
        this.modelName = modelPath.getFileName().toString();
        this.jinja = new JinjaChatTemplate(loaded.tokenizer(), loaded.chatTemplateSource());
    }

    /** The fork constructor: shares the immutable loaded model, owns everything mutable. */
    private ChatEngine(ChatEngine base) {
        this.loaded = base.loaded;
        this.modelName = base.modelName;
        this.jinja = base.jinja; // compiled once per model; stateless at render time
        this.sessionCapacity = base.sessionCapacity;
        this.promptStore = CacheStore.inMemory();
        this.mounted =
                base.mounted; // immutable artifact: safely shared, zero-prefill for forks too
        // a fresh live tree over the shared frozen base; codec-less models stay treeless
        this.prompts = base.prompts == null ? null : tree(loaded, promptStore, mounted);
    }

    /**
     * A sibling engine sharing this one's loaded model - weights, tokenizer and template are
     * immutable and thread-safe to share (all per-run scratch lives in {@link
     * com.qxotic.jinfer.RuntimeState}); the fork owns its own serial pipeline: lock, empty prompt
     * tree, session pool and stream driver. THE cheap answer to "one instance = one pipeline":
     * parallelism without reloading the model.
     */
    public ChatEngine fork() {
        return new ChatEngine(this);
    }

    private static <S extends RuntimeState> PromptCache<S> tree(
            LoadedModel<S> loaded, CacheStore store, FrozenBlocks base) {
        // unbounded: prompts are explicit, few, and deliberately paid for
        return new PromptCache<>(loaded.codec(), store, Long.MAX_VALUE, loaded.seed(), base);
    }

    public LoadedModel<?> loaded() {
        return loaded;
    }

    public String modelName() {
        return modelName;
    }

    /** Idempotent: frees the tree's blobs and the pooled states; later use fails loudly. */
    public void close() {
        lock.lock();
        try {
            closed = true;
            sessions.clear();
            promptStore.close();
        } finally {
            lock.unlock();
        }
        // no interrupt: an in-flight generation finishes; queued streams fail loudly at checkOpen
        streamDriver.shutdown();
    }

    /** Runs a streaming generation on the engine's single lazy driver thread. */
    public void stream(Runnable generation) {
        streamDriver.execute(generation);
    }

    private void checkOpen() {
        if (closed) throw new IllegalStateException("the model is closed");
    }

    /** The encoded prompt plus the reply parser when the model has a native codec. */
    public record Encoded(List<Batch> prompt, Optional<ChatTemplate> template) {}

    /**
     * Native-first encode: the model's own codec when it can frame the conversation byte-exactly,
     * else the scrubbed Jinja whole-render over the caller's lazily-built framework maps (only
     * built when the punt actually happens). Media never reaches the text-only fallback - it fails
     * loudly ({@link UnsupportedOperationException}) instead of being silently dropped;
     * integrations map that to their framework's exception type.
     */
    public Encoded encode(
            Conversation conversation,
            Supplier<List<Object>> messageMaps,
            Supplier<List<Object>> toolMaps) {
        Optional<ChatTemplate> template = loaded.template();
        UnsupportedConversation punted = null;
        if (template.isPresent()) {
            try {
                return new Encoded(template.get().encode(conversation), template);
            } catch (UnsupportedConversation punt) {
                punted = punt; // fall through; the parser (same reply grammar) stays usable
            }
        }
        if (hasMedia(conversation)) {
            throw new UnsupportedOperationException(
                    "image/audio/video input is not supported by this model"
                            + (punted != null ? ": " + punted.getMessage() : "")
                            + " (for Gemma 4, pass the mmproj GGUF via mediaProjector(...))");
        }
        List<Object> tools = toolMaps.get();
        IntSequence ids =
                jinja.render(
                        messageMaps.get(),
                        tools == null || tools.isEmpty() ? null : tools,
                        true,
                        conversation.thinking(),
                        null);
        return new Encoded(List.of(Batch.prefill(ids.toArray())), template);
    }

    private static boolean hasMedia(Conversation conversation) {
        return conversation.messages().stream()
                .flatMap(m -> m.content().stream())
                .anyMatch(p -> p instanceof Part.Blob);
    }

    /**
     * The generation result plus the request's cache accounting: positions served from retained KV
     * instead of prefill (block-tree restore or a pooled live session).
     */
    public record Outcome(Generator.GenerationResult result, int restoredTokens) {}

    /**
     * One generation pass under the engine lock. {@code cached} routes through the prompt tree
     * (resume the longest defined prefix, prefill only the rest) - the uncached path never touches
     * the tree, keeping the base model fully stateless. Either way a pooled live session whose
     * stream strictly prefixes the prompt continues append-only first.
     */
    public Outcome generate(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink,
            boolean cached) {
        lock.lock();
        try {
            checkOpen();
            return run(loaded.model(), prompt, sampler, maxTokens, timeoutNanos, sink, cached);
        } finally {
            lock.unlock();
        }
    }

    /**
     * One generation pass, cheapest source first: a pooled live session when the prompt strictly
     * extends one (zero restore, only the delta prefills), else the prompt tree for cached views
     * (block restore, resume capped one short so the final block re-ingests and leaves fresh
     * logits), else a fresh state. On success the finished state (prompt + reply KV) returns to the
     * pool for the conversation's next turn.
     */
    private <S extends RuntimeState> Outcome run(
            LanguageModel<?, ?, S> model,
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink,
            boolean cached) {
        int total = positions(prompt);
        S state;
        List<Batch> remaining;
        int restored;
        Pooled pooled = acquireSession(prompt, total);
        if (pooled != null) {
            sessionHits++;
            @SuppressWarnings("unchecked")
            S reused = (S) pooled.session().state();
            state = reused;
            restored = pooled.prefixPositions();
            remaining = CachedSession.tail(prompt, restored);
        } else if (cached) {
            state = obtainState(model, total);
            CachedSession<S> resumed =
                    CachedSession.resume(model, tree(), state, prompt, total - 1);
            restored = resumed.position();
            remaining = CachedSession.tail(prompt, restored);
        } else {
            state = obtainState(model, total);
            restored = 0;
            remaining = prompt;
        }
        Generator.GenerationResult result =
                Generator.generate(
                        model,
                        state,
                        Batch.prepare(remaining, state.batchCapacity()),
                        sampler,
                        maxTokens,
                        timeoutNanos,
                        loaded.stopTokens(),
                        sink);
        poolSession(state, prompt, total, result); // not reached on throw: torn states never pool
        return new Outcome(result, restored);
    }

    private static int positions(List<Batch> prompt) {
        int total = 0;
        for (Batch b : prompt) total += b.count();
        return total;
    }

    /**
     * The state for a pool-miss pass: when the pool is FULL, this request's pooling would evict the
     * oldest session anyway - so its allocation is recycled NOW ({@code reset()} + reuse, skipping
     * a full-context allocation) instead of dropped. Families without a {@code reset()} port keep
     * today's fresh-allocation behavior; states are full-context by {@link Generator#stateFor}
     * sizing, so any pooled allocation fits any request.
     */
    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> S obtainState(LanguageModel<?, ?, S> model, int total) {
        if (sessionCapacity > 0 && sessions.size() >= sessionCapacity) {
            LiveSession oldest = sessions.peekFirst();
            if (oldest != null && oldest.state().contextCapacity() >= total) {
                try {
                    oldest.state().reset();
                    sessions.removeFirst();
                    return (S) oldest.state();
                } catch (UnsupportedOperationException noResetPort) {
                    // this family cannot recycle yet: fresh allocation, exactly as before
                }
            }
        }
        return Generator.stateFor(model, total);
    }

    /** A pooled hit: the live session plus how many prompt positions its stream covers. */
    private record Pooled(LiveSession session, int prefixPositions) {}

    /** The pooled session with the LONGEST stream strictly prefixing {@code prompt}, removed. */
    private Pooled acquireSession(List<Batch> prompt, int total) {
        LiveSession best = null;
        int bestLen = -1;
        for (LiveSession s : sessions) {
            if (s.positions() <= bestLen || s.state().contextCapacity() < total) continue;
            int n = CachedSession.strictPrefixPositions(s.stream(), prompt);
            if (n > bestLen) {
                best = s;
                bestLen = n;
            }
        }
        if (best == null) return null;
        sessions.remove(best);
        return new Pooled(best, bestLen);
    }

    /** Pools the finished state under prompt + ingested-reply stream; evicts past cap. */
    private void poolSession(
            RuntimeState state, List<Batch> prompt, int total, Generator.GenerationResult r) {
        if (sessionCapacity == 0) return;
        int ingested = state.position() - total;
        if (ingested < 0) return;
        List<Batch> stream = new ArrayList<>(prompt);
        if (ingested > 0) {
            int[] reply = new int[ingested];
            for (int i = 0; i < ingested; i++) reply[i] = r.tokens().intAt(i);
            stream.add(Batch.prefill(reply));
        }
        sessions.addLast(new LiveSession(state, List.copyOf(stream), total + ingested));
        while (sessions.size() > sessionCapacity) sessions.removeFirst();
    }

    // ---- cached prompts: the block tree behind withCachedPrompt / save / load ----

    /**
     * Encode via the native codec only - cached prompts are a prefix-stability bet the Jinja
     * whole-render cannot honor. Throws {@link UnsupportedOperationException} when the model has no
     * native codec; integrations map it.
     */
    public Encoded encodeNative(Conversation conversation) {
        ChatTemplate template =
                loaded.template()
                        .orElseThrow(
                                () ->
                                        new UnsupportedOperationException(
                                                "cached prompts need a native chat-template codec;"
                                                    + " this model only has the Jinja whole-render"
                                                    + " (no prefix-stability guarantee)"));
        return new Encoded(template.encode(conversation), Optional.of(template));
    }

    /**
     * Defines (prefills) a cached prompt: dedups against the tree, commits one block per encoded
     * batch (turn boundaries) - or ONE block for the whole prompt when the model's codec has a
     * coarse residue ({@link StateCodec#coarseBlocks()}) - then discards the working state: the
     * blocks hold the KV.
     */
    public void define(Conversation prefix) {
        List<Batch> prompt = encodeNative(prefix).prompt();
        lock.lock();
        try {
            checkOpen();
            defineOn(loaded.model(), tree(), prompt);
        } finally {
            lock.unlock();
        }
    }

    private <S extends RuntimeState> void defineOn(
            LanguageModel<?, ?, S> model, PromptCache<S> cache, List<Batch> prompt) {
        // coarse-residue codecs (NemotronH: MBs per block) commit ONE block over everything but
        // the trailing scaffold batch: one chunk (batch capacity = prompt length), one residue
        // per prompt. The scaffold is request-shaped (it re-encodes after the user's turn), so
        // a block containing it would never match - yet would still pay the residue.
        boolean coarse = model.stateCodec().map(StateCodec::coarseBlocks).orElse(false);
        S state =
                coarse
                        ? model.newState(
                                model.config().contextLength(), Math.max(positions(prompt), 16))
                        : Generator.stateFor(model, positions(prompt));
        CachedSession<S> s = CachedSession.resume(model, cache, state, prompt);
        s.ingestGroups(
                coarse
                        ? List.of(prompt.subList(0, Math.max(1, prompt.size() - 1)))
                        : prompt.stream().map(List::of).toList());
    }

    /** Freezes the whole tree (mounted base + everything defined) into one artifact. */
    public void freezePrompts(Path out) {
        lock.lock();
        try {
            checkOpen();
            tree().freeze(out);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to save cached prompts to " + out, e);
        } finally {
            lock.unlock();
        }
    }

    /** Test seam: the tree's stats line ("blocks=.. hits=.." - see PromptCache.stats). */
    public String promptStats() {
        return tree().stats();
    }

    /** Test seam: live-session pool occupancy and hit count. */
    public String sessionStats() {
        lock.lock();
        try {
            return "sessions=" + sessions.size() + " hits=" + sessionHits;
        } finally {
            lock.unlock();
        }
    }

    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> PromptCache<S> tree() {
        if (prompts == null) {
            throw new IllegalStateException(
                    loaded.model().getClass().getSimpleName()
                            + " does not support block caching (no state codec)");
        }
        return (PromptCache<S>) prompts;
    }
}
