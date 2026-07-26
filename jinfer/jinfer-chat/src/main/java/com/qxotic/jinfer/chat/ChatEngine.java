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
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpecialTokens;
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
    private final PromptCache<?> prompts; // the cached-prompt block tree (null = unsupported)
    // cachedSessions(n): the last n live conversation states, reused append-only when a new
    // prompt's batch stream strictly extends one (all access under the generation lock)
    private final ArrayDeque<LiveSession> sessions = new ArrayDeque<>();
    private final int sessionCapacity;
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
            if (cachedPrompts != null) {
                this.prompts =
                        tree(loaded, promptStore, FrozenBlocks.open(cachedPrompts, loaded.seed()));
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
            state = Generator.stateFor(model, total);
            CachedSession<S> resumed =
                    CachedSession.resume(model, tree(), state, prompt, total - 1);
            restored = resumed.position();
            remaining = CachedSession.tail(prompt, restored);
        } else {
            state = Generator.stateFor(model, total);
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

    // ---- the shared request policy: sampling stack, constraint wiring, forced calls ----
    // Statics over LoadedModel so the server (which has no ChatEngine) shares the same code.

    /**
     * The standard jinfer sampling stack: (temperature, topP, seed) plus the reasoning policy -
     * thinking on caps the think span so it cannot starve the visible answer ({@code
     * reasoningOverride}: null = half of {@code maxTokens}, -1 = uncapped); thinking off masks the
     * think markers outright.
     */
    public static Sampler sampler(
            LoadedModel<?> m,
            float temperature,
            float topP,
            long seed,
            boolean think,
            int maxTokens,
            Integer reasoningOverride) {
        Sampler sampler =
                Sampler.select(m.model().config().vocabularySize(), temperature, topP, seed);
        if (!think) {
            return Thinking.banMarkers(sampler, m.tokenizer());
        }
        int budget =
                reasoningOverride != null
                        ? reasoningOverride
                        : maxTokens >= 0 ? Math.max(1, maxTokens / 2) : -1;
        return Thinking.capBudget(sampler, m.tokenizer(), budget);
    }

    /**
     * Grammar-constrained decoding over any compiled cursor (JSON mode, JSON schema, raw GBNF):
     * dormant through the think span (constraining from token 0 would suppress reasoning), the
     * newline after {@code </think>} passed through, and a dead-end forcing one of the model's own
     * stop tokens.
     */
    public static Sampler constrained(
            LoadedModel<?> m, Sampler s, Grammar.Cursor g, boolean think) {
        int eos = m.stopTokens().iterator().next();
        int close = think ? SpecialTokens.find(m.tokenizer(), "</think>").orElse(-1) : -1;
        int open = close >= 0 ? SpecialTokens.find(m.tokenizer(), "<think>").orElse(-1) : -1;
        // native codecs know whether the generation prompt already OPENED the span; then the
        // grammar stays dormant until the close. Otherwise the first token decides: the model
        // may open a span, or answer - grammar-constrained from token zero (a think-capable
        // vocab on a model that answers directly must not leave the grammar dormant forever)
        boolean startInThink =
                close >= 0 && m.template().map(t -> t.replySeed(true).length > 0).orElse(false);
        int[] skipNl = close >= 0 ? SpecialTokens.newlineTokens(m.tokenizer()) : null;
        return Sampler.withGrammar(s, g, eos, open, close, startInThink, skipNl);
    }

    /**
     * The complete forced-call recipe as ONE value - the three parts only work together, so a
     * caller can never seed without pre-feeding the parser (the historical bug class): {@code seed}
     * joins the prompt (the model can only COMPLETE a call), {@code sampler} prefix-pins the
     * offered names and forces the family's header epilogue before releasing, {@code parserSeed}
     * puts the reply parser in the seeded span state. A forced reply never thinks (it is seeded
     * INTO the call block), so the prompt must be rendered with thinking off.
     */
    public record ForcedCall(Batch seed, Sampler sampler, int[] parserSeed) {}

    /**
     * The recipe for forcing a call to one of {@code tools}, or empty when this model cannot force
     * (no native codec, or its template declares no call seed) - the caller's cue to reject.
     */
    public static Optional<ForcedCall> forceCall(LoadedModel<?> m, List<Tool> tools, Sampler base) {
        ChatTemplate template = m.template().orElse(null);
        if (template == null || template.callSeed().length == 0) return Optional.empty();
        int[] callSeed = template.callSeed();
        Sampler sampler = base;
        Optional<String> pin = template.callGrammar(tools);
        if (pin.isPresent()) {
            sampler =
                    Sampler.withPrefixGrammar(
                            base,
                            Grammar.of(pin.get(), m.tokenizer()).cursor(),
                            m.stopTokens().iterator().next(),
                            template.callEpilogue());
        }
        int[] replySeed = template.replySeed(false);
        int[] parserSeed = new int[replySeed.length + callSeed.length];
        System.arraycopy(replySeed, 0, parserSeed, 0, replySeed.length);
        System.arraycopy(callSeed, 0, parserSeed, replySeed.length, callSeed.length);
        return Optional.of(new ForcedCall(Batch.prefill(callSeed), sampler, parserSeed));
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
