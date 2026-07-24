package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.cache.CacheStore;
import com.qxotic.jinfer.cache.CachedSession;
import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JinjaChatTemplate;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.toknroll.IntSequence;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The shared jinfer runtime behind {@link JinferChatModel} and {@link JinferStreamingChatModel}:
 * one loaded model, the two-tier template stack (native codec first, hardened Jinja whole-render
 * fallback), and the single-stream generation lock (a jinfer model runs one generation at a time;
 * concurrent {@code chat} calls queue).
 */
final class JinferEngine {

    final LoadedModel<?> loaded;
    final String modelName;
    private final JinjaChatTemplate jinja;
    private final ReentrantLock lock = new ReentrantLock(true);
    private final PromptCache<?> prompts; // the cached-prompt block tree (empty when unused)
    // cachedSessions(n): the last n live conversation states, reused append-only when a new
    // prompt's fingerprint stream strictly extends one (all access under the generation lock)
    private final java.util.ArrayDeque<LiveSession> sessions = new java.util.ArrayDeque<>();
    private final int sessionCapacity;
    private int sessionHits;

    /** A finished generation's state with the fingerprints of everything ingested into it. */
    private record LiveSession(RuntimeState state, long[] fp) {}

    JinferEngine(
            Path modelPath,
            Path mediaProjector,
            int contextLength,
            Path cachedPrompts,
            int cachedSessions) {
        this.sessionCapacity = Math.max(0, cachedSessions);
        try {
            this.loaded =
                    mediaProjector == null
                            ? Models.load(modelPath, contextLength)
                            : Models.load(modelPath, mediaProjector, contextLength);
            // built only when the model can support it (or a mount demands it): a codec-less
            // model (Qwen35) must still load and chat - the codec throw belongs to the first
            // CACHED-feature use, not to plain construction
            this.prompts =
                    cachedPrompts == null && loaded.model().stateCodec().isEmpty()
                            ? null
                            : tree(
                                    loaded,
                                    cachedPrompts == null
                                            ? null
                                            : FrozenBlocks.open(cachedPrompts, loaded.seed()));
        } catch (IOException e) {
            throw new UncheckedIOException("failed to load " + modelPath, e);
        }
        this.modelName = modelPath.getFileName().toString();
        this.jinja = new JinjaChatTemplate(loaded.tokenizer(), loaded.chatTemplateSource());
    }

    private static <S extends RuntimeState> PromptCache<S> tree(
            LoadedModel<S> loaded, FrozenBlocks base) {
        // unbounded: prompts are explicit, few, and deliberately paid for
        return new PromptCache<>(
                loaded.codec(), CacheStore.inMemory(), Long.MAX_VALUE, loaded.seed(), base);
    }

    /** The encoded prompt plus the reply parser when the model has a native codec. */
    record Encoded(List<Batch> prompt, Optional<ChatTemplate> template) {}

    /**
     * Native-first encode: the model's own codec when it can frame the conversation byte-exactly,
     * else the scrubbed Jinja whole-render (its maps built lazily from the original request only
     * when the punt actually happens). Media never reaches the text-only fallback - it fails loudly
     * instead of being silently dropped.
     */
    Encoded encode(
            Conversation conversation, List<ChatMessage> messages, List<ToolSpecification> tools) {
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
            throw new UnsupportedFeatureException(
                    "image/audio/video input is not supported by this model"
                            + (punted != null ? ": " + punted.getMessage() : "")
                            + " (for Gemma 4, pass the mmproj GGUF via mediaProjector(...))");
        }
        IntSequence ids =
                jinja.render(
                        Mappings.toMessageMaps(messages),
                        tools.isEmpty() ? null : Mappings.toToolMaps(tools),
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
     * One generation pass under the engine lock; a fresh state per request. {@code cached} routes
     * through the prompt tree (resume the longest defined prefix, prefill only the rest) - the
     * uncached path never touches the tree, keeping the base model fully stateless.
     */
    Generator.GenerationResult generate(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink,
            boolean cached) {
        lock.lock();
        try {
            return run(loaded.model(), prompt, sampler, maxTokens, timeoutNanos, sink, cached);
        } finally {
            lock.unlock();
        }
    }

    /**
     * One generation pass, cheapest source first: a pooled live session when the prompt strictly
     * extends one ({@code cachedSessions} - zero restore, only the delta prefills), else the prompt
     * tree for cached views (block restore, resume capped one short so the final block re-ingests
     * and leaves fresh logits), else a fresh state. On success the finished state (prompt + reply
     * KV) returns to the pool for the conversation's next turn.
     */
    private <S extends RuntimeState> Generator.GenerationResult run(
            LanguageModel<?, ?, S> model,
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink,
            boolean cached) {
        long[] fp = CachedSession.fingerprints(prompt);
        S state;
        List<Batch> remaining;
        LiveSession pooled = acquireSession(fp);
        if (pooled != null) {
            sessionHits++;
            @SuppressWarnings("unchecked")
            S reused = (S) pooled.state();
            state = reused;
            remaining = CachedSession.tail(prompt, pooled.fp().length);
        } else if (cached) {
            state = Generator.stateFor(model, fp.length);
            CachedSession<S> resumed =
                    CachedSession.resume(model, tree(), state, fp, fp.length - 1);
            remaining = CachedSession.tail(prompt, resumed.position());
        } else {
            state = Generator.stateFor(model, fp.length);
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
        poolSession(state, fp, result); // not reached on throw: a torn state is never pooled
        return result;
    }

    /** The pooled session with the LONGEST stream strictly extended by {@code fp}, removed. */
    private LiveSession acquireSession(long[] fp) {
        LiveSession best = null;
        for (LiveSession s : sessions) {
            int len = s.fp().length;
            if (len < fp.length
                    && s.state().contextCapacity() >= fp.length
                    && java.util.Arrays.equals(s.fp(), 0, len, fp, 0, len)
                    && (best == null || len > best.fp().length)) {
                best = s;
            }
        }
        if (best != null) sessions.remove(best);
        return best;
    }

    /** Pools the finished state under prompt + ingested-reply fingerprints; evicts past cap. */
    private void poolSession(RuntimeState state, long[] promptFp, Generator.GenerationResult r) {
        if (sessionCapacity == 0) return;
        int ingested = state.position() - promptFp.length;
        if (ingested < 0) return;
        long[] fp = java.util.Arrays.copyOf(promptFp, promptFp.length + ingested);
        for (int i = 0; i < ingested; i++) fp[promptFp.length + i] = r.tokens().intAt(i);
        sessions.addLast(new LiveSession(state, fp));
        while (sessions.size() > sessionCapacity) sessions.removeFirst();
    }

    /** Test seam: live-session pool occupancy and hit count. */
    String sessionStats() {
        lock.lock();
        try {
            return "sessions=" + sessions.size() + " hits=" + sessionHits;
        } finally {
            lock.unlock();
        }
    }

    // ---- cached prompts: the block tree behind withCachedPrompt / save / load ----

    /**
     * Encode via the native codec only - cached prompts are a prefix-stability bet the Jinja
     * whole-render cannot honor.
     */
    Encoded encodeNative(Conversation conversation) {
        ChatTemplate template =
                loaded.template()
                        .orElseThrow(
                                () ->
                                        new UnsupportedFeatureException(
                                                "cached prompts need a native chat-template codec;"
                                                    + " this model only has the Jinja whole-render"
                                                    + " (no prefix-stability guarantee)"));
        return new Encoded(template.encode(conversation), Optional.of(template));
    }

    /**
     * Defines (prefills) a cached prompt: dedups against the tree, commits one block per encoded
     * batch (turn boundaries), then discards the working state - the blocks hold the KV.
     */
    void define(Conversation prefix) {
        List<Batch> prompt = encodeNative(prefix).prompt();
        lock.lock();
        try {
            defineOn(loaded.model(), tree(), prompt);
        } finally {
            lock.unlock();
        }
    }

    private <S extends RuntimeState> void defineOn(
            LanguageModel<?, ?, S> model, PromptCache<S> cache, List<Batch> prompt) {
        long[] fp = CachedSession.fingerprints(prompt);
        S state = Generator.stateFor(model, fp.length);
        CachedSession<S> s = CachedSession.resume(model, cache, state, fp);
        s.ingestGroups(prompt.stream().map(List::of).toList());
    }

    /** Freezes the whole tree (mounted base + everything defined) into one artifact. */
    void freezePrompts(Path out) {
        lock.lock();
        try {
            tree().freeze(out);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to save cached prompts to " + out, e);
        } finally {
            lock.unlock();
        }
    }

    /** Test seam: the tree's stats line ("blocks=.. hits=.." - see PromptCache.stats). */
    String promptStats() {
        return tree().stats();
    }

    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> PromptCache<S> tree() {
        if (prompts == null) loaded.codec(); // throws, naming the missing state codec
        return (PromptCache<S>) prompts;
    }
}
