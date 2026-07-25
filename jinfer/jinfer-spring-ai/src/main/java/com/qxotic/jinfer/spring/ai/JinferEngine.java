package com.qxotic.jinfer.spring.ai;

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
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.chat.UnsupportedConversation;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.toknroll.IntSequence;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.locks.ReentrantLock;
import org.springframework.ai.tool.ToolCallback;

/**
 * The shared jinfer runtime behind {@link JinferChatModel} and its cached-prompt views: one loaded
 * model, the two-tier template stack (native codec first, hardened Jinja whole-render fallback),
 * the single-stream generation lock (a jinfer model runs one generation at a time; concurrent
 * {@code call}s queue), and the cached-prompt block tree.
 */
final class JinferEngine {

    final LoadedModel<?> loaded;
    final String modelName;
    private final JinjaChatTemplate jinja;
    private final ReentrantLock lock = new ReentrantLock(true);
    private final CacheStore promptStore; // owned: its confined arenas close with the engine
    private final PromptCache<?> prompts; // the cached-prompt block tree (empty when unused)
    private volatile boolean closed;

    JinferEngine(Path modelPath, Path mediaProjector, int contextLength, Path cachedPrompts) {
        try {
            this.loaded =
                    mediaProjector == null
                            ? Models.load(modelPath, contextLength)
                            : Models.load(modelPath, mediaProjector, contextLength);
            this.promptStore = CacheStore.inMemory();
            this.prompts =
                    tree(
                            loaded,
                            promptStore,
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
            LoadedModel<S> loaded, CacheStore store, FrozenBlocks base) {
        // unbounded: prompts are explicit, few, and deliberately paid for
        return new PromptCache<>(loaded.codec(), store, Long.MAX_VALUE, loaded.seed(), base);
    }

    /** Idempotent: frees the tree's native arenas; later use fails with IllegalStateException. */
    void close() {
        lock.lock();
        try {
            closed = true;
            promptStore.close();
        } finally {
            lock.unlock();
        }
    }

    private void checkOpen() {
        if (closed) throw new IllegalStateException("the model is closed");
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
            Conversation conversation,
            List<org.springframework.ai.chat.messages.Message> messages,
            List<ToolCallback> toolCallbacks) {
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
        IntSequence ids =
                jinja.render(
                        JinferMappings.toMessageMaps(messages),
                        toolCallbacks == null || toolCallbacks.isEmpty()
                                ? null
                                : JinferMappings.toToolMaps(toolCallbacks),
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
     * The generation result plus the request's cache accounting (tokens restored from the tree).
     */
    record Outcome(Generator.GenerationResult result, long restoredTokens) {}

    /**
     * One generation pass under the engine lock; a fresh state per request. {@code cached} routes
     * through the prompt tree (resume the longest defined prefix, prefill only the rest) - the
     * uncached path never touches the tree, keeping the base model fully stateless.
     */
    Outcome generate(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink,
            boolean cached) {
        lock.lock();
        try {
            checkOpen();
            return cached
                    ? cachedRun(
                            loaded.model(), tree(), prompt, sampler, maxTokens, timeoutNanos, sink)
                    : new Outcome(
                            run(loaded.model(), prompt, sampler, maxTokens, timeoutNanos, sink), 0);
        } finally {
            lock.unlock();
        }
    }

    private <S extends RuntimeState> Generator.GenerationResult run(
            LanguageModel<?, ?, S> model,
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink) {
        int promptLen = prompt.stream().mapToInt(Batch::count).sum();
        S state = Generator.stateFor(model, promptLen);
        List<Batch> prepared = Batch.prepare(prompt, state.batchCapacity());
        return Generator.generate(
                model,
                state,
                prepared,
                sampler,
                maxTokens,
                timeoutNanos,
                loaded.stopTokens(),
                sink);
    }

    /** Structured reply via the native parser, or a plain-text message when the model has none. */
    Message decode(Optional<ChatTemplate> template, IntSequence replyTokens) {
        if (template.isPresent()) {
            return template.get().decode(replyTokens);
        }
        return new Message(Role.ASSISTANT, loaded.tokenizer().decode(replyTokens));
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
                                        new UnsupportedOperationException(
                                                "cached prompts need a native chat-template codec;"
                                                        + " this model only has the Jinja"
                                                        + " whole-render (no prefix-stability"
                                                        + " guarantee)"));
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
            checkOpen();
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

    /**
     * A generation pass that resumes the longest tree-cached prefix and hands the generator only
     * the unrestored tail (the generator ingests it - the tail is never committed, so the tree
     * holds defined prompts only and serving stays stateless).
     */
    private <S extends RuntimeState> Outcome cachedRun(
            LanguageModel<?, ?, S> model,
            PromptCache<S> cache,
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink) {
        long[] fp = CachedSession.fingerprints(prompt);
        S state = Generator.stateFor(model, fp.length);
        // cap at length-1: at least one position re-ingests, leaving fresh logits at the cursor
        CachedSession<S> s = CachedSession.resume(model, cache, state, fp, fp.length - 1);
        int restored = s.position(); // the cache read, before the tail/generation advances it
        return new Outcome(
                Generator.generate(
                        model,
                        state,
                        CachedSession.tail(prompt, restored),
                        sampler,
                        maxTokens,
                        timeoutNanos,
                        loaded.stopTokens(),
                        sink),
                restored);
    }

    /** Freezes the whole tree (mounted base + everything defined) into one artifact. */
    void freezePrompts(Path out) {
        lock.lock();
        try {
            checkOpen();
            prompts.freeze(out);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to save cached prompts to " + out, e);
        } finally {
            lock.unlock();
        }
    }

    /** Test seam: the tree's stats line ("blocks=.. hits=.." - see PromptCache.stats). */
    String promptStats() {
        return prompts.stats();
    }

    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> PromptCache<S> tree() {
        return (PromptCache<S>) prompts;
    }
}
