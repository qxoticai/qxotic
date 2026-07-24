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
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.Role;
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
    private PromptCache<?> prompts; // the cached-prompt block tree; created lazily (or on mount)

    JinferEngine(Path modelPath, Path mediaProjector, int contextLength, Path cachedPrompts) {
        try {
            this.loaded =
                    mediaProjector == null
                            ? Models.load(modelPath, contextLength)
                            : Models.load(modelPath, mediaProjector, contextLength);
            if (cachedPrompts != null) {
                this.prompts = tree(loaded, FrozenBlocks.open(cachedPrompts, loaded.seed()));
            }
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
                    "this model cannot frame media"
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

    /** One generation pass under the engine lock; a fresh state per request. */
    Generator.GenerationResult generate(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink) {
        lock.lock();
        try {
            return run(loaded.model(), prompt, sampler, maxTokens, timeoutNanos, sink);
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
    List<Batch> encodeNative(Conversation conversation) {
        ChatTemplate template =
                loaded.template()
                        .orElseThrow(
                                () ->
                                        new UnsupportedFeatureException(
                                                "cached prompts need a native chat-template codec;"
                                                    + " this model only has the Jinja whole-render"
                                                    + " (no prefix-stability guarantee)"));
        return template.encode(conversation);
    }

    /**
     * Defines (prefills) a cached prompt: dedups against the tree, commits one block per encoded
     * batch (turn boundaries), then discards the working state - the blocks hold the KV.
     */
    void define(Conversation prefix) {
        List<Batch> prompt = encodeNative(prefix);
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
        CachedSession<S> s = CachedSession.resume(model, cache, state, fp, fp.length);
        s.ingestGroups(prompt.stream().map(List::of).toList());
    }

    /**
     * A generation pass that resumes the longest tree-cached prefix and prefills only the rest. The
     * suffix ingests DIRECTLY on the state (never committed): the tree holds defined prompts only,
     * so serving stays stateless and the tree bounded.
     */
    Generator.GenerationResult cachedGenerate(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink) {
        lock.lock();
        try {
            return cachedRun(
                    loaded.model(), tree(), prompt, sampler, maxTokens, timeoutNanos, sink);
        } finally {
            lock.unlock();
        }
    }

    private <S extends RuntimeState> Generator.GenerationResult cachedRun(
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
        List<Batch> suffix = suffix(prompt, s.position());
        for (Batch b : Batch.prepare(suffix, state.batchCapacity())) {
            model.ingest(state, b);
        }
        return Generator.generate(
                model,
                state,
                List.of(),
                sampler,
                maxTokens,
                timeoutNanos,
                loaded.stopTokens(),
                sink);
    }

    /**
     * The batch-list tail after {@code skip} restored positions; a token batch at the seam is
     * sliced (blocks restore whole media groups, so the seam can only land inside tokens).
     */
    private static List<Batch> suffix(List<Batch> prompt, int skip) {
        List<Batch> out = new java.util.ArrayList<>();
        int pos = 0;
        for (Batch b : prompt) {
            int n = b.count();
            if (pos + n <= skip) {
                pos += n;
                continue;
            }
            if (pos >= skip) {
                out.add(b);
            } else {
                int[] ids = ((Batch.Input.Tokens) b.input()).ids();
                out.add(Batch.prefill(java.util.Arrays.copyOfRange(ids, skip - pos, n)));
            }
            pos += n;
        }
        return out;
    }

    /** Freezes the whole tree (mounted base + everything defined) into one artifact. */
    void freezePrompts(Path out) {
        lock.lock();
        try {
            if (prompts == null) {
                throw new IllegalStateException("no cached prompts defined - nothing to save");
            }
            prompts.freeze(out);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to save cached prompts to " + out, e);
        } finally {
            lock.unlock();
        }
    }

    /** Test seam: the tree's stats line ("blocks=.. hits=.." - see PromptCache.stats), or "". */
    String promptStats() {
        return prompts == null ? "" : prompts.stats();
    }

    @SuppressWarnings("unchecked")
    private <S extends RuntimeState> PromptCache<S> tree() {
        if (prompts == null) {
            prompts = tree((LoadedModel<S>) loaded, null);
        }
        return (PromptCache<S>) prompts;
    }
}
