package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
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

    JinferEngine(Path modelPath, Path mediaProjector, int contextLength) {
        try {
            this.loaded =
                    mediaProjector == null
                            ? Models.load(modelPath, contextLength)
                            : Models.load(modelPath, mediaProjector, contextLength);
        } catch (IOException e) {
            throw new UncheckedIOException("failed to load " + modelPath, e);
        }
        this.modelName = modelPath.getFileName().toString();
        this.jinja = new JinjaChatTemplate(loaded.tokenizer(), loaded.chatTemplateSource());
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
}
