package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import java.nio.file.Path;
import java.util.List;

/**
 * The langchain4j adapter over the shared {@link ChatEngine}: the runtime (loading, encoding,
 * generation lock, cached prompts, session pool) is framework-neutral; this class contributes only
 * what is langchain4j's - lazy message/tool maps for the Jinja fallback and the framework's {@link
 * UnsupportedFeatureException} for unsupported features.
 */
final class JinferEngine {

    final ChatEngine engine;
    final LoadedModel<?> loaded;
    final String modelName;

    JinferEngine(
            Path modelPath,
            Path mediaProjector,
            int contextLength,
            Path cachedPrompts,
            int cachedSessions) {
        this.engine =
                new ChatEngine(
                        modelPath, mediaProjector, contextLength, cachedPrompts, cachedSessions);
        this.loaded = engine.loaded();
        this.modelName = engine.modelName();
    }

    ChatEngine.Encoded encode(
            Conversation conversation, List<ChatMessage> messages, List<ToolSpecification> tools) {
        try {
            return engine.encode(
                    conversation,
                    () -> Mappings.toMessageMaps(messages),
                    () -> Mappings.toToolMaps(tools));
        } catch (UnsupportedOperationException e) {
            throw new UnsupportedFeatureException(e.getMessage());
        }
    }

    ChatEngine.Encoded encodeNative(Conversation conversation) {
        try {
            return engine.encodeNative(conversation);
        } catch (UnsupportedOperationException e) {
            throw new UnsupportedFeatureException(e.getMessage());
        }
    }

    ChatEngine.Outcome generate(
            List<Batch> prompt,
            Sampler sampler,
            int maxTokens,
            long timeoutNanos,
            Generator.TokenSink sink,
            boolean cached) {
        return engine.generate(prompt, sampler, maxTokens, timeoutNanos, sink, cached);
    }

    void define(Conversation prefix) {
        try {
            engine.define(prefix);
        } catch (UnsupportedOperationException e) {
            throw new UnsupportedFeatureException(e.getMessage());
        }
    }

    void freezePrompts(Path out) {
        engine.freezePrompts(out);
    }

    void close() {
        engine.close();
    }

    /** A sibling adapter over {@link ChatEngine#fork()}: shared model, fresh pipeline. */
    JinferEngine fork() {
        return new JinferEngine(engine.fork(), modelName);
    }

    private JinferEngine(ChatEngine engine, String modelName) {
        this.engine = engine;
        this.modelName = modelName;
        this.loaded = engine.loaded();
    }

    /** Runs a streaming generation on the engine's single lazy driver thread. */
    void stream(Runnable generation) {
        engine.stream(generation);
    }

    /** Test seam: the prompt-tree stats string ({@code blocks=.. hits=..}). */
    String promptStats() {
        return engine.promptStats();
    }

    /** Test seam: the session-pool stats string ({@code hits=..}). */
    String sessionStats() {
        return engine.sessionStats();
    }
}
