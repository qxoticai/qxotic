package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import java.nio.file.Path;
import java.util.List;
import org.springframework.ai.tool.ToolCallback;

/**
 * The Spring AI adapter over the shared {@link ChatEngine}: the runtime (loading, encoding,
 * generation lock, cached prompts, session pool) is framework-neutral; this class contributes only
 * what is Spring AI's - lazy message/tool maps for the Jinja fallback. The engine's neutral {@link
 * UnsupportedOperationException}s pass through unchanged (Spring AI has no dedicated
 * unsupported-feature exception type).
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
            Conversation conversation,
            List<org.springframework.ai.chat.messages.Message> messages,
            List<ToolCallback> toolCallbacks) {
        return engine.encode(
                conversation,
                () -> JinferMappings.toMessageMaps(messages),
                () -> toolCallbacks == null ? List.of() : JinferMappings.toToolMaps(toolCallbacks));
    }

    ChatEngine.Encoded encodeNative(Conversation conversation) {
        return engine.encodeNative(conversation);
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
        engine.define(prefix);
    }

    void freezePrompts(Path out) {
        engine.freezePrompts(out);
    }

    void close() {
        engine.close();
    }

    /** Test seams (see {@link ChatEngine}). */
    /** Runs a streaming generation on the engine's single lazy driver thread. */
    void stream(Runnable generation) {
        engine.stream(generation);
    }

    String promptStats() {
        return engine.promptStats();
    }

    String sessionStats() {
        return engine.sessionStats();
    }
}
