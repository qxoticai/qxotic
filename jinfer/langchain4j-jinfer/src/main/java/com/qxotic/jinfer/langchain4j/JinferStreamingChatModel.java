package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.PendingUtf8;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.llm.Generator;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.List;
import java.util.Optional;

/**
 * Streaming twin of {@link JinferChatModel}: one virtual thread per request drives the generation
 * and forwards the native {@link ReplyParser}'s two lanes to langchain4j's two streaming channels -
 * content fragments to {@code onPartialResponse}, reasoning fragments to {@code onPartialThinking}.
 * Models without a native parser stream raw decoded text.
 */
public final class JinferStreamingChatModel implements StreamingChatModel {

    private final JinferEngine engine;
    private final ChatRequestParameters defaults;
    private final boolean thinking;
    private final long seed;
    private final long timeoutNanos;
    private final List<dev.langchain4j.data.message.ChatMessage> prefixMessages;
    private final List<dev.langchain4j.agent.tool.ToolSpecification> prefixTools;

    private JinferStreamingChatModel(
            JinferEngine engine,
            ChatRequestParameters defaults,
            boolean thinking,
            long seed,
            long timeoutNanos,
            List<dev.langchain4j.data.message.ChatMessage> prefixMessages,
            List<dev.langchain4j.agent.tool.ToolSpecification> prefixTools) {
        this.engine = engine;
        this.defaults = defaults;
        this.thinking = thinking;
        this.seed = seed;
        this.timeoutNanos = timeoutNanos;
        this.prefixMessages = prefixMessages;
        this.prefixTools = prefixTools;
    }

    static JinferStreamingChatModel over(
            JinferEngine engine,
            ChatRequestParameters defaults,
            boolean thinking,
            long seed,
            long timeoutNanos,
            List<dev.langchain4j.data.message.ChatMessage> prefixMessages,
            List<dev.langchain4j.agent.tool.ToolSpecification> prefixTools) {
        return new JinferStreamingChatModel(
                engine, defaults, thinking, seed, timeoutNanos, prefixMessages, prefixTools);
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return defaults;
    }

    @Override
    public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        Thread.ofVirtual()
                .name("jinfer-stream")
                .start(
                        () -> {
                            try {
                                stream(request, handler);
                            } catch (Throwable t) {
                                handler.onError(t);
                            }
                        });
    }

    private void stream(ChatRequest request, StreamingChatResponseHandler handler) {
        JinferChatModel.Prepared p =
                JinferChatModel.prepare(
                        engine, request, thinking, seed, prefixMessages, prefixTools);
        Optional<ChatTemplate> template = p.encoded().template();
        ReplyParser parser = template.map(ChatTemplate::parser).orElse(null);
        PendingUtf8 raw = parser == null ? new PendingUtf8() : null;

        Generator.TokenSink sink =
                token -> {
                    if (parser != null) {
                        String fragment = parser.feed(token);
                        if (!fragment.isEmpty()) {
                            if (parser.reasoning()) {
                                handler.onPartialThinking(new PartialThinking(fragment));
                            } else {
                                handler.onPartialResponse(fragment);
                            }
                        }
                    } else {
                        PendingUtf8.Fragment f =
                                raw.add(
                                        engine.loaded.tokenizer().decodeBytes(new int[] {token}),
                                        token);
                        if (!f.text().isEmpty()) handler.onPartialResponse(f.text());
                    }
                    return true;
                };
        Generator.GenerationResult result =
                p.cached()
                        ? engine.cachedGenerate(
                                p.encoded().prompt(),
                                p.sampler(),
                                p.maxTokens(),
                                timeoutNanos,
                                sink)
                        : engine.generate(
                                p.encoded().prompt(),
                                p.sampler(),
                                p.maxTokens(),
                                timeoutNanos,
                                sink);

        AiMessage ai =
                parser != null
                        ? Mappings.toAiMessage(parser.finish())
                        : Mappings.toAiMessage(engine.decode(Optional.empty(), result.tokens()));
        handler.onCompleteResponse(
                Mappings.response(engine.modelName, ai, p.promptTokens(), result));
    }
}
