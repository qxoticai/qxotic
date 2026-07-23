package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.PendingUtf8;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.llm.Generator;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.TokenUsage;
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

    private JinferStreamingChatModel(
            JinferEngine engine,
            ChatRequestParameters defaults,
            boolean thinking,
            long seed,
            long timeoutNanos) {
        this.engine = engine;
        this.defaults = defaults;
        this.thinking = thinking;
        this.seed = seed;
        this.timeoutNanos = timeoutNanos;
    }

    static JinferStreamingChatModel over(
            JinferEngine engine,
            ChatRequestParameters defaults,
            boolean thinking,
            long seed,
            long timeoutNanos) {
        return new JinferStreamingChatModel(engine, defaults, thinking, seed, timeoutNanos);
    }

    /** Standalone construction; prefer {@link JinferChatModel#streaming()} to share one engine. */
    public static JinferStreamingChatModel from(JinferChatModel blocking) {
        return blocking.streaming();
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
        JinferChatModel.Prepared p = JinferChatModel.prepare(engine, request, thinking, seed);
        Optional<ChatTemplate> template = p.encoded().template();
        ReplyParser parser = template.map(ChatTemplate::parser).orElse(null);
        PendingUtf8 raw = parser == null ? new PendingUtf8() : null;

        Generator.GenerationResult result =
                engine.generate(
                        p.encoded().prompt(),
                        p.sampler(),
                        p.maxTokens(),
                        timeoutNanos,
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
                                                engine.loaded
                                                        .tokenizer()
                                                        .decodeBytes(new int[] {token}),
                                                token);
                                if (!f.text().isEmpty()) handler.onPartialResponse(f.text());
                            }
                            return true;
                        });

        AiMessage ai =
                parser != null
                        ? Mappings.toAiMessage(parser.finish())
                        : Mappings.toAiMessage(engine.decode(Optional.empty(), result.tokens()));
        handler.onCompleteResponse(
                ChatResponse.builder()
                        .aiMessage(ai)
                        .modelName(engine.modelName)
                        .tokenUsage(new TokenUsage(p.promptTokens(), result.completionTokens()))
                        .finishReason(
                                Mappings.toFinishReason(
                                        result.finishReason(), ai.hasToolExecutionRequests()))
                        .build());
    }
}
