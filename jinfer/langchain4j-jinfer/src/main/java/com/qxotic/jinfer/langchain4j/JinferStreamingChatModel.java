package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.PendingUtf8;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.llm.Generator;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Streaming twin of {@link JinferChatModel}: one virtual thread per request drives the generation
 * and forwards the native {@link ReplyParser}'s two lanes to langchain4j's two streaming channels -
 * content fragments to {@code onPartialResponse} (with a cancellation handle), reasoning fragments
 * to {@code onPartialThinking}. Models without a native parser stream raw decoded text.
 *
 * <p>Contract details (the streaming compliance kit's): invalid requests throw synchronously from
 * {@code chat}; a cancelled handle stops generation and suppresses {@code onCompleteResponse};
 * exceptions thrown by user callbacks are forwarded to {@code onError} per occurrence while the
 * generation continues; parsed tool calls are announced via {@code onCompleteToolCall} before the
 * complete response.
 */
public final class JinferStreamingChatModel implements StreamingChatModel {

    private final JinferChatModel model; // the blocking twin: engine, defaults, prefix, listeners

    JinferStreamingChatModel(JinferChatModel model) {
        this.model = model;
    }

    @Override
    public ChatRequestParameters defaultRequestParameters() {
        return model.defaults;
    }

    @Override
    public List<ChatModelListener> listeners() {
        return model.listeners;
    }

    @Override
    public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        JinferChatModel.validate(model, request); // invalid requests throw HERE, synchronously
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
        JinferEngine engine = model.engine;
        JinferChatModel.Prepared p = JinferChatModel.prepare(model, request);
        Optional<ChatTemplate> template = p.encoded().template();
        ReplyParser parser = template.map(ChatTemplate::parser).orElse(null);
        PendingUtf8 raw = parser == null ? new PendingUtf8() : null;

        AtomicBoolean cancelled = new AtomicBoolean();
        StreamingHandle handle =
                new StreamingHandle() {
                    @Override
                    public void cancel() {
                        cancelled.set(true);
                    }

                    @Override
                    public boolean isCancelled() {
                        return cancelled.get();
                    }
                };
        PartialResponseContext context = new PartialResponseContext(handle);

        Generator.TokenSink sink =
                token -> {
                    if (cancelled.get()) return false;
                    String fragment;
                    boolean reasoning;
                    if (parser != null) {
                        fragment = parser.feed(token);
                        reasoning = parser.reasoning();
                    } else {
                        fragment =
                                raw.add(
                                                engine.loaded
                                                        .tokenizer()
                                                        .decodeBytes(new int[] {token}),
                                                token)
                                        .text();
                        reasoning = false;
                    }
                    if (!fragment.isEmpty()) {
                        if (reasoning) {
                            safely(
                                    handler,
                                    () -> handler.onPartialThinking(new PartialThinking(fragment)));
                        } else {
                            safely(
                                    handler,
                                    () ->
                                            handler.onPartialResponse(
                                                    new PartialResponse(fragment), context));
                        }
                    }
                    return !cancelled.get();
                };

        Generator.GenerationResult result =
                engine.generate(
                        p.encoded().prompt(),
                        p.sampler(),
                        p.maxTokens(),
                        model.timeoutNanos,
                        sink,
                        p.cached());

        if (cancelled.get()) {
            return; // a cancelled stream ends silently: no complete callback
        }
        AiMessage ai =
                parser != null
                        ? Mappings.toAiMessage(parser.finish())
                        : Mappings.toAiMessage(engine.decode(Optional.empty(), result.tokens()));
        if (ai.hasToolExecutionRequests()) {
            for (int i = 0; i < ai.toolExecutionRequests().size(); i++) {
                CompleteToolCall call = new CompleteToolCall(i, ai.toolExecutionRequests().get(i));
                safely(handler, () -> handler.onCompleteToolCall(call));
            }
        }
        ChatResponse response = Mappings.response(engine.modelName, ai, p.promptTokens(), result);
        safely(handler, () -> handler.onCompleteResponse(response));
    }

    /** User-callback exceptions go to onError, once per occurrence; the stream carries on. */
    private static void safely(StreamingChatResponseHandler handler, Runnable callback) {
        try {
            callback.run();
        } catch (Throwable t) {
            try {
                handler.onError(t);
            } catch (Throwable ignored) {
                // an onError that itself throws has nowhere left to report
            }
        }
    }
}
