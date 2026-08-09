package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.llm.TextStops;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.Capability;
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
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Streaming twin of {@link JinferChatModel} - built via {@code
 * JinferChatModel.builder()....buildStreaming()}, or from a live model via {@code streaming()} (the
 * GGUF loads once either way): the engine's single lazy driver thread runs the generation and
 * forwards the native {@link ReplyParser}'s two lanes to langchain4j's two streaming channels -
 * content fragments to {@code onPartialResponse} (with a cancellation handle), reasoning fragments
 * to {@code onPartialThinking}. Models without a native parser stream raw decoded text.
 *
 * <p>Contract details (the streaming compliance kit's): invalid requests throw synchronously from
 * {@code chat}; a cancelled handle stops generation and suppresses {@code onCompleteResponse};
 * exceptions thrown by user callbacks are forwarded to {@code onError} per occurrence while the
 * generation continues; parsed tool calls are announced via {@code onCompleteToolCall} before the
 * complete response.
 *
 * <p>One divergence from the blocking twin, and it is langchain4j's: {@code
 * StreamingChatModel.chat} calls {@code doChat} with no try/catch, so a request this provider
 * rejects SYNCHRONOUSLY (an unsupported parameter, unframeable media) reaches the caller as a
 * thrown exception but never reaches a registered {@link ChatModelListener} - where {@code
 * ChatModel.chat} would have reported {@code onError}. Listeners see {@code onRequest} with no
 * terminal event. Reporting it here is not on offer: the listener plumbing ({@code
 * ChatModelListenerUtils}, the request's attribute map) is package-private in core, and a
 * hand-rolled notification would double-report the day core adds the catch. Catch around {@code
 * chat} for those, or use the blocking twin's listeners.
 */
public final class JinferStreamingChatModel implements StreamingChatModel, AutoCloseable {

    private final JinferChatModel model; // the blocking twin: engine, defaults, prefix, listeners

    /** Closes the shared engine (see {@link JinferChatModel#close()}); blocking, idempotent. */
    @Override
    public void close() {
        model.close();
    }

    JinferStreamingChatModel(JinferChatModel model) {
        this.model = model;
    }

    /**
     * The blocking twin over the same engine - the way to {@code withCachedPrompt}, {@code
     * saveCachedPrompts} and {@code tokenCountEstimator()} from a model built with {@code
     * buildStreaming()}, which otherwise hands out a streaming face and no way back. ({@code
     * fork()} additionally needs weights YOU loaded - a {@code model(LoadedModel)} build - like on
     * any blocking model.)
     */
    public JinferChatModel blocking() {
        return model;
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
    public Set<Capability> supportedCapabilities() {
        return model.supportedCapabilities();
    }

    @Override
    public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        // the WHOLE preparation is synchronous: every request-shape rejection (unsupported
        // params, media the model cannot frame, remote URLs) throws raw from chat(), unwrapped
        ChatEngine.Prepared p = model.prepare(request);
        model.engine.stream(
                () -> {
                    try {
                        stream(p, handler);
                    } catch (Throwable t) {
                        handler.onError(t);
                    }
                });
    }

    private void stream(ChatEngine.Prepared p, StreamingChatResponseHandler handler) {
        ChatEngine engine = model.engine;
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
        ChatEngine.Completion done =
                engine.complete(
                        p,
                        new ChatEngine.ReplySink() {
                            @Override
                            public void content(String delta) {
                                safely(
                                        handler,
                                        () ->
                                                handler.onPartialResponse(
                                                        new PartialResponse(delta), context));
                            }

                            @Override
                            public void thinking(String delta) {
                                safely(
                                        handler,
                                        () ->
                                                handler.onPartialThinking(
                                                        new PartialThinking(delta)));
                            }

                            @Override
                            public boolean cancelled() {
                                return cancelled.get();
                            }
                        });
        if (done.cancelled()) {
            return; // a cancelled stream ends silently: no complete callback
        }
        AiMessage ai = Mappings.toAiMessage(done.reply());
        if (done.stopped()) {
            ai = Mappings.withText(ai, TextStops.apply(ai.text(), p.stops()).text());
        }
        if (ai.hasToolExecutionRequests()) {
            for (int i = 0; i < ai.toolExecutionRequests().size(); i++) {
                CompleteToolCall call = new CompleteToolCall(i, ai.toolExecutionRequests().get(i));
                safely(handler, () -> handler.onCompleteToolCall(call));
            }
        }
        ChatResponse response = Mappings.response(engine.modelName(), ai, p.promptTokens(), done);
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
