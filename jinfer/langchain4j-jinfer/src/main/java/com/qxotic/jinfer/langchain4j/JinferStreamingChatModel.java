package com.qxotic.jinfer.langchain4j;

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
    public java.util.Set<dev.langchain4j.model.chat.Capability> supportedCapabilities() {
        return model.supportedCapabilities();
    }

    @Override
    public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        // the WHOLE preparation is synchronous: every request-shape rejection (unsupported
        // params, media the model cannot frame, remote URLs) throws raw from chat(), unwrapped
        JinferChatModel.Prepared p = JinferChatModel.prepare(model, request);
        Thread.ofVirtual()
                .name("jinfer-stream")
                .start(
                        () -> {
                            try {
                                stream(p, handler);
                            } catch (Throwable t) {
                                handler.onError(t);
                            }
                        });
    }

    private void stream(JinferChatModel.Prepared p, StreamingChatResponseHandler handler) {
        JinferEngine engine = model.engine;
        StopSequences stops = StopSequences.of(p.stops());
        ReplyLanes lanes =
                new ReplyLanes(p.encoded().template(), engine.loaded.tokenizer(), p.parserSeed());

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
                    String fragment = lanes.feed(token);
                    boolean reasoning = lanes.reasoning();
                    if (!fragment.isEmpty()) {
                        if (reasoning) {
                            safely(
                                    handler,
                                    () -> handler.onPartialThinking(new PartialThinking(fragment)));
                        } else {
                            // the matcher holds back a possible stop prefix; emit what's safe
                            String out = stops == null ? fragment : stops.feed(fragment);
                            if (!out.isEmpty()) {
                                safely(
                                        handler,
                                        () ->
                                                handler.onPartialResponse(
                                                        new PartialResponse(out), context));
                            }
                        }
                    }
                    return !cancelled.get() && (stops == null || !stops.hit());
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
        if (stops != null) {
            String tail = stops.flush(); // release the held-back chars (nothing past a cut)
            if (!tail.isEmpty()) {
                safely(
                        handler,
                        () -> handler.onPartialResponse(new PartialResponse(tail), context));
            }
        }
        AiMessage ai = Mappings.toAiMessage(lanes.finish());
        boolean stopHit = stops != null && stops.hit();
        if (stopHit) {
            ai = Mappings.withText(ai, stops.beforeCut());
        }
        if (ai.hasToolExecutionRequests()) {
            for (int i = 0; i < ai.toolExecutionRequests().size(); i++) {
                CompleteToolCall call = new CompleteToolCall(i, ai.toolExecutionRequests().get(i));
                safely(handler, () -> handler.onCompleteToolCall(call));
            }
        }
        ChatResponse response =
                Mappings.response(engine.modelName, ai, p.promptTokens(), result, stopHit);
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
