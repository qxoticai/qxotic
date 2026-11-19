package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.TextStops;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.media.Media;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.model.chat.Capability;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.response.ChatModelStreamingEvent;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.PartialThinkingContext;
import dev.langchain4j.model.chat.response.RawStreamingEvent;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.chat.response.StreamingHandle;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.Flow;
import java.util.concurrent.Flow.Publisher;
import java.util.concurrent.SubmissionPublisher;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Streaming twin of {@link JinferChatModel} - built via {@code
 * JinferChatModel.builder()....buildStreaming()}, or from a live model via {@code streaming()} (the
 * GGUF loads once either way): the engine's single lazy driver thread runs the generation and
 * forwards the native {@link ReplyParser}'s two lanes to langchain4j's two streaming channels -
 * content fragments to {@code onPartialResponse}, reasoning fragments to {@code onPartialThinking},
 * each with the cancellation handle; a spoken reply's speech comes a stretch at a time through
 * {@code onUnmappedRawEvent} as the engine's {@code Media.Audio}, and whole on the complete
 * response. Models without a native parser stream raw decoded text. The reactive {@code
 * chat(ChatRequest)} publishes the same callbacks as events.
 *
 * <p>Lifecycle: the twin shares the blocking model's engine - {@code close()} on either closes
 * both, and with them every cached-prompt view (see {@link JinferChatModel}'s Lifecycle paragraph).
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

    private static final Logger LOG = LoggerFactory.getLogger(JinferStreamingChatModel.class);

    /** Where events reach subscribers: the pool CompletableFuture runs async work on. */
    private static final Executor DELIVERY = new CompletableFuture<Void>().defaultExecutor();

    /** Past any context window, so submit never blocks; the buffer grows only as events queue. */
    private static final int MAX_BUFFERED_EVENTS = 1 << 20;

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
     * buildStreaming()}, which otherwise hands out a streaming face and no way back.
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
        List<Tool> offered = model.offeredTools(request);
        try {
            model.engine.stream(
                    () -> {
                        try (p) {
                            stream(p, offered, handler);
                        } catch (Throwable t) {
                            handler.onError(t);
                        }
                    });
        } catch (RuntimeException | Error rejected) {
            p.close();
            throw rejected;
        }
    }

    /**
     * The callbacks as a {@code Flow.Publisher} of events, what AI Services returning a {@code
     * Publisher} or {@code Flux} of events subscribe to. Each subscriber gets its own generation
     * through a {@link SubmissionPublisher}, which keeps the subscriber's demand and delivers off
     * the engine's thread. Its buffer outgrows any reply, so a subscriber that stops requesting
     * holds one buffered reply, never the engine; a cancelled subscription stops the generation at
     * its next streamed token.
     */
    @Override
    public Publisher<ChatModelStreamingEvent> doChat(ChatRequest request) {
        return subscriber -> {
            SubmissionPublisher<ChatModelStreamingEvent> events =
                    new SubmissionPublisher<>(DELIVERY, MAX_BUFFERED_EVENTS);
            events.subscribe(new Guarded(subscriber));
            try {
                doChat(request, new Relay(events));
            } catch (RuntimeException | Error refused) {
                events.closeExceptionally(refused);
            }
        };
    }

    /**
     * A subscriber whose throwing {@code onNext} is cancelled, as langchain4j's publisher contract
     * asks (Reactive Streams rule 2.13): left to it, {@link SubmissionPublisher} would report the
     * exception to {@code onError}, and the listeners with it. The cancel stops the generation.
     */
    private static final class Guarded implements Flow.Subscriber<ChatModelStreamingEvent> {
        private final Flow.Subscriber<? super ChatModelStreamingEvent> downstream;
        private Flow.Subscription subscription;

        Guarded(Flow.Subscriber<? super ChatModelStreamingEvent> downstream) {
            this.downstream = downstream;
        }

        @Override
        public void onSubscribe(Flow.Subscription subscription) {
            this.subscription = subscription;
            downstream.onSubscribe(subscription);
        }

        @Override
        public void onNext(ChatModelStreamingEvent event) {
            try {
                downstream.onNext(event);
            } catch (RuntimeException broken) {
                LOG.warn(
                        "a stream subscriber threw from onNext; its subscription is cancelled",
                        broken);
                subscription.cancel();
            }
        }

        @Override
        public void onError(Throwable error) {
            downstream.onError(error);
        }

        @Override
        public void onComplete() {
            downstream.onComplete();
        }
    }

    /** The handler callbacks submitted as events; a gone subscriber cancels the generation. */
    private record Relay(SubmissionPublisher<ChatModelStreamingEvent> events)
            implements StreamingChatResponseHandler {

        @Override
        public void onPartialResponse(PartialResponse partial, PartialResponseContext context) {
            submit(partial, context.streamingHandle());
        }

        @Override
        public void onPartialThinking(PartialThinking partial, PartialThinkingContext context) {
            submit(partial, context.streamingHandle());
        }

        private void submit(ChatModelStreamingEvent event, StreamingHandle handle) {
            if (events.hasSubscribers()) events.submit(event);
            else handle.cancel();
        }

        @Override
        public void onUnmappedRawEvent(Object raw) {
            events.submit((RawStreamingEvent) () -> raw);
        }

        @Override
        public void onCompleteToolCall(CompleteToolCall call) {
            events.submit(call);
        }

        @Override
        public void onCompleteResponse(ChatResponse response) {
            events.submit(new CompleteResponse(response));
            events.close();
        }

        @Override
        public void onError(Throwable error) {
            events.closeExceptionally(error);
        }
    }

    private void stream(
            ChatEngine.Prepared p, List<Tool> offered, StreamingChatResponseHandler handler) {
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
        PartialThinkingContext thinkingContext = new PartialThinkingContext(handle);
        ChatEngine.Completion done =
                engine.complete(
                        p,
                        new ChatEngine.ReplySink() {
                            @Override
                            public void on(ChatEngine.Delta delta) {
                                if (delta.channel() == Channel.CONTENT) {
                                    safely(
                                            handler,
                                            () ->
                                                    handler.onPartialResponse(
                                                            new PartialResponse(delta.text()),
                                                            context));
                                } else if (delta.channel() == Channel.REASONING) {
                                    safely(
                                            handler,
                                            () ->
                                                    handler.onPartialThinking(
                                                            new PartialThinking(delta.text()),
                                                            thinkingContext));
                                }
                                // other channels are structural: a claimed call span surfaces
                                // nothing until the finished reply's parsed calls
                            }

                            @Override
                            public void onMedia(Media part) {
                                // a stretch of speech as it is decoded, the engine's own clip:
                                // langchain4j has no partial-audio event, this is its door for
                                // a provider's; the words are the content deltas above, the whole
                                // clip is the complete response's audio
                                safely(handler, () -> handler.onUnmappedRawEvent(part));
                            }

                            @Override
                            public boolean cancelled() {
                                return cancelled.get();
                            }
                        });
        if (done.cancelled()) {
            return; // a cancelled stream ends silently: no complete callback
        }
        AiMessage ai =
                Mappings.unwrapStringifiedArguments(Mappings.toAiMessage(done.reply()), offered);
        if (done.stopped()) {
            ai = Mappings.withText(ai, TextStops.apply(ai.text(), p.stops()).text());
        } else {
            model.replies.remember(ai, done.reply().verbatim()); // the reply as delivered
        }
        if (ai.hasToolExecutionRequests()) {
            for (int i = 0; i < ai.toolExecutionRequests().size(); i++) {
                CompleteToolCall call = new CompleteToolCall(i, ai.toolExecutionRequests().get(i));
                safely(handler, () -> handler.onCompleteToolCall(call));
            }
        }
        ChatResponse response =
                Mappings.response(engine.modelName(), ai, done.promptTokens(), done);
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
