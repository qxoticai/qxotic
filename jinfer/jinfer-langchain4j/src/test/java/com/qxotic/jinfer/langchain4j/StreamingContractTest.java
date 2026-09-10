package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/** Callback ordering, cancellation and lifecycle over a fresh weightless model per test. */
class StreamingContractTest {

    private JinferChatModel model;

    @BeforeEach
    void load() {
        model = ChatFixtures.builder().build();
    }

    @AfterEach
    void close() {
        if (model != null) model.close();
    }

    /** One writer; tests read after a callback latch or the stream-driver barrier. */
    private static class Recorder implements StreamingChatResponseHandler {
        final StringBuilder text = new StringBuilder();
        final List<String> events = new ArrayList<>();
        ChatResponse response;
        Throwable error;
        int partials;
        final CountDownLatch done = new CountDownLatch(1);

        final CountDownLatch firstDelta = new CountDownLatch(1);

        @Override
        public void onPartialResponse(PartialResponse partial, PartialResponseContext context) {
            text.append(partial.text());
            events.add("partial");
            partials++;
            firstDelta.countDown();
        }

        @Override
        public void onCompleteResponse(ChatResponse complete) {
            events.add("complete");
            response = complete;
            done.countDown();
        }

        @Override
        public void onError(Throwable t) {
            events.add("error");
            error = t;
            // deliberately NOT counting down: onError is not terminal for a handler fault, and a
            // latch that released here would stop the test watching before the stream finished
        }

        void awaitCompletion() throws InterruptedException {
            assertTrue(done.await(10, TimeUnit.SECONDS), "the stream never completed");
        }
    }

    @Test
    void deltasArriveAndCompleteFiresExactlyOnce() throws Exception {
        Recorder r = new Recorder();
        model.streaming().chat("hello", r);
        r.awaitCompletion();

        assertNull(r.error, "a healthy stream reports no error");
        assertEquals(1, Collections.frequency(r.events, "complete"), r.events.toString());
        assertEquals("complete", r.events.getLast());
        assertFalse(r.text.isEmpty(), "the stream produced no text");
    }

    /**
     * The streamed deltas must reconstruct the same answer the blocking call returns - they are the
     * same generation, so a divergence means the streaming path parses or holds back differently.
     */
    @Test
    void streamedTextMatchesTheBlockingReply() throws Exception {
        String prompt = "hello";
        String blocking = model.chat(prompt);
        Recorder r = new Recorder();
        model.streaming().chat(prompt, r);
        r.awaitCompletion();
        assertEquals(blocking, r.text.toString(), "streamed text diverged from the blocking reply");
    }

    @Test
    void cancellationEndsTheStreamSilently() throws Exception {
        // A barrier behind the cancelled stream proves its callbacks have finished.
        Recorder r =
                new Recorder() {
                    @Override
                    public void onPartialResponse(
                            PartialResponse partial, PartialResponseContext context) {
                        super.onPartialResponse(partial, context);
                        if (partials == 2) context.streamingHandle().cancel();
                    }
                };
        model.streaming().chat("hello", r);
        CountDownLatch drained = new CountDownLatch(1);
        model.engine.stream(drained::countDown);
        assertTrue(drained.await(10, TimeUnit.SECONDS), "cancelled stream did not finish");
        assertTrue(r.partials >= 2, "partials flowed before cancel");
        assertNull(r.response, "cancellation has no completion callback");
        assertNull(r.error, "cancellation has no error callback");
    }

    @Test
    void theTwinsShareOneLifecycle() {
        JinferStreamingChatModel streaming = model.streaming();
        model.close();
        assertThrows(IllegalStateException.class, () -> streaming.chat("hi", new Recorder()));
        try (JinferChatModel other = ChatFixtures.builder().build()) {
            other.streaming().close();
            assertThrows(IllegalStateException.class, () -> other.chat(UserMessage.from("hi")));
        }
    }

    @Test
    void hittingTheContextWallMidStreamKeepsThePartialsAndFinishesLength() throws Exception {
        // Context exhaustion preserves partials and completes with LENGTH, not an error.
        try (JinferChatModel tiny =
                ChatFixtures.builder().contextCapacity(64).maxOutputTokens(128).build()) {
            Recorder r = new Recorder();
            tiny.streaming().chat("hello", r);
            r.awaitCompletion();

            assertNull(r.error, "the wall is not an error");
            assertTrue(r.partials > 10, "partials must flow before the wall");
            assertEquals("complete", r.events.getLast());
            assertEquals(FinishReason.LENGTH, r.response.finishReason());
        }
    }

    @Test
    void startingAStreamDoesNotWaitForTheRunningGeneration() throws Exception {
        // chat() prepares the request on the caller's thread; preparing used to take the
        // generation lock, so a second stream could not even be enqueued until the first reply
        // had finished generating. Hold its callback so the overlap is guaranteed.
        CountDownLatch release = new CountDownLatch(1);
        Recorder first =
                new Recorder() {
                    @Override
                    public void onPartialResponse(
                            PartialResponse partial, PartialResponseContext context) {
                        super.onPartialResponse(partial, context);
                        try {
                            assertTrue(
                                    release.await(10, TimeUnit.SECONDS),
                                    "callback was not released");
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new AssertionError(e);
                        }
                    }
                };
        try (var caller = Executors.newSingleThreadExecutor()) {
            Recorder second = new Recorder();
            try {
                model.streaming().chat("hello", first);
                assertTrue(first.firstDelta.await(10, TimeUnit.SECONDS));
                caller.submit(() -> model.streaming().chat("hello", second))
                        .get(5, TimeUnit.SECONDS);
                assertFalse(first.events.contains("complete"));
            } finally {
                release.countDown();
            }
            first.awaitCompletion();
            second.awaitCompletion();
            assertNull(first.error);
            assertNull(second.error);
        }
    }

    @Test
    void aThrowingHandlerIsReportedWithoutKillingTheStream() throws Exception {
        Recorder r =
                new Recorder() {
                    @Override
                    public void onPartialResponse(
                            PartialResponse partial, PartialResponseContext context) {
                        super.onPartialResponse(partial, context);
                        if (partials == 1) {
                            throw new IllegalStateException("handler blew up");
                        }
                    }
                };
        model.streaming().chat("hello", r);
        r.awaitCompletion();

        assertTrue(r.events.contains("error"), "the handler fault must be reported: " + r.events);
        assertTrue(r.events.contains("complete"), "the generation must still finish: " + r.events);
        assertTrue(
                r.partials > 1,
                "deltas must keep arriving after a handler fault: " + r.events + " " + r.text);
    }
}
