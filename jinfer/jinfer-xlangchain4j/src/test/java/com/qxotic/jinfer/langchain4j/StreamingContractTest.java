package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialResponse;
import dev.langchain4j.model.chat.response.PartialResponseContext;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * Streaming had NO default coverage: the TCK that exercises it is opt-in, so a normal build never
 * ran a single delta. It is also the most failure-prone adapter here - callback ordering,
 * cancellation, and what happens when the caller's own handler throws are all easy to get subtly
 * wrong and invisible until someone integrates.
 *
 * <p>One model load, a handful of tokens per test.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class StreamingContractTest {

    private static JinferChatModel model;

    @BeforeAll
    void load() {
        Path gguf =
                TestModels.require(
                        "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf");
        model = JinferChatModel.builder().modelPath(gguf).maxOutputTokens(12).seed(7L).build();
    }

    @AfterAll
    void close() {
        if (model != null) model.close();
    }

    /** Collects everything a handler can be told, so ordering can be asserted afterwards. */
    private static class Recorder implements StreamingChatResponseHandler {
        final StringBuilder text = new StringBuilder();
        final List<String> events = new CopyOnWriteArrayList<>();
        final AtomicReference<ChatResponse> response = new AtomicReference<>();
        final AtomicReference<Throwable> error = new AtomicReference<>();
        final CountDownLatch done = new CountDownLatch(1);

        @Override
        public void onPartialResponse(PartialResponse partial, PartialResponseContext context) {
            text.append(partial.text());
            events.add("partial");
        }

        @Override
        public void onCompleteResponse(ChatResponse complete) {
            events.add("complete");
            response.set(complete);
            done.countDown();
        }

        @Override
        public void onError(Throwable t) {
            events.add("error");
            error.set(t);
            // deliberately NOT counting down: onError is not terminal for a handler fault, and a
            // latch that released here would stop the test watching before the stream finished
        }

        void awaitCompletion() throws InterruptedException {
            assertTrue(done.await(120, TimeUnit.SECONDS), "the stream never completed: " + events);
        }
    }

    @Test
    void deltasArriveAndCompleteFiresExactlyOnce() throws Exception {
        Recorder r = new Recorder();
        model.streaming().chat("Name one colour.", r);
        r.awaitCompletion();

        assertEquals(null, r.error.get(), "a healthy stream reports no error");
        assertEquals(
                1,
                r.events.stream().filter("complete"::equals).count(),
                "onCompleteResponse must fire exactly once: " + r.events);
        assertTrue(r.events.indexOf("complete") == r.events.size() - 1, "complete must come last");
        assertTrue(!r.text.isEmpty(), "the stream produced no text");
    }

    /**
     * The streamed deltas must reconstruct the same answer the blocking call returns - they are the
     * same generation, so a divergence means the streaming path parses or holds back differently.
     */
    @Test
    void streamedTextMatchesTheBlockingReply() throws Exception {
        String prompt = "Name one colour.";
        model.chat(prompt); // warm: interpreted and compiled kernels differ by ~1 LSB

        String blocking = model.chat(prompt);
        Recorder r = new Recorder();
        model.streaming().chat(prompt, r);
        r.awaitCompletion();
        assertEquals(blocking, r.text.toString(), "streamed text diverged from the blocking reply");
    }

    /**
     * A caller's handler throwing is a bug in THEIR code, not a reason to lose the generation: it
     * is reported to onError and the stream carries on to completion. Pinning it because the
     * alternative - aborting mid-generation on a transient handler fault - is a tempting change
     * that would silently drop work.
     */
    @Test
    void aThrowingHandlerIsReportedWithoutKillingTheStream() throws Exception {
        AtomicInteger deltas = new AtomicInteger();
        Recorder r =
                new Recorder() {
                    @Override
                    public void onPartialResponse(
                            PartialResponse partial, PartialResponseContext context) {
                        super.onPartialResponse(partial, context);
                        if (deltas.incrementAndGet() == 1) {
                            throw new IllegalStateException("handler blew up");
                        }
                    }
                };
        model.streaming().chat("Name one colour.", r);
        r.awaitCompletion();

        assertTrue(r.events.contains("error"), "the handler fault must be reported: " + r.events);
        assertTrue(r.events.contains("complete"), "the generation must still finish: " + r.events);
        assertTrue(deltas.get() > 1, "deltas must keep arriving after a handler fault");
    }
}
