package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.decision.listener.DecisionModelErrorContext;
import dev.langchain4j.model.decision.listener.DecisionModelListener;
import dev.langchain4j.model.decision.listener.DecisionModelRequestContext;
import dev.langchain4j.model.decision.listener.DecisionModelResponseContext;
import dev.langchain4j.model.decision.request.ChoiceQuestion;
import dev.langchain4j.model.decision.request.DecisionRequest;
import dev.langchain4j.model.decision.request.ScaleQuestion;
import dev.langchain4j.model.decision.request.YesNoQuestion;
import dev.langchain4j.model.decision.response.DecisionResponse;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The langchain4j adapter's concurrency contract - ONE serial pipeline an instance, a close that
 * waits out an in-flight call, listeners that throw - and the resilience of the pipeline after a
 * failure. Liquid d1-2B, so {@code integration}.
 */
@Tag("integration")
final class JinferDecisionModelConcurrencyIT {

    static final String D1 = "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf";
    static final String TICKET =
            "Customer: I was charged twice for my order last week and I want my money back today."
                    + " This is the third time I am writing.";

    static final Arena ARENA = Arena.ofShared();
    static DecisionModel core;

    @BeforeAll
    static void load() throws Exception {
        core = Models.loadDecisionModel(TestModels.require(D1), ARENA);
    }

    @AfterAll
    static void close() {
        ARENA.close();
    }

    private static DecisionRequest request() {
        return DecisionRequest.builder()
                .input(TICKET)
                .question("refund", YesNoQuestion.from("Is the customer asking for a refund?"))
                .question(
                        "team",
                        ChoiceQuestion.from(
                                "Which team?",
                                List.of("billing", "shipping", "technical", "other")))
                .question(
                        "urgency",
                        ScaleQuestion.from(
                                "How urgent?",
                                List.of("Can wait", "This week", "Today", "Right now")))
                .build();
    }

    @Test
    void manyThreadsOnOneInstanceAreSerializedAndAgree() throws Exception {
        try (JinferDecisionModel model = JinferDecisionModel.builder().model(core).build()) {
            DecisionResponse expected = model.decide(request());
            int threads = 6, calls = 3;
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            CountDownLatch start = new CountDownLatch(1);
            try {
                List<Future<DecisionResponse>> futures = new ArrayList<>();
                for (int t = 0; t < threads; t++)
                    futures.add(
                            pool.submit(
                                    () -> {
                                        start.await();
                                        DecisionResponse last = null;
                                        for (int i = 0; i < calls; i++)
                                            last = model.decide(request());
                                        return last;
                                    }));
                start.countDown();
                for (Future<DecisionResponse> future : futures) {
                    DecisionResponse answer = future.get(180, TimeUnit.SECONDS);
                    assertEquals(
                            expected.yesNo("refund").probability(),
                            answer.yesNo("refund").probability(),
                            1e-4);
                    assertEquals(expected.choice("team").value(), answer.choice("team").value());
                }
            } finally {
                pool.shutdownNow();
            }
        }
    }

    /**
     * close() waits out a call that has entered the pipeline, and that call still answers; a call
     * that has not started fails. The wrapper blocks INSIDE the adapter's lock, so the test knows
     * the call holds the pipeline when it closes: no timing luck.
     */
    @Test
    void closeWaitsOutAnInFlightCall() throws Exception {
        CountDownLatch inside = new CountDownLatch(1), release = new CountDownLatch(1);
        DecisionModel slow =
                new DecisionModel() {
                    @Override
                    public ContextState newState(int capacity) {
                        return core.newState(capacity);
                    }

                    @Override
                    public float[] decide(
                            ContextState state, List<Content> input, Question question) {
                        return core.decide(state, input, question);
                    }

                    @Override
                    public float[][] decide(
                            ContextState state,
                            List<Content> input,
                            List<? extends Question> questions) {
                        inside.countDown();
                        await(release);
                        return core.decide(state, input, questions);
                    }
                };
        JinferDecisionModel model = JinferDecisionModel.builder().model(slow).build();
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Future<DecisionResponse> call = pool.submit(() -> model.decide(request()));
            assertTrue(inside.await(30, TimeUnit.SECONDS), "the call holds the pipeline");
            Future<?> closing = pool.submit(model::close);
            Thread.sleep(100);
            assertFalse(closing.isDone(), "close blocks while a call runs");
            release.countDown();
            assertEquals("billing", call.get(180, TimeUnit.SECONDS).choice("team").value());
            closing.get(180, TimeUnit.SECONDS);
            assertThrows(IllegalStateException.class, () -> model.decide(request()));
        } finally {
            pool.shutdownNow();
        }
    }

    private static void await(CountDownLatch latch) {
        try {
            assertTrue(latch.await(60, TimeUnit.SECONDS));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    /**
     * A listener that throws is the framework's to log and ignore: the caller still gets its
     * answer.
     */
    @Test
    void aThrowingListenerDoesNotBreakTheCall() {
        DecisionModelListener broken =
                new DecisionModelListener() {
                    @Override
                    public void onRequest(DecisionModelRequestContext context) {
                        throw new IllegalStateException("listener failure");
                    }

                    @Override
                    public void onResponse(DecisionModelResponseContext context) {
                        throw new IllegalStateException("listener failure");
                    }
                };
        AtomicInteger seen = new AtomicInteger();
        DecisionModelListener counting =
                new DecisionModelListener() {
                    @Override
                    public void onRequest(DecisionModelRequestContext context) {
                        seen.incrementAndGet();
                    }
                };
        try (JinferDecisionModel model =
                JinferDecisionModel.builder()
                        .model(core)
                        .listeners(List.of(broken, counting))
                        .build()) {
            DecisionResponse response = model.decide(request());
            assertEquals("billing", response.choice("team").value());
            assertEquals(1, seen.get(), "the listener after the throwing one still runs");
        }
    }

    /** A failed request reaches onError with the failure, and the pipeline stays usable. */
    @Test
    void aFailedRequestReachesOnErrorAndLeavesTheModelUsable() {
        AtomicInteger errors = new AtomicInteger();
        DecisionModelListener listener =
                new DecisionModelListener() {
                    @Override
                    public void onError(DecisionModelErrorContext context) {
                        errors.incrementAndGet();
                        assertTrue(context.error() instanceof UnsupportedFeatureException);
                    }
                };
        try (JinferDecisionModel model =
                JinferDecisionModel.builder().model(core).listeners(List.of(listener)).build()) {
            DecisionResponse first = model.decide(request());
            assertThrows(
                    UnsupportedFeatureException.class,
                    () ->
                            model.decide(
                                    DecisionRequest.builder()
                                            .inputMessages(UserMessage.from("hi"))
                                            .question("q", YesNoQuestion.from("?"))
                                            .build()));
            assertEquals(1, errors.get());
            DecisionResponse again = model.decide(request());
            assertEquals(
                    first.yesNo("refund").probability(), again.yesNo("refund").probability(), 1e-4);
        }
    }

    /** Every input shape in one instance, in sequence: no shape leaks into the next. */
    @Test
    void everyInputShapeInSequenceOnOneInstance() {
        try (JinferDecisionModel model = JinferDecisionModel.builder().model(core).build()) {
            double text = model.decide(request()).yesNo("refund").probability();

            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put("message", TICKET);
            model.decide(
                    DecisionRequest.builder()
                            .inputFields(fields)
                            .question("refund", YesNoQuestion.from("Is this a refund request?"))
                            .build());

            assertEquals(text, model.decide(request()).yesNo("refund").probability(), 1e-4);
        }
    }
}
