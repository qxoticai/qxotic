package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
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
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.input.DecisionInput;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;

/**
 * The adapter's concurrency contract: ONE serial pipeline an instance, many threads on it, and a
 * close that waits out an in-flight call. Liquid d1-2B, so {@code integration}.
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

    private static Map<String, DecisionQuestion> questions() {
        Map<String, DecisionQuestion> questions = new LinkedHashMap<>();
        questions.put("refund", YesNoQuestion.of("Is the customer asking for a refund?"));
        questions.put(
                "team",
                ChoiceQuestion.of("Which team?", "billing", "shipping", "technical", "other"));
        questions.put(
                "urgency",
                ScoreQuestion.of("How urgent?", "Can wait", "This week", "Today", "Right now"));
        return questions;
    }

    /** One instance is one serial pipeline: many threads get the same, correct answer. */
    @Test
    void manyThreadsOnOneInstanceAreSerializedAndAgree() throws Exception {
        try (JinferDecisionModel model = JinferDecisionModel.builder().model(core).build()) {
            DecisionResponse expected = model.call(TICKET, questions());
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
                                            last = model.call(TICKET, questions());
                                        return last;
                                    }));
                start.countDown();
                for (Future<DecisionResponse> future : futures) {
                    DecisionResponse answer = future.get(180, TimeUnit.SECONDS);
                    assertEquals(
                            expected.getYesNo("refund").getProbability(),
                            answer.getYesNo("refund").getProbability(),
                            1e-4);
                    assertEquals(
                            expected.getChoice("team").getValue(),
                            answer.getChoice("team").getValue());
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
            Future<DecisionResponse> call = pool.submit(() -> model.call(TICKET, questions()));
            assertTrue(inside.await(30, TimeUnit.SECONDS), "the call holds the pipeline");
            Future<?> closing = pool.submit(model::close);
            Thread.sleep(100);
            assertFalse(closing.isDone(), "close blocks while a call runs");
            release.countDown();
            assertEquals("billing", call.get(180, TimeUnit.SECONDS).getChoice("team").getValue());
            closing.get(180, TimeUnit.SECONDS);
            assertThrows(IllegalStateException.class, () -> model.call(TICKET, questions()));
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

    /** A failed call leaves the pipeline usable: an unsupported input does not poison the state. */
    @Test
    void aFailedCallLeavesTheModelUsable() {
        try (JinferDecisionModel model = JinferDecisionModel.builder().model(core).build()) {
            model.call(TICKET, questions()); // warm: a cold first call differs by a hair (JIT)
            DecisionResponse first = model.call(TICKET, questions());
            assertThrows(
                    UnsupportedOperationException.class,
                    () ->
                            model.call(
                                    DecisionInput.content(
                                            org.springframework.ai.chat.messages.part.ReasoningPart
                                                    .of("not something to judge")),
                                    questions()));
            DecisionResponse again = model.call(TICKET, questions());
            assertEquals(
                    first.getYesNo("refund").getProbability(),
                    again.getYesNo("refund").getProbability(),
                    1e-4);
        }
    }

    /**
     * The same state serves every shape in one instance: text, fields, media parts, several
     * questions, and each question kind, in sequence, without one shape leaking into the next.
     */
    @Test
    void everyInputShapeInSequenceOnOneInstance() {
        try (JinferDecisionModel model = JinferDecisionModel.builder().model(core).build()) {
            double text = model.call(TICKET, questions()).getYesNo("refund").getProbability();

            // fields
            Map<String, Object> fields = new LinkedHashMap<>();
            fields.put("message", TICKET);
            model.call(DecisionInput.fields(fields), questions());

            // text again: identical to the first call
            assertEquals(
                    text,
                    model.call(TICKET, questions()).getYesNo("refund").getProbability(),
                    1e-4);
        }
    }
}
