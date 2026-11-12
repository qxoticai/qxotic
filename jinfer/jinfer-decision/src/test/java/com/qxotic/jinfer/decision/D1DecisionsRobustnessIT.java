package com.qxotic.jinfer.decision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.SequencedMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The typed {@link Decisions} layer beyond the routing case: the two ways of asking agree, a state
 * is reused and closed safely, several runs share one model, capacity is the model's rule and bad
 * questions fail before any inference. Liquid d1-2B, so {@code integration}.
 */
@Tag("integration")
final class D1DecisionsRobustnessIT {

    static final String TICKET =
            "Customer: I was charged twice for my order last week and I want my money back today."
                    + " This is the third time I am writing.";

    static final Arena ARENA = Arena.ofShared();
    static DecisionModel d1;

    enum Team {
        BILLING,
        SHIPPING,
        TECHNICAL
    }

    static final Question.Predicate REFUND =
            Question.predicate("Is the customer asking for a refund?");
    static final Question.Choice<Team> TEAM =
            Question.choice("Which team should handle this?", Team.class);
    static final Question.Rating<String> URGENCY =
            Question.rating("How urgent is this?", "Can wait", "This week", "Today", "Right now");

    @BeforeAll
    static void load() throws Exception {
        d1 =
                Models.loadDecisionModel(
                        TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf"), ARENA);
    }

    @AfterAll
    static void close() {
        ARENA.close();
    }

    @Test
    void askEachAgreesWithAskAndTogetherLeavesALoneQuestionAsAlone() {
        try (Decisions decisions = Decisions.over(d1, 1024)) {
            Answers each = decisions.askEach(TICKET, REFUND, TEAM, URGENCY);
            assertTrue(each.get(REFUND).isYes(), "refund " + each.get(REFUND));
            assertEquals(Team.BILLING, each.get(TEAM).selectedOption());
            assertTrue(each.get(URGENCY).selectedIndex() >= 2, "urgency " + each.get(URGENCY));

            assertEquals(decisions.ask(TICKET, REFUND).yes(), each.get(REFUND).yes(), 1e-4f);
            assertEquals(
                    decisions.ask(TICKET, TEAM).selectedOption(), each.get(TEAM).selectedOption());
            assertEquals(
                    decisions.ask(TICKET, URGENCY).probabilities(),
                    each.get(URGENCY).probabilities());

            // one question in a packed prompt is asked as if alone
            Answers together = decisions.askTogether(TICKET, REFUND);
            assertEquals(decisions.ask(TICKET, REFUND).yes(), together.get(REFUND).yes(), 1e-4f);
        }
    }

    @Test
    void aReusedStateDoesNotCarryThePreviousInput() {
        try (Decisions decisions = Decisions.over(d1, 1024)) {
            float first = decisions.ask(TICKET, REFUND).yes();
            for (int i = 0; i < 3; i++)
                assertEquals(first, decisions.ask(TICKET, REFUND).yes(), 1e-5f);
            decisions.ask("Please tell me where my parcel is.", TEAM);
            assertEquals(first, decisions.ask(TICKET, REFUND).yes(), 1e-5f);
        }
    }

    @Test
    void separateDecisionsRunSideBySide() throws Exception {
        int threads = 4, repeats = 3;
        float expected;
        try (Decisions decisions = Decisions.over(d1, 512)) {
            expected = decisions.ask(TICKET, REFUND).yes();
        }
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<Float>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                futures.add(
                        pool.submit(
                                () -> {
                                    try (Decisions decisions = Decisions.over(d1, 512)) {
                                        float answer = 0;
                                        for (int i = 0; i < repeats; i++)
                                            answer = decisions.ask(TICKET, REFUND).yes();
                                        return answer;
                                    }
                                }));
            }
            for (Future<Float> future : futures)
                assertEquals(expected, future.get(120, TimeUnit.SECONDS), 1e-4f);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void aClosedStateRefusesFurtherCalls() {
        Decisions decisions = Decisions.over(d1, 512);
        decisions.close();
        decisions.close(); // idempotent
        assertThrows(IllegalStateException.class, () -> decisions.ask(TICKET, REFUND));
    }

    @Test
    void theCapacityIsTheModelsRule() {
        assertThrows(IllegalArgumentException.class, () -> Decisions.over(d1, 200_000));
        assertThrows(IllegalArgumentException.class, () -> Decisions.over(d1, -2));
        try (Decisions small = Decisions.over(d1, 16)) {
            IllegalArgumentException refused =
                    assertThrows(IllegalArgumentException.class, () -> small.ask(TICKET, REFUND));
            assertTrue(refused.getMessage().contains("open a larger one"), refused.getMessage());
        }
    }

    @Test
    void badQuestionsFailBeforeAnyInference() {
        // a question is validated where it is written, before any state or model is touched
        assertThrows(IllegalArgumentException.class, () -> Question.predicate(" "));
        assertThrows(IllegalArgumentException.class, () -> Question.predicate(null));
        assertThrows(IllegalArgumentException.class, () -> Question.choice("Which?", "only"));
        assertThrows(
                IllegalArgumentException.class, () -> Question.choice("Which?", new String[0]));
        assertThrows(IllegalArgumentException.class, () -> Question.rating("How much?", "only"));
        assertThrows(
                IllegalArgumentException.class, () -> Question.rating("How much?", new String[0]));
        assertThrows(
                IllegalArgumentException.class,
                () -> Question.choice(" ", "a", "b")); // a blank question text too
        assertThrows(
                IllegalArgumentException.class,
                () -> Question.choice("Which?", new LinkedHashMap<>()));
    }

    /**
     * "Together" is the model's packed reading: past sixteen questions it packs several prompts.
     */
    @Test
    void seventeenQuestionsTogetherAreStillAllAnswered() {
        try (Decisions decisions = Decisions.over(d1, 2048)) {
            List<Question<?>> many = new ArrayList<>();
            for (int i = 0; i < 17; i++) many.add(Question.predicate("Is this a refund request?"));
            Answers answers = decisions.askTogether(TICKET, many.toArray(Question<?>[]::new));
            for (Question<?> q : many) {
                float yes = answers.get(q).probabilities().get(1);
                assertTrue(yes > 0.5f, "question at " + many.indexOf(q) + ": " + yes);
            }
        }
    }

    /** -1 is the model's maximum capacity, whatever the checkpoint declares. */
    @Test
    void aCapacityOfMinusOneIsTheModelsMaximum() throws Exception {
        try (var state = d1.newState(-1)) {
            assertTrue(state.contextCapacity() > 4096, "the maximum, not the default");
        }
    }

    @Test
    void noQuestionsAnswerNothing() {
        try (Decisions decisions = Decisions.over(d1, 512)) {
            Answers empty = decisions.askEach(TICKET);
            assertThrows(IllegalArgumentException.class, () -> empty.get(REFUND));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> decisions.askTogether(TICKET).get(REFUND));
        }
    }

    @Test
    void aChoiceWithDescriptionsKeepsTheCallerKeys() {
        SequencedMap<String, String> options = new LinkedHashMap<>();
        options.put("billing", "Charges and refunds");
        options.put("shipping", "Delivery problems");
        options.put("other", null);
        try (Decisions decisions = Decisions.over(d1, 1024)) {
            Answer.Choice<String> answer =
                    decisions.ask(TICKET, Question.choice("Which team?", options));
            assertEquals(List.of("billing", "shipping", "other"), answer.options());
            assertEquals("billing", answer.selectedOption());
            assertEquals(3, answer.probabilities().size());
        }
    }
}
