package com.qxotic.jinfer.decision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
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
    void askIndependentlyAgreesWithAskAndALonePackedQuestionIsAsAlone() {
        Decisions decisions = Decisions.of(d1);
        try (ContextState state = d1.newState(1024)) {
            Answers each = decisions.askIndependently(state, TICKET, REFUND, TEAM, URGENCY);
            assertTrue(each.get(REFUND).isYes(), "refund " + each.get(REFUND));
            assertEquals(Team.BILLING, each.get(TEAM).selectedOption());
            assertTrue(each.get(URGENCY).selectedIndex() >= 2, "urgency " + each.get(URGENCY));

            assertEquals(decisions.ask(state, TICKET, REFUND).yes(), each.get(REFUND).yes(), 1e-4f);
            assertEquals(
                    decisions.ask(state, TICKET, TEAM).selectedOption(),
                    each.get(TEAM).selectedOption());
            assertEquals(
                    decisions.ask(state, TICKET, URGENCY).probabilities(),
                    each.get(URGENCY).probabilities());

            // one question in a packed prompt is asked as if alone
            Answers lone = decisions.ask(state, TICKET, new Question<?>[] {REFUND});
            assertEquals(decisions.ask(state, TICKET, REFUND).yes(), lone.get(REFUND).yes(), 1e-4f);
        }
    }

    @Test
    void aReusedStateDoesNotCarryThePreviousInput() {
        Decisions decisions = Decisions.of(d1);
        try (ContextState state = d1.newState(1024)) {
            float first = decisions.ask(state, TICKET, REFUND).yes();
            for (int i = 0; i < 3; i++)
                assertEquals(first, decisions.ask(state, TICKET, REFUND).yes(), 1e-5f);
            decisions.ask(state, "Please tell me where my parcel is.", TEAM);
            assertEquals(first, decisions.ask(state, TICKET, REFUND).yes(), 1e-5f);
        }
    }

    @Test
    void oneDecisionsServesAStatePerThread() throws Exception {
        int threads = 4, repeats = 3;
        float expected;
        Decisions decisions = Decisions.of(d1);
        try (ContextState state = d1.newState(512)) {
            expected = decisions.ask(state, TICKET, REFUND).yes();
        }
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<Float>> futures = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                futures.add(
                        pool.submit(
                                () -> {
                                    try (ContextState state = d1.newState(512)) {
                                        float answer = 0;
                                        for (int i = 0; i < repeats; i++)
                                            answer = decisions.ask(state, TICKET, REFUND).yes();
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
        ContextState state = d1.newState(512);
        state.close();
        state.close(); // idempotent
        assertThrows(
                IllegalStateException.class, () -> Decisions.of(d1).ask(state, TICKET, REFUND));
    }

    @Test
    void theCapacityIsTheModelsRule() {
        assertThrows(IllegalArgumentException.class, () -> d1.newState(200_000));
        assertThrows(IllegalArgumentException.class, () -> d1.newState(-2));
        Decisions small = Decisions.of(d1);
        try (ContextState state = d1.newState(16)) {
            IllegalArgumentException refused =
                    assertThrows(
                            IllegalArgumentException.class, () -> small.ask(state, TICKET, REFUND));
            assertTrue(refused.getMessage().contains("open a larger one"), refused.getMessage());
        }
    }

    /** d1 packs sixteen questions a prompt: past sixteen it packs several prompts. */
    @Test
    void seventeenQuestionsTogetherAreStillAllAnswered() {
        Decisions decisions = Decisions.of(d1);
        try (ContextState state = d1.newState(2048)) {
            List<Question<?>> many = new ArrayList<>();
            for (int i = 0; i < 17; i++) many.add(Question.predicate("Is this a refund request?"));
            Answers answers = decisions.ask(state, TICKET, many.toArray(Question<?>[]::new));
            for (Question<?> q : many) {
                float yes = answers.get(q).probabilities().get(1);
                assertTrue(yes > 0.5f, "question at " + many.indexOf(q) + ": " + yes);
            }
        }
    }

    /** d1-Omni listens through the typed API: a clip, a clip with text, and both with a picture. */
    @Test
    void omniHearsThroughTheTypedApi() throws Exception {
        String omni = "hf.co/LiquidAI/d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf";
        String mmproj = "hf.co/LiquidAI/d1-Omni-2B-GGUF/mmproj-d1-Omni-2B-F16.gguf";
        try (Arena arena = Arena.ofShared()) {
            DecisionModel model =
                    Models.loadDecisionModel(
                            TestModels.require(omni),
                            arena,
                            Map.of("media", TestModels.require(mmproj)));
            var speech = Question.predicate("Is someone speaking in the recording?");
            var holds =
                    Question.choice(
                            "What does the recording hold?",
                            "speech",
                            "music",
                            "a steady tone",
                            "silence");
            Decisions decisions = Decisions.of(model);
            try (ContextState state = model.newState(2048)) {
                Path clip = Path.of("../../test-fixtures/parakeet/ls0.wav");
                List<Content> audioOnly =
                        List.of(new Content.Media(AudioCodec.decode(Files.readAllBytes(clip))));
                List<Content> withText = new ArrayList<>(audioOnly);
                withText.add(new Content.Text("A recording sent by a customer."));

                // a clip alone, and the same clip with a caption: both read, both answer
                Answers alone = decisions.askIndependently(state, audioOnly, speech, holds);
                assertEquals(0.50960f, alone.get(speech).yes(), 2e-2f);
                assertEquals("speech", alone.get(holds).selectedOption());

                Answers captioned = decisions.askIndependently(state, withText, speech, holds);
                assertEquals(0.53227f, captioned.get(speech).yes(), 2e-2f);
                assertEquals(0.74172f, captioned.get(holds).confidence(), 2e-2f);

                // a picture and a clip in one input: both towers, one reading
                List<Content> both = new ArrayList<>();
                both.add(
                        new Content.Media(
                                ImageCodec.load(
                                        Path.of("../../test-fixtures/lfm3/images/cat.png"))));
                both.addAll(audioOnly);
                both.add(new Content.Text("A photo and a recording."));
                Answers mixed = decisions.ask(state, both, speech, holds);
                assertEquals(2, mixed.get(speech).probabilities().size());
                assertEquals(4, mixed.get(holds).probabilities().size());
            }
        }
    }

    @Test
    void noQuestionsAnswerNothing() {
        Decisions decisions = Decisions.of(d1);
        try (ContextState state = d1.newState(512)) {
            Answers empty = decisions.askIndependently(state, TICKET);
            assertThrows(IllegalArgumentException.class, () -> empty.get(REFUND));
            assertThrows(
                    IllegalArgumentException.class, () -> decisions.ask(state, TICKET).get(REFUND));
        }
    }

    @Test
    void aChoiceWithDescriptionsKeepsTheCallerKeys() {
        SequencedMap<String, String> options = new LinkedHashMap<>();
        options.put("billing", "Charges and refunds");
        options.put("shipping", "Delivery problems");
        options.put("other", null);
        Decisions decisions = Decisions.of(d1);
        try (ContextState state = d1.newState(1024)) {
            Answer.Choice<String> answer =
                    decisions.ask(state, TICKET, Question.choice("Which team?", options));
            assertEquals(List.of("billing", "shipping", "other"), answer.options());
            assertEquals("billing", answer.selectedOption());
            assertEquals(3, answer.probabilities().size());
        }
    }
}
