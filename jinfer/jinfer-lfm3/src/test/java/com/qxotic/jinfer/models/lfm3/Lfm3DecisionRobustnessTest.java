package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.DecisionModel.Choice;
import com.qxotic.jinfer.chat.DecisionModel.Option;
import com.qxotic.jinfer.chat.DecisionModel.Predicate;
import com.qxotic.jinfer.chat.DecisionModel.Question;
import com.qxotic.jinfer.chat.DecisionModel.Rating;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * d1 beyond the llama.cpp parity cases: the quantized checkpoints the repository ships, question
 * groups past the sixteen a prompt holds, a state reused and states side by side, and the questions
 * the engine refuses. Weights, so {@code integration}.
 */
@Tag("integration")
final class Lfm3DecisionRobustnessTest {

    static final String D1 = Lfm3DecisionParityTest.D1;
    static final String STATE = Lfm3DecisionParityTest.STATE;
    static final Predicate REFUND = Lfm3DecisionParityTest.REFUND;
    static final Choice TEAM = Lfm3DecisionParityTest.TEAM;
    static final Rating URGENCY = Lfm3DecisionParityTest.URGENCY;

    /** The d1-2B quants the model repository ships: every kernel shape users can download. */
    static final List<String> QUANTS =
            List.of(
                    D1,
                    "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-F16.gguf",
                    "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-BF16.gguf",
                    "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q4_K_M.gguf",
                    "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q4_0.gguf");

    @Test
    void quantizedCheckpointsAnswerAsQ8Does() throws Exception {
        float[][] reference = null;
        String referenceName = "";
        for (String ref : QUANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d1 = Models.loadDecisionModel(TestModels.require(ref), arena);
                try (ContextState state = d1.newState(1024)) {
                    float[][] rows =
                            d1.decideEach(state, text(STATE), List.of(REFUND, TEAM, URGENCY));
                    for (float[] row : rows) assertRow(row, ref);
                    assertEquals(1, argmax(rows[0]), ref + ": refund");
                    assertEquals(0, argmax(rows[1]), ref + ": billing");
                    assertEquals(2, argmax(rows[2]), ref + ": today");
                    if (reference == null) {
                        reference = rows;
                        referenceName = ref;
                    } else {
                        for (int i = 0; i < rows.length; i++) {
                            assertEquals(
                                    argmax(reference[i]),
                                    argmax(rows[i]),
                                    ref + " answers question " + i + " as " + referenceName);
                            for (int option = 0; option < rows[i].length; option++)
                                assertEquals(
                                        reference[i][option],
                                        rows[i][option],
                                        0.15f,
                                        ref
                                                + " question "
                                                + i
                                                + " option "
                                                + option
                                                + " against "
                                                + referenceName);
                        }
                    }
                }
            }
        }
    }

    /** Group is sixteen questions a prompt: 33 is two packed prompts and one question alone. */
    @Test
    void thirtyThreeQuestionsRunAsThreePrompts() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = Models.loadDecisionModel(TestModels.require(D1), arena);
            try (ContextState state = d1.newState(4096)) {
                List<Question> many = new ArrayList<>();
                for (int i = 0; i < 2 * Lfm3Decision.GROUP + 1; i++) many.add(REFUND);
                float[][] packed = d1.decideTogether(state, text(STATE), many);
                assertEquals(2 * Lfm3Decision.GROUP + 1, packed.length);
                for (float[] row : packed) assertRow(row, "packed");
                // the thirty-third question opens a prompt of its own: the answer asked alone
                assertArrayEquals(
                        d1.decide(state, text(STATE), REFUND),
                        packed[2 * Lfm3Decision.GROUP],
                        1e-4f);
            }
        }
    }

    @Test
    void aReusedStateAnswersTheSameAgain() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = Models.loadDecisionModel(TestModels.require(D1), arena);
            try (ContextState state = d1.newState(1024)) {
                d1.decide(state, text(STATE), TEAM); // warm the kernels: the first call may differ
                float[] first = d1.decide(state, text(STATE), REFUND);
                for (int i = 0; i < 3; i++)
                    assertArrayEquals(first, d1.decide(state, text(STATE), REFUND), 1e-6f);
                // another input in between leaves nothing behind
                d1.decide(
                        state,
                        text("The delivery is late; please check where my parcel is."),
                        TEAM);
                assertArrayEquals(first, d1.decide(state, text(STATE), REFUND), 1e-6f);
            }
        }
    }

    /**
     * One state a thread, all over one model: the concurrency the DecisionModel contract allows.
     */
    @Test
    void separateStatesRunSideBySide() throws Exception {
        int threads = 4, repeats = 3;
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = Models.loadDecisionModel(TestModels.require(D1), arena);
            float[] expected;
            try (ContextState state = d1.newState(1024)) {
                d1.decide(state, text(STATE), TEAM); // warm the kernels
                expected = d1.decide(state, text(STATE), REFUND);
            }
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            try {
                List<Future<float[]>> futures = new ArrayList<>();
                for (int t = 0; t < threads; t++) {
                    futures.add(
                            pool.submit(
                                    () -> {
                                        try (ContextState state = d1.newState(1024)) {
                                            float[] answer = null;
                                            for (int i = 0; i < repeats; i++)
                                                answer = d1.decide(state, text(STATE), REFUND);
                                            return answer;
                                        }
                                    }));
                }
                for (Future<float[]> future : futures)
                    assertArrayEquals(expected, future.get(120, TimeUnit.SECONDS), 1e-6f);
            } finally {
                pool.shutdownNow();
            }
        }
    }

    /** An empty question list answers nothing, and asks the model nothing. */
    @Test
    void noQuestionsAnswerNothing() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = Models.loadDecisionModel(TestModels.require(D1), arena);
            try (ContextState state = d1.newState(256)) {
                assertEquals(0, d1.decideEach(state, text(STATE), List.of()).length);
                assertEquals(0, d1.decideTogether(state, text(STATE), List.of()).length);
            }
        }
    }

    /** Ten levels fit the digits; eleven do not, and the model says why. */
    @Test
    void aRatingHasAtMostTenLevels() throws Exception {
        List<Option> ten = new ArrayList<>();
        List<Option> eleven = new ArrayList<>();
        for (int i = 0; i < 10; i++) ten.add(new Option("level " + i));
        for (int i = 0; i <= 10; i++) eleven.add(new Option("level " + i));
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = Models.loadDecisionModel(TestModels.require(D1), arena);
            try (ContextState state = d1.newState(256)) {
                float[] row = d1.decide(state, text("Pick a level."), new Rating("How much?", ten));
                assertRow(row, "ten levels");
                assertEquals(10, row.length);
                IllegalArgumentException refused =
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        d1.decide(
                                                state,
                                                text("Pick a level."),
                                                new Rating("How much?", eleven)));
                assertTrue(
                        refused.getMessage().contains("at most 10 levels"), refused.getMessage());
            }
        }
    }

    /** The batch splits past 1024 tokens a prompt: each answer is still the one asked alone. */
    @Test
    void longPromptsAnsweredEachOnTheirOwnMatchTheIndividualAsks() throws Exception {
        String license;
        try (var in = getClass().getResourceAsStream("/d1/long-state.txt")) {
            license = new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
        Question isLicense = new Predicate("Is this text a software license?");
        Question kind =
                new Choice(
                        "What kind of document is this?",
                        Option.of("legal_text", "news_article", "recipe", "source_code"));
        Question formality =
                new Rating(
                        "How formal is the writing?",
                        List.of(new Option("Casual"), new Option("Neutral"), new Option("Formal")));
        List<Question> questions = List.of(isLicense, kind, formality);
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = Models.loadDecisionModel(TestModels.require(D1), arena);
            try (ContextState state = d1.newState(1024)) {
                float[][] batched = d1.decideEach(state, text(license), questions);
                for (int i = 0; i < questions.size(); i++)
                    assertArrayEquals(
                            d1.decide(state, text(license), questions.get(i)),
                            batched[i],
                            1e-4f,
                            "question " + i);
            }
        }
    }

    /** Two hundred options: past A..Z and AA..AZ, the fallback pool still finds one each. */
    @Test
    void twoHundredOptionsAreAllToldApart() throws Exception {
        List<Option> many = new ArrayList<>();
        for (int i = 0; i < 200; i++) many.add(new Option("item " + i));
        many.set(157, new Option("the needle"));
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = Models.loadDecisionModel(TestModels.require(D1), arena);
            try (ContextState state = d1.newState(2048)) {
                float[] row =
                        d1.decide(
                                state,
                                text("Pick the needle: item 157 is the one to choose."),
                                new Choice("Which item is the needle?", many));
                assertEquals(200, row.length);
                assertRow(row, "200 options");
                assertEquals(157, argmax(row), "the needle: " + row[157]);
            }
        }
    }

    /** A prompt of exactly the state's capacity fits; one token less does not. */
    @Test
    void theCapacityBoundaryIsOffByNone() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            Lfm3Decision d1 =
                    (Lfm3Decision) Models.loadDecisionModel(TestModels.require(D1), arena);
            int length = d1.prompt(STATE, REFUND).length;
            try (ContextState exact = d1.newState(length)) {
                float[] answer = d1.decide(exact, text(STATE), REFUND);
                assertRow(answer, "exact capacity");
            }
            try (ContextState shortState = d1.newState(length - 1)) {
                IllegalArgumentException refused =
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> d1.decide(shortState, text(STATE), REFUND));
                assertTrue(
                        refused.getMessage().contains("the state holds " + (length - 1)),
                        refused.getMessage());
            }
        }
    }

    /** A packed prompt and a second one, over one picture: the lone question is asked as alone. */
    @Test
    void mediaWithASecondPrompt() throws Exception {
        String mmproj = "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf";
        List<Content> input =
                List.of(
                        new Content.Media(
                                ImageCodec.load(
                                        java.nio.file.Path.of(
                                                "../../test-fixtures/lfm3/images/cat.png"))),
                        new Content.Text("A photo sent by a customer."));
        Question shows =
                new Choice(
                        "What does the picture show?",
                        Option.of("cat", "dog", "dice", "landscape"));
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 =
                    Models.loadDecisionModel(
                            TestModels.require(D1),
                            arena,
                            java.util.Map.of("media", TestModels.require(mmproj)));
            try (ContextState state = d1.newState(4096)) {
                List<Question> many = new ArrayList<>();
                for (int i = 0; i < Lfm3Decision.GROUP + 1; i++) many.add(shows);
                float[][] packed = d1.decideTogether(state, input, many);
                assertEquals(Lfm3Decision.GROUP + 1, packed.length);
                for (float[] row : packed) assertRow(row, "packed with media");
                assertArrayEquals(
                        d1.decide(state, input, shows), packed[Lfm3Decision.GROUP], 1e-4f);
            }
        }
    }

    /** The vision tower side by side over separate states: one image, every answer the same. */
    @Test
    void separateStatesEncodeMediaSideBySide() throws Exception {
        String mmproj = "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf";
        List<Content> input =
                List.of(
                        new Content.Media(
                                ImageCodec.load(
                                        java.nio.file.Path.of(
                                                "../../test-fixtures/lfm3/images/cat.png"))),
                        new Content.Text("A photo sent by a customer."));
        Question cat = new Predicate("Is there a cat in the picture?");
        int threads = 3, repeats = 2;
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 =
                    Models.loadDecisionModel(
                            TestModels.require(D1),
                            arena,
                            java.util.Map.of("media", TestModels.require(mmproj)));
            float[] expected;
            try (ContextState state = d1.newState(4096)) {
                // the first calls may compile to another tier: warm the towers before measuring
                for (int i = 0; i < 3; i++) d1.decide(state, input, cat);
                expected = d1.decide(state, input, cat);
                assertTrue(expected[1] > 0.9f, "the cat: " + expected[1]);
                for (int i = 0; i < 3; i++)
                    assertArrayEquals(expected, d1.decide(state, input, cat), 1e-6f);
            }
            for (int i = 0; i < 3; i++) {
                try (ContextState state = d1.newState(4096)) {
                    assertArrayEquals(expected, d1.decide(state, input, cat), 1e-3f, "state " + i);
                }
            }
            ExecutorService pool = Executors.newFixedThreadPool(threads);
            try {
                List<Future<float[]>> futures = new ArrayList<>();
                for (int t = 0; t < threads; t++) {
                    futures.add(
                            pool.submit(
                                    () -> {
                                        try (ContextState state = d1.newState(4096)) {
                                            float[] answer = null;
                                            for (int i = 0; i < repeats; i++)
                                                answer = d1.decide(state, input, cat);
                                            return answer;
                                        }
                                    }));
                }
                for (Future<float[]> future : futures)
                    assertArrayEquals(expected, future.get(120, TimeUnit.SECONDS), 1e-6f);
            } finally {
                pool.shutdownNow();
            }
        }
    }

    /**
     * What the model advertises is what it reads: the decision takes video when its vision tower is
     * attached, the chat template of the same family does not, and neither does a bare file.
     */
    @Test
    void aDecisionAdvertisesVideoOnlyWithItsTower() throws Exception {
        String mmproj = "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf";
        try (Arena arena = Arena.ofShared()) {
            DecisionModel bare = Models.loadDecisionModel(TestModels.require(D1), arena);
            assertTrue(bare instanceof com.qxotic.jinfer.media.Multimodal);
            com.qxotic.jinfer.media.Multimodal bareMedia =
                    (com.qxotic.jinfer.media.Multimodal) bare;
            assertFalse(bareMedia.supports(com.qxotic.jinfer.media.Media.Video.class));
            assertFalse(bareMedia.supports(com.qxotic.jinfer.media.Media.Image.class));

            DecisionModel withTower =
                    Models.loadDecisionModel(
                            TestModels.require(D1),
                            arena,
                            java.util.Map.of("media", TestModels.require(mmproj)));
            com.qxotic.jinfer.media.Multimodal media =
                    (com.qxotic.jinfer.media.Multimodal) withTower;
            assertTrue(media.supports(com.qxotic.jinfer.media.Media.Video.class));
            assertTrue(media.supports(com.qxotic.jinfer.media.Media.Image.class));
            // d1 has no audio tower; d1-Omni does
            assertFalse(media.supports(com.qxotic.jinfer.media.Media.Audio.class));

            // the same file is not a chat model: it was trained to answer, not to generate
            assertThrows(
                    com.qxotic.jinfer.chat.ModelProvider.IncompatibleModelException.class,
                    () -> Models.load(TestModels.require(D1), arena));
        }
    }

    private static List<Content> text(String text) {
        return List.of(new Content.Text(text));
    }

    private static void assertRow(float[] row, String what) {
        float sum = 0;
        for (float p : row) {
            assertTrue(Float.isFinite(p), what + ": not finite: " + p);
            assertTrue(p >= -1e-6f && p <= 1 + 1e-6f, what + ": outside 0..1: " + p);
            sum += p;
        }
        assertEquals(1f, sum, 1e-3f, what + ": probabilities sum");
    }

    private static int argmax(float[] row) {
        int best = 0;
        for (int i = 1; i < row.length; i++) if (row[i] > row[best]) best = i;
        return best;
    }
}
