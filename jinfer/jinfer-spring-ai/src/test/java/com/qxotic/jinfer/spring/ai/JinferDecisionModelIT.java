package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import io.micrometer.observation.tck.TestObservationRegistry;
import io.micrometer.observation.tck.TestObservationRegistryAssert;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.metadata.Usage;
import org.springframework.ai.content.Media;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.answer.ScoreAnswer;
import org.springframework.ai.decision.input.DecisionInput;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;

/**
 * Spring AI's decision model over Liquid d1-2B with its vision tower. The numbers are llama.cpp's
 * {@code /v1/systemone} on the same files (2026-11-10), the tolerance two engines' Q8_0 kernels.
 */
@Tag("integration")
final class JinferDecisionModelIT {

    static final double TOLERANCE = 2e-2;

    static final TestObservationRegistry observations = TestObservationRegistry.create();
    static JinferDecisionModel model;

    @BeforeAll
    static void load() {
        model =
                JinferDecisionModel.builder()
                        .modelPath(TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf"))
                        .companionPath(
                                "media",
                                TestModels.require(
                                        "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf"))
                        .independent(true) // the pinned numbers are each question asked alone
                        .observationRegistry(observations)
                        .build();
    }

    @AfterAll
    static void close() {
        if (model != null) model.close();
    }

    /** What the ticket below costs: each question asked alone, and the three packed together. */
    static final int EACH_TOKENS = 203, PACKED_TOKENS = 139;

    static final YesNoQuestion REFUND = YesNoQuestion.of("Is the customer asking for a refund?");
    static final ChoiceQuestion TEAM =
            ChoiceQuestion.builder()
                    .text("Which team should handle this?")
                    .option("billing", "Charges and refunds")
                    .option("shipping", "Delivery problems")
                    .option("technical", "Product does not work")
                    .option("other")
                    .build();
    static final ScoreQuestion URGENCY =
            ScoreQuestion.of("How urgent is this?", "Can wait", "This week", "Today", "Right now");

    private static Map<String, DecisionQuestion> questions(Object... namesAndQuestions) {
        Map<String, DecisionQuestion> questions = new LinkedHashMap<>();
        for (int i = 0; i < namesAndQuestions.length; i += 2)
            questions.put(
                    (String) namesAndQuestions[i], (DecisionQuestion) namesAndQuestions[i + 1]);
        return questions;
    }

    /** The observation names the model that answered, as chat's does: d1 is told from d3. */
    @Test
    void theObservationNamesTheModel() {
        observations.clear();
        model.call("I was charged twice.", questions("refund", REFUND));
        TestObservationRegistryAssert.assertThat(observations)
                .hasAnObservationWithAKeyValue("gen_ai.request.model", "d1-2B-Q8_0.gguf");
        TestObservationRegistryAssert.assertThat(observations)
                .hasAnObservationWithAKeyValue("gen_ai.response.model", "d1-2B-Q8_0.gguf");
    }

    /** A map is put to the model as JSON, a member a line: llama.cpp's reading of the same map. */
    @Test
    void fieldsAreJudgedAsJson() {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("customer", "Ana");
        fields.put("order", 4411);
        fields.put("total", 42.5);
        fields.put("vip", true);
        fields.put("note", null);
        fields.put("tags", List.of("billing", "repeat"));
        fields.put("message", "I was charged twice for my order and I want my money back today.");
        DecisionResponse response =
                model.call(
                        DecisionInput.fields(fields),
                        questions(
                                "refund",
                                REFUND,
                                "vip",
                                YesNoQuestion.of("Is the customer a VIP?"),
                                "team",
                                TEAM,
                                "urgency",
                                URGENCY));
        assertEquals(0.98477, response.getYesNo("refund").getProbability(), TOLERANCE);
        assertEquals(0.99808, response.getYesNo("vip").getProbability(), TOLERANCE);
        ChoiceAnswer team = response.getChoice("team");
        assertEquals("billing", team.getValue());
        assertEquals(0.95346, team.getProbability("billing"), TOLERANCE);
        assertEquals(0.04278, team.getProbability("other"), TOLERANCE);
        ScoreAnswer urgency = response.getScore("urgency");
        assertEquals(0.76559, urgency.getProbabilities().get(2), TOLERANCE);
        assertEquals(2, urgency.getNearestLevel());
        assertEquals(2.13, urgency.getValue(), 0.05);
    }

    @Test
    void aTextIsJudgedEachQuestionAloneOrTogether() {
        String ticket =
                "Customer: I was charged twice for my order last week and I want my money back"
                        + " today. This is the third time I am writing.";
        Map<String, DecisionQuestion> three =
                questions("refund", REFUND, "team", TEAM, "urgency", URGENCY);
        DecisionResponse each = model.call(ticket, three);
        assertEquals(0.99328, each.getYesNo("refund").getProbability(), TOLERANCE);
        assertTrue(each.getYesNo("refund").isYes());
        assertEquals("billing", each.getChoice("team").getValue());
        assertEquals(0.60066, each.getScore("urgency").getProbabilities().get(2), TOLERANCE);
        // the ticket read once a question; the same count on every surface (LangChain4j, the
        // server)
        Usage usage = each.getMetadata().getUsage();
        assertEquals(EACH_TOKENS, usage.getPromptTokens());
        assertEquals(0, usage.getCompletionTokens(), "a decision generates nothing");

        try (JinferDecisionModel together =
                JinferDecisionModel.builder()
                        .modelPath(TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf"))
                        .build()) {
            DecisionResponse packed = together.call(ticket, three);
            assertTrue(packed.getYesNo("refund").isYes());
            assertEquals("billing", packed.getChoice("team").getValue());
            assertEquals(
                    PACKED_TOKENS, packed.getMetadata().getUsage().getPromptTokens(), "read once");
        }
    }

    @Test
    void aPictureIsJudgedThroughTheTower() throws Exception {
        Media dice =
                Media.builder()
                        .mimeType(MimeType.valueOf("image/png"))
                        .data(
                                new ByteArrayResource(
                                        Files.readAllBytes(
                                                TestModels.fixture("lfm3/images/dice.png"))))
                        .build();
        Map<String, DecisionQuestion> two =
                questions(
                        "cat",
                        YesNoQuestion.of("Is there a cat in the picture?"),
                        "shows",
                        ChoiceQuestion.of(
                                "What does the picture show?", "cat", "dog", "dice", "landscape"));
        DecisionResponse response =
                model.call(
                        DecisionInput.content(
                                MediaPart.of(dice), TextPart.of("A photo sent by a customer.")),
                        two);
        assertEquals(0.00525, response.getYesNo("cat").getProbability(), TOLERANCE);
        assertEquals("dice", response.getChoice("shows").getValue());
        assertEquals(0.99865, response.getChoice("shows").getProbability("dice"), TOLERANCE);

        // a field of content parts: its media ahead, the field named by their kind
        DecisionResponse inField =
                model.call(DecisionInput.fields(Map.of("photo", MediaPart.of(dice))), two);
        assertEquals("dice", inField.getChoice("shows").getValue());
    }

    /** d1-Omni hears a clip through an audio {@code MediaPart}: llama.cpp's answers for it. */
    @Test
    void aClipIsJudgedByOmni() throws Exception {
        Media clip =
                Media.builder()
                        .mimeType(MimeType.valueOf("audio/wav"))
                        .data(
                                new ByteArrayResource(
                                        Files.readAllBytes(TestModels.fixture("parakeet/ls0.wav"))))
                        .build();
        try (JinferDecisionModel omni =
                JinferDecisionModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/LiquidAI/d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf"))
                        .companionPath(
                                "media",
                                TestModels.require(
                                        "hf.co/LiquidAI/d1-Omni-2B-GGUF/mmproj-d1-Omni-2B-F16.gguf"))
                        .independent(true) // the pinned numbers are each question alone
                        .build()) {
            DecisionResponse response =
                    omni.call(
                            DecisionInput.content(
                                    MediaPart.of(clip),
                                    TextPart.of("A recording sent by a customer.")),
                            questions(
                                    "speech",
                                    YesNoQuestion.of("Is someone speaking in the recording?"),
                                    "holds",
                                    ChoiceQuestion.of(
                                            "What does the recording hold?",
                                            "speech",
                                            "music",
                                            "a_steady_tone",
                                            "silence")));
            assertEquals(0.53227, response.getYesNo("speech").getProbability(), TOLERANCE);
            assertEquals("speech", response.getChoice("holds").getValue());
            assertEquals(0.74172, response.getChoice("holds").getProbability("speech"), TOLERANCE);
        }
    }

    /** A conversation is read whole: the same last words mean what the turns before make them. */
    @Test
    void aConversationIsReadWhole() {
        Function<String, Double> refund =
                offer ->
                        model.call(
                                        DecisionInput.conversation(
                                                List.of(
                                                        new UserMessage(
                                                                "I was charged twice for order"
                                                                        + " A-17."),
                                                        AssistantMessage.builder()
                                                                .content(offer)
                                                                .build(),
                                                        new UserMessage("Yes please."))),
                                        questions("refund", REFUND))
                                .getYesNo("refund")
                                .getProbability();
        double asked = refund.apply("Sorry about that! Shall I refund the duplicate charge?");
        double declined = refund.apply("Sorry about that! Would you like our newsletter instead?");
        assertTrue(asked > 0.5, "the offer accepted is a refund: " + asked);
        assertTrue(declined < asked, "a newsletter accepted is less of one: " + declined);
    }
}
