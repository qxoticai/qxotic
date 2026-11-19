package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.model.decision.request.ChoiceQuestion;
import dev.langchain4j.model.decision.request.DecisionRequest;
import dev.langchain4j.model.decision.request.ScaleQuestion;
import dev.langchain4j.model.decision.request.YesNoQuestion;
import dev.langchain4j.model.decision.response.ChoiceAnswer;
import dev.langchain4j.model.decision.response.DecisionResponse;
import dev.langchain4j.model.decision.response.ScaleAnswer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * langchain4j's decision model over Liquid d1-2B with its vision tower. The numbers are llama.cpp's
 * {@code /v1/systemone} on the same files (2026-11-10), the tolerance two engines' Q8_0 kernels.
 */
@Tag("integration")
final class JinferDecisionModelIT {

    static final double TOLERANCE = 2e-2;

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
                        .build();
    }

    @AfterAll
    static void close() {
        if (model != null) model.close();
    }

    /** What the ticket below costs: each question asked alone, and the three packed together. */
    static final int EACH_TOKENS = 203, PACKED_TOKENS = 139;

    static final YesNoQuestion REFUND = YesNoQuestion.from("Is the customer asking for a refund?");
    static final ChoiceQuestion TEAM =
            ChoiceQuestion.builder()
                    .text("Which team should handle this?")
                    .option("billing", "Charges and refunds")
                    .option("shipping", "Delivery problems")
                    .option("technical", "Product does not work")
                    .option("other")
                    .build();
    static final ScaleQuestion URGENCY =
            ScaleQuestion.from(
                    "How urgent is this?", List.of("Can wait", "This week", "Today", "Right now"));

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
                model.decide(
                        DecisionRequest.builder()
                                .inputFields(fields)
                                .question("refund", REFUND)
                                .question("vip", YesNoQuestion.from("Is the customer a VIP?"))
                                .question("team", TEAM)
                                .question("urgency", URGENCY)
                                .build());
        assertEquals(0.98477, response.yesNo("refund").probability(), TOLERANCE);
        assertEquals("d1-2B-Q8_0.gguf", response.modelName());
        assertEquals("d1-2B-Q8_0.gguf", model.modelName());
        assertEquals(0.99808, response.yesNo("vip").probability(), TOLERANCE);
        ChoiceAnswer team = response.choice("team");
        assertEquals("billing", team.value());
        assertEquals(0.95346, team.probabilityOf("billing"), TOLERANCE);
        assertEquals(0.04278, team.probabilityOf("other"), TOLERANCE);
        assertEquals(List.of("billing", "shipping", "technical", "other"), team.options());
        ScaleAnswer urgency = response.scale("urgency");
        assertEquals(0.76559, urgency.probabilities().get(2), TOLERANCE);
        assertEquals(0.19221, urgency.probabilities().get(3), TOLERANCE);
        assertEquals(2.13, urgency.mean(), 0.05);
    }

    @Test
    void aTextIsJudgedEachQuestionAloneOrTogether() {
        String ticket =
                "Customer: I was charged twice for my order last week and I want my money back"
                        + " today. This is the third time I am writing.";
        DecisionRequest request =
                DecisionRequest.builder()
                        .input(ticket)
                        .question("refund", REFUND)
                        .question("team", TEAM)
                        .question("urgency", URGENCY)
                        .build();
        DecisionResponse each = model.decide(request);
        assertEquals(0.99328, each.yesNo("refund").probability(), TOLERANCE);
        assertTrue(each.yesNo("refund").isYes(0.5));
        assertEquals("billing", each.choice("team").value());
        assertEquals(0.60066, each.scale("urgency").probabilities().get(2), TOLERANCE);
        // the ticket read once a question; the same count on every surface (Spring AI, the server)
        assertEquals(EACH_TOKENS, each.tokenUsage().inputTokenCount());
        assertEquals(0, each.tokenUsage().outputTokenCount(), "a decision generates nothing");

        try (JinferDecisionModel together =
                JinferDecisionModel.builder()
                        .modelPath(TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf"))
                        .build()) {
            DecisionResponse packed = together.decide(request);
            assertTrue(packed.yesNo("refund").isYes(0.5));
            assertEquals("billing", packed.choice("team").value());
            assertEquals(PACKED_TOKENS, packed.tokenUsage().inputTokenCount(), "read once");
        }
    }

    @Test
    void aPictureIsJudgedThroughTheTower() throws Exception {
        Path dice = TestModels.fixture("lfm3/images/dice.png");
        String base64 = Base64.getEncoder().encodeToString(Files.readAllBytes(dice));
        ChoiceQuestion shows =
                ChoiceQuestion.from(
                        "What does the picture show?", List.of("cat", "dog", "dice", "landscape"));
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .inputContents(
                                        ImageContent.from(base64, "image/png"),
                                        TextContent.from("A photo sent by a customer."))
                                .question(
                                        "cat", YesNoQuestion.from("Is there a cat in the picture?"))
                                .question("shows", shows)
                                .build());
        assertEquals(0.00525, response.yesNo("cat").probability(), TOLERANCE);
        assertEquals("dice", response.choice("shows").value());
        assertEquals(0.99865, response.choice("shows").probabilityOf("dice"), TOLERANCE);
    }

    /** d1-Omni hears a clip through {@code AudioContent}: llama.cpp's answers for the same clip. */
    @Test
    void aClipIsJudgedByOmni() throws Exception {
        String wav =
                Base64.getEncoder()
                        .encodeToString(Files.readAllBytes(TestModels.fixture("parakeet/ls0.wav")));
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
                    omni.decide(
                            DecisionRequest.builder()
                                    .inputContents(
                                            AudioContent.from(wav, "audio/wav"),
                                            TextContent.from("A recording sent by a customer."))
                                    .question(
                                            "speech",
                                            YesNoQuestion.from(
                                                    "Is someone speaking in the recording?"))
                                    .question(
                                            "holds",
                                            ChoiceQuestion.from(
                                                    "What does the recording hold?",
                                                    List.of(
                                                            "speech",
                                                            "music",
                                                            "a_steady_tone",
                                                            "silence")))
                                    .build());
            assertEquals(0.53227, response.yesNo("speech").probability(), TOLERANCE);
            assertEquals("speech", response.choice("holds").value());
            assertEquals(0.74172, response.choice("holds").probabilityOf("speech"), TOLERANCE);
        }
    }
}
