package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.content.Media;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.input.DecisionInput;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;

/**
 * The adapter's shapes: many questions in one call, parts in every order, an empty text part, a
 * part kind a decision cannot judge, a field of empty parts, and one instance serving every shape
 * in sequence. Liquid d1-2B with its vision tower, so {@code integration}.
 */
@Tag("integration")
final class JinferDecisionModelShapesIT {

    static final String D1 = "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf";
    static final String MMPROJ = "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf";
    static final String TICKET =
            "Customer: I was charged twice for my order last week and I want my money back today.";

    static final Arena ARENA = Arena.ofShared();
    static DecisionModel weights;
    static JinferDecisionModel model;

    @BeforeAll
    static void load() throws Exception {
        weights =
                Models.loadDecisionModel(
                        TestModels.require(D1), ARENA, Map.of("media", TestModels.require(MMPROJ)));
        model = JinferDecisionModel.builder().model(weights).build();
    }

    @AfterAll
    static void close() {
        if (model != null) model.close();
        ARENA.close();
    }

    private static Map<String, DecisionQuestion> questions(Object... namesAndQuestions) {
        Map<String, DecisionQuestion> questions = new LinkedHashMap<>();
        for (int i = 0; i < namesAndQuestions.length; i += 2)
            questions.put(
                    (String) namesAndQuestions[i], (DecisionQuestion) namesAndQuestions[i + 1]);
        return questions;
    }

    private static Media png(String name) throws Exception {
        return new Media(
                MimeType.valueOf("image/png"),
                new ByteArrayResource(
                        Files.readAllBytes(TestModels.fixture("lfm3/images/" + name + ".png"))));
    }

    /** Forty questions in one call, each answered, in the prompt's order. */
    @Test
    void manyQuestionsInOneCall() {
        Map<String, DecisionQuestion> many = new LinkedHashMap<>();
        for (int i = 0; i < 40; i++)
            many.put("q" + i, YesNoQuestion.of("Is the customer asking for a refund?"));
        DecisionResponse response = model.call(TICKET, many);
        assertEquals(40, response.getAnswers().size());
        assertEquals(40, response.getResults().size());
        for (int i = 0; i < 40; i++) {
            assertEquals("q" + i, response.getResults().get(i).getName(), "order kept");
            assertTrue(response.getYesNo("q" + i).getProbability() > 0.9, "q" + i);
        }
    }

    /** Parts in every order: the media are read first whatever their place, as the family says. */
    @Test
    void partsInAnyOrderAreRead() throws Exception {
        Media cat = png("cat");
        DecisionResponse afterText =
                model.call(
                        DecisionInput.content(TextPart.of(TICKET), MediaPart.of(cat)),
                        questions(
                                "cat", YesNoQuestion.of("Is there a cat in the picture?"),
                                "shows",
                                        ChoiceQuestion.of(
                                                "What does the picture show?",
                                                "cat",
                                                "dog",
                                                "dice",
                                                "landscape")));
        DecisionResponse beforeText =
                model.call(
                        DecisionInput.content(MediaPart.of(cat), TextPart.of(TICKET)),
                        questions(
                                "cat", YesNoQuestion.of("Is there a cat in the picture?"),
                                "shows",
                                        ChoiceQuestion.of(
                                                "What does the picture show?",
                                                "cat",
                                                "dog",
                                                "dice",
                                                "landscape")));
        // the answer does not depend on where the caller put the picture among the text
        assertEquals(
                afterText.getChoice("shows").getValue(), beforeText.getChoice("shows").getValue());
        // two calls round the Vector API kernels' reductions in an unspecified order
        assertEquals(
                afterText.getYesNo("cat").getProbability(),
                beforeText.getYesNo("cat").getProbability(),
                1e-3);
        assertEquals("cat", afterText.getChoice("shows").getValue());
    }

    /**
     * An empty text part is an empty text, not a missing part: the picture still reaches the model.
     */
    @Test
    void anEmptyTextPartIsAText() throws Exception {
        DecisionResponse response =
                model.call(
                        DecisionInput.content(MediaPart.of(png("dice")), TextPart.of("")),
                        questions(
                                "shows",
                                ChoiceQuestion.of(
                                        "What does the picture show?",
                                        "cat",
                                        "dog",
                                        "dice",
                                        "landscape")));
        assertEquals("dice", response.getChoice("shows").getValue());
    }

    /** A part kind a decision cannot judge is refused in Spring AI's own vocabulary. */
    @Test
    void aReasoningPartIsRefused() {
        UnsupportedOperationException refused =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                model.call(
                                        DecisionInput.content(
                                                TextPart.of(TICKET),
                                                org.springframework.ai.chat.messages.part
                                                        .ReasoningPart.of("thinking")),
                                        questions("q", YesNoQuestion.of("?"))));
        assertTrue(
                refused.getMessage().contains("a decision's input is text and media"),
                refused.getMessage());
    }

    /** A field of empty parts contributes an empty value; the rest of the JSON is intact. */
    @Test
    void aFieldOfEmptyPartsKeepsItsValue() {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("note", List.of(TextPart.of("")));
        fields.put("message", TICKET);
        DecisionResponse response =
                model.call(
                        DecisionInput.fields(fields),
                        questions(
                                "refund",
                                YesNoQuestion.of("Is the customer asking for a refund?")));
        assertTrue(response.getYesNo("refund").isYes());
    }

    /** One instance, every shape in sequence: nothing leaks from one request into the next. */
    @Test
    void everyShapeInSequenceOnOneInstance() throws Exception {
        double first =
                model.call(TICKET, questions("q", YesNoQuestion.of("Refund?")))
                        .getYesNo("q")
                        .getProbability();

        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("message", "Where is my parcel?");
        model.call(
                DecisionInput.fields(fields),
                questions("q", ChoiceQuestion.of("Which team?", "billing", "shipping", "other")));
        model.call(
                DecisionInput.content(MediaPart.of(png("cat")), TextPart.of("A photo.")),
                questions("q", YesNoQuestion.of("Is there a cat in the picture?")));
        model.call(
                DecisionInput.content(MediaPart.of(png("dice"))),
                questions("q", ScoreQuestion.of("How bright is it?", "dark", "dim", "bright")));

        assertEquals(
                first,
                model.call(TICKET, questions("q", YesNoQuestion.of("Refund?")))
                        .getYesNo("q")
                        .getProbability(),
                1e-4,
                "the first shape again reads the same");
    }

    /** The smallest question: a two-option choice. */
    @Test
    void theSmallestChoiceAnswers() {
        DecisionResponse response =
                model.call(
                        TICKET,
                        questions(
                                "c",
                                ChoiceQuestion.builder()
                                        .text("Is this a billing issue?")
                                        .option("yes", "a charge or a refund")
                                        .option("no", "anything else")
                                        .build()));
        assertEquals("yes", response.getChoice("c").getValue());
        assertEquals(2, response.getChoice("c").getProbabilities().size());
    }
}
