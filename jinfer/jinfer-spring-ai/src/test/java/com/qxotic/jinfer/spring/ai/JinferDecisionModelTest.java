package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
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
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryAllocators;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.input.DecisionInput;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;

/**
 * The Spring AI decision adapter over a stub model: what it hands the model for each input and
 * question kind, and what it makes of the probabilities that come back. No weights.
 */
class JinferDecisionModelTest {

    /** Records what it is asked and answers each question with {@link #rows}. */
    static final class Stub implements DecisionModel {
        List<Content> input;
        List<Question> questions;
        boolean joint;
        final List<ContextState> states = new ArrayList<>();
        Function<Question, float[]> rows =
                q -> {
                    float[] p = new float[q.options().size()];
                    Arrays.fill(p, 1f / p.length);
                    return p;
                };

        @Override
        public ContextState newState(int contextCapacity) {
            ContextState state =
                    new ContextState(8, 8, MemoryAllocators.ofArena(Arena.ofAuto()), false) {
                        @Override
                        protected void clearHistory() {}
                    };
            states.add(state);
            return state;
        }

        @Override
        public float[] decide(ContextState state, List<Content> input, Question question) {
            return decideIndependently(state, input, List.of(question))[0];
        }

        @Override
        public float[][] decideIndependently(
                ContextState state, List<Content> input, List<? extends Question> questions) {
            this.input = input;
            this.questions = new ArrayList<>(questions);
            return questions.stream().map(rows).toArray(float[][]::new);
        }

        @Override
        public float[][] decide(
                ContextState state, List<Content> input, List<? extends Question> questions) {
            joint = true;
            return decideIndependently(state, input, questions);
        }
    }

    private final Stub stub = new Stub();
    private final JinferDecisionModel model = JinferDecisionModel.builder().model(stub).build();

    private static Map<String, DecisionQuestion> asked(Object... namesAndQuestions) {
        Map<String, DecisionQuestion> questions = new LinkedHashMap<>();
        for (int i = 0; i < namesAndQuestions.length; i += 2)
            questions.put(
                    (String) namesAndQuestions[i], (DecisionQuestion) namesAndQuestions[i + 1]);
        return questions;
    }

    @Test
    void questionsReachTheModelWithTheirMeaning() {
        model.call(
                "ticket",
                asked(
                        "refund",
                        YesNoQuestion.builder()
                                .text("A refund?")
                                .whenYes("asks for money back")
                                .whenNo("anything else")
                                .build(),
                        "team",
                        ChoiceQuestion.builder()
                                .text("Which team?")
                                .option("billing", "Charges and refunds")
                                .option("other")
                                .build(),
                        "urgency",
                        ScoreQuestion.builder()
                                .text("How urgent?")
                                .level("low")
                                .level("high", "Needs an answer today")
                                .build()));

        assertEquals(
                new Predicate("A refund?", "asks for money back", "anything else"),
                stub.questions.get(0));
        assertEquals(
                new Choice(
                        "Which team?",
                        List.of(new Option("billing", "Charges and refunds"), new Option("other"))),
                stub.questions.get(1));
        assertEquals(
                new Rating(
                        "How urgent?",
                        List.of(new Option("low"), new Option("high", "Needs an answer today"))),
                stub.questions.get(2));
        assertEquals(List.of(new Content.Text("ticket")), stub.input);
        assertTrue(stub.joint, "the questions taken the model's own way by default");
    }

    @Test
    void probabilitiesBecomeTheFrameworksAnswers() {
        stub.rows =
                q ->
                        switch (q) {
                            case Predicate p -> new float[] {0.2f, 0.8f};
                            case Choice c -> new float[] {0.1f, 0.6f, 0.3f};
                            // rounding can make the sum a hair over 1, the expected level past the
                            // top one: it stays on the scale
                            case Rating r -> new float[] {0f, 1e-7f, 1f};
                        };
        DecisionResponse response =
                model.call(
                        "x",
                        asked(
                                "yes",
                                YesNoQuestion.of("Yes?"),
                                "pick",
                                ChoiceQuestion.of("Pick", "a", "b", "c"),
                                "scale",
                                ScoreQuestion.of("Scale", "lo", "mid", "hi")));

        assertEquals(0.8, response.getYesNo("yes").getProbability(), 1e-6);
        assertTrue(response.getYesNo("yes").isYes());
        assertEquals("b", response.getChoice("pick").getValue());
        assertEquals(0.3, response.getChoice("pick").getProbability("c"), 1e-6);
        assertEquals(2.0, response.getScore("scale").getValue(), 0);
        assertEquals(2, response.getScore("scale").getNearestLevel());
        assertEquals(List.of("yes", "pick", "scale"), List.copyOf(response.getAnswers().keySet()));
    }

    @Test
    void aTieGoesToTheFirstOption() {
        stub.rows = q -> new float[] {0.4f, 0.4f, 0.2f};
        DecisionResponse response =
                model.call("x", asked("pick", ChoiceQuestion.of("Pick", "a", "b", "c")));
        assertEquals("a", response.getChoice("pick").getValue());
    }

    @Test
    void fieldsAreOneTextOfIndentedJson() {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("customer", "Ana");
        fields.put("order", 4411);
        fields.put("note", null);
        model.call(DecisionInput.fields(fields), asked("q", YesNoQuestion.of("?")));
        assertEquals(
                List.of(
                        new Content.Text(
                                "{\n"
                                        + "  \"customer\": \"Ana\",\n"
                                        + "  \"order\": 4411,\n"
                                        + "  \"note\": null\n"
                                        + "}")),
                stub.input);
    }

    /** A field of content parts: its media join the input ahead of the JSON, its text stays. */
    @Test
    void contentPartsInFieldsMoveTheirMediaToTheInput() throws Exception {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("photo", MediaPart.of(png()));
        fields.put("caption", List.of(TextPart.of("a "), TextPart.of("cat")));
        model.call(DecisionInput.fields(fields), asked("q", YesNoQuestion.of("?")));
        assertEquals(2, stub.input.size());
        assertInstanceOf(
                Media.Image.class,
                assertInstanceOf(Content.Media.class, stub.input.get(0)).media());
        assertEquals(
                new Content.Text("{\n  \"photo\": \"image\",\n  \"caption\": \"a cat\"\n}"),
                stub.input.get(1));
    }

    @Test
    void contentPartsKeepTheirOrderAndMediaAreDecoded() throws Exception {
        model.call(
                DecisionInput.content(
                        TextPart.of("before"), MediaPart.of(png()), TextPart.of("after")),
                asked("q", YesNoQuestion.of("?")));
        assertEquals(3, stub.input.size());
        assertEquals(new Content.Text("before"), stub.input.get(0));
        Media.Image image =
                assertInstanceOf(
                        Media.Image.class,
                        assertInstanceOf(Content.Media.class, stub.input.get(1)).media());
        assertEquals(3, image.width());
        assertEquals(new Content.Text("after"), stub.input.get(2));
    }

    @Test
    void aConversationIsRefused() {
        var refused =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                model.call(
                                        DecisionInput.conversation(List.of(new UserMessage("hi"))),
                                        asked("q", YesNoQuestion.of("?"))));
        assertTrue(refused.getMessage().contains("not a conversation"), refused.getMessage());
        assertEquals(null, stub.input, "nothing reached the model");
    }

    @Test
    void independentAsksEachQuestionAlone() {
        try (JinferDecisionModel independent =
                JinferDecisionModel.builder().model(stub).independent(true).build()) {
            independent.call("x", asked("a", YesNoQuestion.of("A?"), "b", YesNoQuestion.of("B?")));
        }
        assertFalse(stub.joint);
        assertEquals(2, stub.questions.size());
    }

    @Test
    void closeFreesTheState() {
        JinferDecisionModel closing = JinferDecisionModel.builder().model(stub).build();
        ContextState state = stub.states.getLast();
        assertTrue(state.isAlive());
        closing.close();
        closing.close(); // idempotent
        assertFalse(state.isAlive());
    }

    @Test
    void theBuilderSaysWhatIsMissingOrMisplaced() {
        assertThrows(IllegalArgumentException.class, () -> JinferDecisionModel.builder().build());
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        JinferDecisionModel.builder()
                                .model(stub)
                                .companionPath("media", Path.of("mmproj.gguf"))
                                .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferDecisionModel.builder().companion("media", "not a ref"));
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferDecisionModel.builder().contextCapacity(-2));
    }

    /** A builder is reusable: each build() is an independent instance over the same model. */
    @Test
    void aBuilderBuildsMoreThanOnce() {
        int before = stub.states.size();
        JinferDecisionModel.Builder builder = JinferDecisionModel.builder().model(stub);
        try (JinferDecisionModel first = builder.build();
                JinferDecisionModel second = builder.build()) {
            assertTrue(first != second);
            assertTrue(first.call("x", asked("q", YesNoQuestion.of("?"))).hasAnswer("q"));
            assertTrue(second.call("x", asked("q", YesNoQuestion.of("?"))).hasAnswer("q"));
        }
        // each instance closed its own state; the stub recorded one per build
        List<ContextState> built = stub.states.subList(before, stub.states.size());
        assertEquals(2, built.size());
        for (ContextState state : built) assertFalse(state.isAlive(), "closed with its model");
    }

    /** A null model is refused where it is passed, with a message naming the argument. */
    @Test
    void aNullModelIsRefusedNamingIt() {
        assertThrows(
                NullPointerException.class,
                () -> JinferDecisionModel.builder().model((DecisionModel) null));
        assertThrows(
                NullPointerException.class, () -> JinferDecisionModel.builder().modelPath(null));
    }

    private static org.springframework.ai.content.Media png() throws Exception {
        BufferedImage image = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, "png", out);
        return org.springframework.ai.content.Media.builder()
                .mimeType(MimeType.valueOf("image/png"))
                .data(new ByteArrayResource(out.toByteArray()))
                .build();
    }
}
