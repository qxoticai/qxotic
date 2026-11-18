package com.qxotic.jinfer.langchain4j;

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
import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryAllocators;
import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ContentType;
import dev.langchain4j.data.message.CustomMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.PdfFileContent;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.internal.DecisionModelInputUtils;
import dev.langchain4j.model.decision.request.ChoiceQuestion;
import dev.langchain4j.model.decision.request.DecisionRequest;
import dev.langchain4j.model.decision.request.ScaleQuestion;
import dev.langchain4j.model.decision.request.YesNoQuestion;
import dev.langchain4j.model.decision.response.DecisionResponse;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

/**
 * The langchain4j decision adapter over a stub model: what it hands the model for each input and
 * question kind, and what it makes of the probabilities that come back. No weights.
 */
class JinferDecisionModelTest {

    /** Records what it is asked and answers each question with {@link #rows}. */
    static final class Stub implements DecisionModel {
        List<Content> input;
        List<Question> questions;
        boolean joint;
        final List<ContextState> states = new ArrayList<>();
        Function<Question, float[]> rows = q -> uniform(q.options().size());

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

    static float[] uniform(int n) {
        float[] p = new float[n];
        Arrays.fill(p, 1f / n);
        return p;
    }

    private final Stub stub = new Stub();
    private final JinferDecisionModel model = JinferDecisionModel.builder().model(stub).build();

    @Test
    void questionsReachTheModelWithTheirMeaning() {
        model.decide(
                DecisionRequest.builder()
                        .input("ticket")
                        .question(
                                "refund",
                                YesNoQuestion.builder()
                                        .text("A refund?")
                                        .yesWhen("asks for money back")
                                        .noWhen("anything else")
                                        .build())
                        .question(
                                "team",
                                ChoiceQuestion.builder()
                                        .text("Which team?")
                                        .option("billing", "Charges and refunds")
                                        .option("other")
                                        .build())
                        .question(
                                "urgency",
                                ScaleQuestion.builder()
                                        .text("How urgent?")
                                        .level("low")
                                        .level("high", "Needs an answer today")
                                        .build())
                        .build());

        assertEquals(
                new Predicate("A refund?", "asks for money back", "anything else"),
                stub.questions.get(0));
        // an option added without a description carries its name as one: the model sees none
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
                            // rounding can make the sum a hair over 1, the mean past the top
                            // level: it stays on the scale
                            case Rating r -> new float[] {0f, 1e-7f, 1f};
                        };
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .input("x")
                                .question("yes", YesNoQuestion.from("Yes?"))
                                .question(
                                        "pick", ChoiceQuestion.from("Pick", List.of("a", "b", "c")))
                                .question(
                                        "scale",
                                        ScaleQuestion.from("Scale", List.of("lo", "mid", "hi")))
                                .build());

        assertEquals(0.8, response.yesNo("yes").probability(), 1e-6);
        assertEquals("b", response.choice("pick").value());
        assertEquals(0.3, response.choice("pick").probabilityOf("c"), 1e-6);
        assertEquals(List.of("a", "b", "c"), response.choice("pick").options());
        assertEquals(2.0, response.scale("scale").mean(), 0);
        assertEquals(3, response.scale("scale").probabilities().size());
        // jinfer's confidence, as its server reports it: the likeliest outcome's probability
        assertEquals(0.6, response.choice("pick").confidence(), 1e-6);
        assertEquals(1.0, response.scale("scale").confidence(), 1e-6);
    }

    @Test
    void aTieGoesToTheFirstOption() {
        stub.rows = q -> new float[] {0.4f, 0.4f, 0.2f};
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .input("x")
                                .question(
                                        "pick", ChoiceQuestion.from("Pick", List.of("a", "b", "c")))
                                .build());
        assertEquals("a", response.choice("pick").value());
    }

    @Test
    void fieldsAreOneTextOfIndentedJson() {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("customer", "Ana");
        fields.put("order", 4411);
        fields.put("note", null);
        model.decide(
                DecisionRequest.builder()
                        .inputFields(fields)
                        .question("q", YesNoQuestion.from("?"))
                        .build());
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

    /** A field of contents: its media join the input ahead of the JSON, its text stays a value. */
    @Test
    void contentsInFieldsMoveTheirMediaToTheInput() throws Exception {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("photo", ImageContent.from(png(), "image/png"));
        fields.put("caption", List.of(TextContent.from("a "), TextContent.from("cat")));
        model.decide(
                DecisionRequest.builder()
                        .inputFields(fields)
                        .question("q", YesNoQuestion.from("?"))
                        .build());
        assertEquals(2, stub.input.size());
        Media.Image image =
                assertInstanceOf(
                        Media.Image.class,
                        assertInstanceOf(Content.Media.class, stub.input.get(0)).media());
        assertEquals(3, image.width());
        assertEquals(
                new Content.Text("{\n  \"photo\": \"image\",\n  \"caption\": \"a cat\"\n}"),
                stub.input.get(1));
    }

    @Test
    void contentsKeepTheirOrderAndMediaAreDecoded() throws Exception {
        model.decide(
                DecisionRequest.builder()
                        .inputContents(
                                TextContent.from("before"),
                                ImageContent.from(png(), "image/png"),
                                TextContent.from("after"))
                        .question("q", YesNoQuestion.from("?"))
                        .build());
        assertEquals(3, stub.input.size());
        assertEquals(new Content.Text("before"), stub.input.get(0));
        Media.Image image =
                assertInstanceOf(
                        Media.Image.class,
                        assertInstanceOf(Content.Media.class, stub.input.get(1)).media());
        assertEquals(3, image.width());
        assertEquals(new Content.Text("after"), stub.input.get(2));
    }

    /**
     * A conversation reaches the model as the transcript langchain4j's own decision models send
     * (Ollama's, Gemini's), its media ahead. Core's builder of it is internal, so the adapter owns
     * its own and core's is the oracle here: an upstream change to the shape fails this test.
     */
    @Test
    void aConversationIsItsTranscriptWithItsMediaAhead() throws Exception {
        ToolExecutionRequest call =
                ToolExecutionRequest.builder()
                        .id("call-1")
                        .name("lookupOrder")
                        .arguments("{\"order\": \"A-17\"}")
                        .build();
        List<ChatMessage> conversation =
                List.of(
                        SystemMessage.from("You are the support assistant of a shop."),
                        UserMessage.builder()
                                .name("ana")
                                .contents(
                                        List.of(
                                                TextContent.from("I was charged twice, receipt:"),
                                                ImageContent.from(png(), "image/png")))
                                .build(),
                        AiMessage.builder()
                                .text("Let me check.")
                                .thinking("left out")
                                .toolExecutionRequests(List.of(call))
                                .build(),
                        ToolExecutionResultMessage.builder()
                                .id("call-1")
                                .toolName("lookupOrder")
                                .text("order A-17 was not found")
                                .isError(true)
                                .build(),
                        AiMessage.builder().thinking("nothing to send").build(),
                        UserMessage.from("I want my money back."));
        model.decide(
                DecisionRequest.builder()
                        .inputMessages(conversation)
                        .question("q", YesNoQuestion.from("?"))
                        .build());

        var oracle =
                DecisionModelInputUtils.transcript(
                        conversation,
                        "oracle",
                        Set.of(ContentType.IMAGE, ContentType.AUDIO, ContentType.VIDEO));
        assertEquals(List.of("[image 1]"), List.copyOf(oracle.media().keySet()));
        assertEquals(2, stub.input.size());
        Media.Image image =
                assertInstanceOf(
                        Media.Image.class,
                        assertInstanceOf(Content.Media.class, stub.input.get(0)).media());
        assertEquals(3, image.width());
        assertEquals(
                new Content.Text(JsonCodec.stringify(Map.of("messages", oracle.messages()), true)),
                stub.input.get(1));
    }

    @Test
    void aConversationRefusesWhatItCannotHold() {
        for (ChatMessage message :
                List.of(
                        CustomMessage.from(Map.of("k", "v")),
                        UserMessage.from(
                                PdfFileContent.from(
                                        Base64.getEncoder().encodeToString(new byte[] {37}),
                                        "application/pdf")))) {
            assertThrows(
                    UnsupportedFeatureException.class,
                    () ->
                            model.decide(
                                    DecisionRequest.builder()
                                            .inputMessages(message)
                                            .question("q", YesNoQuestion.from("?"))
                                            .build()),
                    message.toString());
        }
        var empty =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.decide(
                                        DecisionRequest.builder()
                                                .inputMessages(
                                                        AiMessage.builder()
                                                                .thinking("only thinking")
                                                                .build())
                                                .question("q", YesNoQuestion.from("?"))
                                                .build()));
        assertTrue(empty.getMessage().contains("nothing to judge"), empty.getMessage());
        assertEquals(null, stub.input, "nothing reached the model");
    }

    @Test
    void independentAsksEachQuestionAlone() {
        try (JinferDecisionModel independent =
                JinferDecisionModel.builder().model(stub).independent(true).build()) {
            independent.decide(
                    DecisionRequest.builder()
                            .input("x")
                            .question("a", YesNoQuestion.from("A?"))
                            .question("b", YesNoQuestion.from("B?"))
                            .build());
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
            assertTrue(
                    first.decide(
                                    DecisionRequest.builder()
                                            .input("x")
                                            .question("q", YesNoQuestion.from("?"))
                                            .build())
                            .answers()
                            .containsKey("q"));
            assertTrue(
                    second.decide(
                                    DecisionRequest.builder()
                                            .input("x")
                                            .question("q", YesNoQuestion.from("?"))
                                            .build())
                            .answers()
                            .containsKey("q"));
        }
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

    private static String png() throws Exception {
        BufferedImage image = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB);
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        ImageIO.write(image, "png", out);
        return Base64.getEncoder().encodeToString(out.toByteArray());
    }
}
