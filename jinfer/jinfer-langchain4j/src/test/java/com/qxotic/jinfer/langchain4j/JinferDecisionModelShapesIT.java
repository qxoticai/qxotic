package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.PdfFileContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.router.DecisionModelChatModelRouter;
import dev.langchain4j.model.chat.router.RoutingChatModel;
import dev.langchain4j.model.decision.request.ChoiceQuestion;
import dev.langchain4j.model.decision.request.DecisionRequest;
import dev.langchain4j.model.decision.request.ScaleQuestion;
import dev.langchain4j.model.decision.request.YesNoQuestion;
import dev.langchain4j.model.decision.response.DecisionResponse;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiFunction;
import java.util.function.Function;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The adapter's shapes: many questions in one request, contents in every order, an empty text
 * content, a content a decision cannot judge, a field of empty contents, a conversation, and one
 * instance serving every shape in sequence. Liquid d1-2B with its vision tower, so {@code
 * integration}.
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

    private static String base64(String name) throws Exception {
        return Base64.getEncoder()
                .encodeToString(
                        Files.readAllBytes(TestModels.fixture("lfm3/images/" + name + ".png")));
    }

    /** Forty questions in one request, each answered, in the request's order. */
    @Test
    void manyQuestionsInOneRequest() {
        DecisionRequest.Builder builder = DecisionRequest.builder().input(TICKET);
        for (int i = 0; i < 40; i++)
            builder.question("q" + i, YesNoQuestion.from("Is the customer asking for a refund?"));
        DecisionResponse response = model.decide(builder.build());
        assertEquals(40, response.answers().size());
        List<String> names = List.copyOf(response.answers().keySet());
        assertEquals("q0", names.getFirst(), "order kept");
        assertEquals("q39", names.getLast(), "order kept");
        for (int i = 0; i < 40; i++)
            assertTrue(response.yesNo("q" + i).probability() > 0.9, "q" + i);
    }

    /**
     * Contents in every order: the media are read first whatever their place, as the family says.
     */
    @Test
    void contentsInAnyOrderAreRead() throws Exception {
        String cat = base64("cat");
        BiFunction<String, String, DecisionRequest> request =
                (first, second) ->
                        DecisionRequest.builder()
                                .inputContents(
                                        first.equals("text")
                                                ? TextContent.from(TICKET)
                                                : ImageContent.from(cat, "image/png"),
                                        second.equals("text")
                                                ? TextContent.from(TICKET)
                                                : ImageContent.from(cat, "image/png"))
                                .question(
                                        "cat", YesNoQuestion.from("Is there a cat in the picture?"))
                                .question(
                                        "shows",
                                        ChoiceQuestion.from(
                                                "What does the picture show?",
                                                List.of("cat", "dog", "dice", "landscape")))
                                .build();
        // warm-up: a cold call reduces in un-compiled code, in another order than the
        // compiled Vector API kernels, which then answer alike run after run
        model.decide(request.apply("text", "image"));
        DecisionResponse textFirst = model.decide(request.apply("text", "image"));
        DecisionResponse imageFirst = model.decide(request.apply("image", "text"));
        assertEquals(textFirst.choice("shows").value(), imageFirst.choice("shows").value());
        // two calls round the Vector API kernels' reductions in an unspecified order
        assertEquals(
                textFirst.yesNo("cat").probability(), imageFirst.yesNo("cat").probability(), 1e-3);
        assertEquals("cat", textFirst.choice("shows").value());
    }

    /** An empty text content is an empty text: the picture still reaches the model. */
    @Test
    void anEmptyTextContentIsAText() throws Exception {
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .inputContents(
                                        ImageContent.from(base64("dice"), "image/png"),
                                        TextContent.from(""))
                                .question(
                                        "shows",
                                        ChoiceQuestion.from(
                                                "What does the picture show?",
                                                List.of("cat", "dog", "dice", "landscape")))
                                .build());
        assertEquals("dice", response.choice("shows").value());
    }

    /** Contents of media alone: no text at all, the question follows the picture. */
    @Test
    void mediaAloneContentsAnswer() throws Exception {
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .inputContents(ImageContent.from(base64("cat"), "image/png"))
                                .question("cat", YesNoQuestion.from("Is there a cat?"))
                                .build());
        assertTrue(response.yesNo("cat").isYes(0.5));
    }

    /** A content a decision cannot judge is refused in the framework's own vocabulary. */
    @Test
    void anUnsupportedContentIsRefused() throws Exception {
        UnsupportedFeatureException refused =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.decide(
                                        DecisionRequest.builder()
                                                .inputContents(
                                                        PdfFileContent.from(
                                                                base64("cat"), "application/pdf"))
                                                .question("q", YesNoQuestion.from("?"))
                                                .build()));
        assertTrue(refused.getMessage().contains("not supported"), refused.getMessage());
    }

    /** A field of empty contents contributes an empty value; the rest of the JSON is intact. */
    @Test
    void aFieldOfEmptyContentsKeepsItsValue() {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("note", List.of(TextContent.from("")));
        fields.put("message", TICKET);
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .inputFields(fields)
                                .question(
                                        "refund",
                                        YesNoQuestion.from("Is the customer asking for a refund?"))
                                .build());
        assertTrue(response.yesNo("refund").isYes(0.5));
    }

    /** One instance, every shape in sequence: nothing leaks from one request into the next. */
    @Test
    void everyShapeInSequenceOnOneInstance() throws Exception {
        double first =
                model.decide(
                                DecisionRequest.builder()
                                        .input(TICKET)
                                        .question("q", YesNoQuestion.from("Refund?"))
                                        .build())
                        .yesNo("q")
                        .probability();

        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("message", "Where is my parcel?");
        model.decide(
                DecisionRequest.builder()
                        .inputFields(fields)
                        .question(
                                "q",
                                ChoiceQuestion.from(
                                        "Which team?", List.of("billing", "shipping", "other")))
                        .build());
        model.decide(
                DecisionRequest.builder()
                        .inputContents(
                                ImageContent.from(base64("cat"), "image/png"),
                                TextContent.from("A photo."))
                        .question("q", YesNoQuestion.from("Is there a cat?"))
                        .build());
        model.decide(
                DecisionRequest.builder()
                        .inputContents(ImageContent.from(base64("dice"), "image/png"))
                        .question(
                                "q",
                                ScaleQuestion.from(
                                        "How bright is it?", List.of("dark", "dim", "bright")))
                        .build());

        assertEquals(
                first,
                model.decide(
                                DecisionRequest.builder()
                                        .input(TICKET)
                                        .question("q", YesNoQuestion.from("Refund?"))
                                        .build())
                        .yesNo("q")
                        .probability(),
                1e-4,
                "the first shape again reads the same");
    }

    /** A conversation is read whole: the same last words mean what the turns before make them. */
    @Test
    void aConversationIsReadWhole() {
        Function<String, Double> refund =
                offer ->
                        model.decide(
                                        DecisionRequest.builder()
                                                .inputMessages(
                                                        UserMessage.from(
                                                                "I was charged twice for order"
                                                                        + " A-17."),
                                                        AiMessage.from(offer),
                                                        UserMessage.from("Yes please."))
                                                .question(
                                                        "refund",
                                                        YesNoQuestion.from(
                                                                "Is the customer asking for a"
                                                                        + " refund?"))
                                                .build())
                                .yesNo("refund")
                                .probability();
        double asked = refund.apply("Sorry about that! Shall I refund the duplicate charge?");
        double declined = refund.apply("Sorry about that! Would you like our newsletter instead?");
        assertTrue(asked > 0.5, "the offer accepted is a refund: " + asked);
        assertTrue(declined < asked, "a newsletter accepted is less of one: " + declined);
    }

    /**
     * langchain4j's own router over this model: {@code DecisionModelChatModelRouter} asks with the
     * request's conversation, so a follow-up is routed by what it follows. {@code FAIL}, so a
     * refusal surfaces instead of quietly taking the default route.
     */
    @Test
    void routesAChatModelByItsConversation() {
        List<String> routed = new ArrayList<>();
        Function<String, ChatModel> named =
                name ->
                        new ChatModel() {
                            @Override
                            public ChatResponse doChat(ChatRequest request) {
                                routed.add(name);
                                return ChatResponse.builder()
                                        .aiMessage(AiMessage.from(name))
                                        .build();
                            }
                        };
        ChatModel routing =
                RoutingChatModel.builder()
                        .route(
                                "small",
                                "Greetings, small talk, short factual questions",
                                named.apply("small"))
                        .route(
                                "large",
                                "Writing code, multi-step reasoning, analysis",
                                named.apply("large"))
                        .router(
                                DecisionModelChatModelRouter.builder()
                                        .decisionModel(model)
                                        .fallbackStrategy(
                                                DecisionModelChatModelRouter.FallbackStrategy.FAIL)
                                        .build())
                        .defaultRoute("small")
                        .build();
        List<ChatMessage> code =
                List.of(
                        UserMessage.from("I need to parse ISO-8601 dates with time zones in Java."),
                        AiMessage.from(
                                "I can write a parser that handles offsets, zone ids and leap"
                                        + " seconds, with tests. Shall I go ahead?"),
                        UserMessage.from("Yes, go ahead."));
        routing.chat(List.of(UserMessage.from("Hi there, how are you?")));
        routing.chat(code);
        assertEquals(List.of("small", "large"), routed);
    }
}
