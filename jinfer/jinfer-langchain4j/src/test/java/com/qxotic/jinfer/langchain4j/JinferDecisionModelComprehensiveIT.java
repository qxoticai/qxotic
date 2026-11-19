package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.CustomMessage;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.PdfFileContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.VideoContent;
import dev.langchain4j.exception.AsyncNotSupportedException;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.decision.listener.DecisionModelErrorContext;
import dev.langchain4j.model.decision.listener.DecisionModelListener;
import dev.langchain4j.model.decision.listener.DecisionModelRequestContext;
import dev.langchain4j.model.decision.listener.DecisionModelResponseContext;
import dev.langchain4j.model.decision.request.ChoiceQuestion;
import dev.langchain4j.model.decision.request.DecisionRequest;
import dev.langchain4j.model.decision.request.DecisionRequestParameters;
import dev.langchain4j.model.decision.request.ScaleQuestion;
import dev.langchain4j.model.decision.request.YesNoQuestion;
import dev.langchain4j.model.decision.response.ChoiceAnswer;
import dev.langchain4j.model.decision.response.DecisionResponse;
import dev.langchain4j.model.decision.response.ScaleAnswer;
import dev.langchain4j.model.decision.response.YesNoAnswer;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * langchain4j's decision model beyond the parity cases: video through the sampler seam, a scale's
 * mean and probabilities, nested fields, and the failures a caller can act on. Liquid d1-2B with
 * its vision tower, so {@code integration}.
 */
@Tag("integration")
final class JinferDecisionModelComprehensiveIT {

    static final String D1 = "hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf";
    static final String MMPROJ = "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf";
    static final double TOLERANCE = 2e-2;

    static final Arena ARENA = Arena.ofShared();
    static DecisionModel core;
    static JinferDecisionModel model;

    @BeforeAll
    static void load() throws Exception {
        core =
                Models.loadDecisionModel(
                        TestModels.require(D1), ARENA, Map.of("media", TestModels.require(MMPROJ)));
        model = JinferDecisionModel.builder().model(core).build();
    }

    @AfterAll
    static void close() {
        if (model != null) model.close();
        ARENA.close();
    }

    /** The sampler is the ffmpeg seam: these are the six frames the parity clip is judged on. */
    @Test
    void videoIsJudgedThroughTheTower() throws Exception {
        List<Media.Video.Frame> frames = new ArrayList<>();
        for (int index : new int[] {0, 2, 4, 7, 9, 11})
            frames.add(
                    new Media.Video.Frame(
                            ImageCodec.load(
                                    TestModels.fixture(
                                            String.format("lfm3/video/clip-%02d.png", index))),
                            Duration.ofMillis(index * 250)));
        Media.Video clip = new Media.Video(frames);
        Path placeholder = Files.createTempFile("clip", ".mp4");
        try (JinferDecisionModel video =
                JinferDecisionModel.builder().model(core).videoSampler(ignored -> clip).build()) {
            DecisionResponse response =
                    video.decide(
                            DecisionRequest.builder()
                                    .inputContents(
                                            VideoContent.from(placeholder.toUri().toString()),
                                            TextContent.from("A clip sent by a customer."))
                                    .question(
                                            "both",
                                            YesNoQuestion.from(
                                                    "Does the video show both a cat and dice?"))
                                    .question(
                                            "first",
                                            ChoiceQuestion.from(
                                                    "What does the video show first?",
                                                    List.of("cat", "dog", "dice", "landscape")))
                                    .question(
                                            "last",
                                            ChoiceQuestion.from(
                                                    "What does the video show at the end?",
                                                    List.of("cat", "dog", "dice", "landscape")))
                                    .build());
            assertTrue(response.yesNo("both").isYes(0.5));
            assertEquals(0.98811, response.yesNo("both").probability(), TOLERANCE);
            assertEquals("cat", response.choice("first").value());
            assertEquals(0.97993, response.choice("first").probabilityOf("cat"), TOLERANCE);
            assertEquals("dice", response.choice("last").value());
            assertEquals(0.98170, response.choice("last").probabilityOf("dice"), TOLERANCE);
        } finally {
            Files.deleteIfExists(placeholder);
        }
    }

    @Test
    void aScaleIsTheProbabilityWeightedMean() {
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .input(
                                        "Customer: I was charged twice and I want my money back"
                                                + " today.")
                                .question(
                                        "urgency",
                                        ScaleQuestion.from(
                                                "How urgent is this?",
                                                List.of(
                                                        "Can wait",
                                                        "This week",
                                                        "Today",
                                                        "Right now")))
                                .build());
        ScaleAnswer scale = response.scale("urgency");
        assertEquals(4, scale.probabilities().size());
        double sum = 0, mean = 0, likeliest = 0;
        for (int i = 0; i < scale.probabilities().size(); i++) {
            sum += scale.probabilities().get(i);
            mean += i * scale.probabilities().get(i);
            likeliest = Math.max(likeliest, scale.probabilities().get(i));
        }
        assertEquals(1, sum, 1e-5);
        assertEquals(mean, scale.mean(), 1e-5);
        // jinfer's confidence, as on every face: the likeliest level's probability
        assertEquals(likeliest, scale.confidence(), 1e-6);
    }

    @Test
    void nestedFieldsArePutToTheModelAsJson() {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("customer", Map.of("name", "Ana", "vip", true));
        fields.put("order", 4411);
        fields.put("tags", List.of("billing", "repeat"));
        fields.put("message", "I was charged twice and I want my money back today.");
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .inputFields(fields)
                                .question(
                                        "refund",
                                        YesNoQuestion.from("Is the customer asking for a refund?"))
                                .question(
                                        "team",
                                        ChoiceQuestion.from(
                                                "Which team?",
                                                List.of(
                                                        "billing",
                                                        "shipping",
                                                        "technical",
                                                        "other")))
                                .build());
        assertTrue(response.yesNo("refund").isYes(0.5));
        assertEquals("billing", response.choice("team").value());
    }

    /** The documented quick start: a hub ref and its companion ref, resolved by the builder. */
    @Test
    void modelRefsResolveThroughTheStore() throws Exception {
        Path models = fixtureModels();
        Assumptions.assumeTrue(models != null, "no models tree beside the checkout");
        String previous = System.getProperty("jinfer.models");
        System.setProperty("jinfer.models", models.toString());
        try (JinferDecisionModel refs =
                JinferDecisionModel.builder()
                        .model("LiquidAI/d1-2B-GGUF:Q8_0")
                        .companion("media", "LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf")
                        .build()) {
            DecisionResponse response =
                    refs.decide(
                            DecisionRequest.builder()
                                    .input(
                                            "Customer: I was charged twice and I want my money"
                                                    + " back today.")
                                    .question(
                                            "refund",
                                            YesNoQuestion.from(
                                                    "Is the customer asking for a refund?"))
                                    .build());
            assertTrue(response.yesNo("refund").isYes(0.5));
        } finally {
            if (previous == null) System.clearProperty("jinfer.models");
            else System.setProperty("jinfer.models", previous);
        }
    }

    /** A field holding an image is sent as media, not JSON text: the framework's own contract. */
    @Test
    void aFieldHoldingAnImageIsJudgedAsMedia() throws Exception {
        String cat =
                Base64.getEncoder()
                        .encodeToString(
                                Files.readAllBytes(TestModels.fixture("lfm3/images/cat.png")));
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("message", "A photo sent by a customer.");
        fields.put("photo", ImageContent.from(cat, "image/png"));
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .inputFields(fields)
                                .question(
                                        "cat", YesNoQuestion.from("Is there a cat in the picture?"))
                                .question(
                                        "shows",
                                        ChoiceQuestion.from(
                                                "What does the picture show?",
                                                List.of("cat", "dog", "dice", "landscape")))
                                .build());
        assertTrue(response.yesNo("cat").isYes(0.9), String.valueOf(response.yesNo("cat")));
        assertEquals("cat", response.choice("shows").value());
    }

    @Test
    void whatTheCallerGotWrongIsRefusedWithAReason() {
        String broken = Base64.getEncoder().encodeToString(new byte[] {1, 2, 3});
        UncheckedIOException undecodable =
                assertThrows(
                        UncheckedIOException.class,
                        () ->
                                model.decide(
                                        DecisionRequest.builder()
                                                .inputContents(
                                                        ImageContent.from(broken, "image/png"))
                                                .question("q", YesNoQuestion.from("?"))
                                                .build()));
        assertTrue(
                undecodable.getMessage().contains("could not be decoded"),
                undecodable.getMessage());

        UnsupportedFeatureException remote =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.decide(
                                        DecisionRequest.builder()
                                                .inputContents(
                                                        ImageContent.from(
                                                                "https://example.com/a.png"))
                                                .question("q", YesNoQuestion.from("?"))
                                                .build()));
        assertTrue(
                remote.getMessage().contains("remote image/media URLs are not supported"),
                remote.getMessage());

        UnsupportedFeatureException inlineVideo =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.decide(
                                        DecisionRequest.builder()
                                                .inputContents(
                                                        VideoContent.from(broken, "video/mp4"))
                                                .question("q", YesNoQuestion.from("?"))
                                                .build()));
        assertTrue(
                inlineVideo.getMessage().contains("inline base64 video is not supported"),
                inlineVideo.getMessage());

        // a document is not something a decision model judges: refused in the API's vocabulary
        UnsupportedFeatureException document =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                model.decide(
                                        DecisionRequest.builder()
                                                .inputContents(
                                                        PdfFileContent.from(
                                                                broken, "application/pdf"))
                                                .question("q", YesNoQuestion.from("?"))
                                                .build()));
        assertTrue(document.getMessage().contains("is not supported"), document.getMessage());

        IllegalArgumentException empty =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.decide(
                                        DecisionRequest.builder()
                                                .inputContents(List.of())
                                                .question("q", YesNoQuestion.from("?"))
                                                .build()));
        assertTrue(
                empty.getMessage().contains("contents cannot be null or empty"),
                empty.getMessage());
    }

    /** The synchronous pipeline is the model; the async entry point says so in its own channel. */
    @Test
    void asyncIsNotImplementedAndSaysSo() {
        CompletableFuture<DecisionResponse> future =
                model.decideAsync(
                        DecisionRequest.builder()
                                .input("x")
                                .question("q", YesNoQuestion.from("?"))
                                .build());
        ExecutionException failure = assertThrows(ExecutionException.class, future::get);
        assertTrue(
                failure.getCause() instanceof AsyncNotSupportedException,
                String.valueOf(failure.getCause()));
    }

    @Test
    void capacityAndLifecycleAreTheModelsRule() throws Exception {
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferDecisionModel.builder().model(core).contextCapacity(200_000).build());

        JinferDecisionModel small =
                JinferDecisionModel.builder().model(core).contextCapacity(64).build();
        try (small) {
            String cat =
                    Base64.getEncoder()
                            .encodeToString(
                                    Files.readAllBytes(TestModels.fixture("lfm3/images/cat.png")));
            IllegalArgumentException tooLong =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    small.decide(
                                            DecisionRequest.builder()
                                                    .inputContents(
                                                            ImageContent.from(cat, "image/png"))
                                                    .question("cat", YesNoQuestion.from("A cat?"))
                                                    .build()));
            assertTrue(tooLong.getMessage().contains("media take"), tooLong.getMessage());
        }

        JinferDecisionModel closed = JinferDecisionModel.builder().model(core).build();
        closed.close();
        closed.close(); // idempotent
        assertThrows(
                IllegalStateException.class,
                () ->
                        closed.decide(
                                DecisionRequest.builder()
                                        .input("x")
                                        .question("q", YesNoQuestion.from("?"))
                                        .build()));
    }

    /** Audit logging, metrics and tracing hang off the framework's listener callbacks. */
    @Test
    void listenersSeeTheRequestTheResponseAndTheError() {
        AtomicInteger requests = new AtomicInteger(), responses = new AtomicInteger();
        AtomicInteger errors = new AtomicInteger();
        AtomicReference<DecisionResponse> answer = new AtomicReference<>();
        DecisionModelListener listener =
                new DecisionModelListener() {
                    @Override
                    public void onRequest(DecisionModelRequestContext context) {
                        requests.incrementAndGet();
                        assertFalse(context.decisionRequest().questions().isEmpty());
                    }

                    @Override
                    public void onResponse(DecisionModelResponseContext context) {
                        responses.incrementAndGet();
                        answer.set(context.decisionResponse());
                    }

                    @Override
                    public void onError(DecisionModelErrorContext context) {
                        errors.incrementAndGet();
                    }
                };
        try (JinferDecisionModel listened =
                JinferDecisionModel.builder().model(core).listeners(List.of(listener)).build()) {
            assertEquals(List.of(listener), listened.listeners());
            DecisionResponse decided =
                    listened.decide(
                            DecisionRequest.builder()
                                    .input(
                                            "I was charged twice and I want my money back"
                                                    + " today.")
                                    .question(
                                            "refund",
                                            YesNoQuestion.from(
                                                    "Is the customer asking for a refund?"))
                                    .build());
            assertEquals(1, requests.get());
            assertEquals(1, responses.get());
            assertEquals(0, errors.get());
            assertTrue(answer.get().yesNo("refund").isYes(0.5));
            assertEquals(decided, answer.get());

            // a refused input reaches onError, and the caller sees the same failure
            assertThrows(
                    UnsupportedFeatureException.class,
                    () ->
                            listened.decide(
                                    DecisionRequest.builder()
                                            .inputMessages(CustomMessage.from(Map.of("k", "v")))
                                            .question("q", YesNoQuestion.from("?"))
                                            .build()));
            assertEquals(1, errors.get());
        }
    }

    /** One loaded checkpoint names itself; a request naming another is refused, not answered. */
    @Test
    void theModelNameIsTheCheckpointAndCannotBeSwitched() {
        try (JinferDecisionModel named =
                JinferDecisionModel.builder().modelPath(TestModels.require(D1)).build()) {
            assertEquals("d1-2B-Q8_0.gguf", named.modelName());
            DecisionRequest same =
                    DecisionRequest.builder()
                            .input("I was charged twice and I want my money back today.")
                            .question(
                                    "refund",
                                    YesNoQuestion.from("Is the customer asking for a refund?"))
                            .parameters(
                                    DecisionRequestParameters.builder()
                                            .modelName("d1-2B-Q8_0.gguf")
                                            .build())
                            .build();
            DecisionResponse response = named.decide(same);
            assertEquals("d1-2B-Q8_0.gguf", response.modelName());
            assertTrue(response.yesNo("refund").isYes(0.5));

            DecisionRequest other =
                    DecisionRequest.builder()
                            .input("x")
                            .question("q", YesNoQuestion.from("?"))
                            .parameters(
                                    DecisionRequestParameters.builder()
                                            .modelName("some-other.gguf")
                                            .build())
                            .build();
            UnsupportedFeatureException refused =
                    assertThrows(UnsupportedFeatureException.class, () -> named.decide(other));
            assertTrue(refused.getMessage().contains("this model IS"), refused.getMessage());
        }
    }

    /**
     * A support triage pipeline: route, gate a refund, rate urgency, and escalate on a narrow
     * margin.
     */
    @Test
    void aSupportTriagePipelineRunsEndToEnd() {
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .inputFields(
                                        Map.of(
                                                "customer", "Ana",
                                                "plan", "enterprise",
                                                "message",
                                                        "I was charged twice for my order and I"
                                                                + " want my money back today."))
                                .question(
                                        "team",
                                        ChoiceQuestion.builder()
                                                .text("Which team should handle this ticket?")
                                                .option("billing", "Payments, invoices, refunds")
                                                .option("support", "Problems using the product")
                                                .option("sales", "Plans, pricing, upgrades")
                                                .build())
                                .question(
                                        "refund",
                                        YesNoQuestion.builder()
                                                .text("Is the customer asking for a refund?")
                                                .yesWhen("asks for money back")
                                                .noWhen("asks anything else")
                                                .build())
                                .question(
                                        "urgency",
                                        ScaleQuestion.from(
                                                "How urgent is this?",
                                                List.of(
                                                        "Can wait",
                                                        "This week",
                                                        "Today",
                                                        "Right now")))
                                .question(
                                        "human",
                                        YesNoQuestion.from(
                                                "Should a human read this before it is answered?"))
                                .build());

        // the routing answer is the one to act on, with its margin as the escalation signal
        assertEquals("billing", response.choice("team").value());
        assertTrue(response.choice("team").margin() > 0.5, String.valueOf(response.choice("team")));
        assertTrue(response.yesNo("refund").isYes(0.9));
        assertTrue(
                response.scale("urgency").mean() >= 2, String.valueOf(response.scale("urgency")));
        assertEquals(
                List.of("team", "refund", "urgency", "human"),
                List.copyOf(response.answers().keySet()));
    }

    /**
     * The same adapter answers a second decision family: vLLM's d3 on Qwen3.5, whose prompt and
     * readout differ entirely from d1's. Numbers are llama.cpp's on the same file.
     */
    @Test
    void aSecondDecisionFamilyAnswersThroughTheSameAdapter() throws Exception {
        try (JinferDecisionModel d3 =
                JinferDecisionModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/ggml-org/vllm-d3-edge-GGUF/vllm-d3-edge-Q8_0.gguf"))
                        .companionPath(
                                "media",
                                TestModels.require(
                                        "hf.co/ggml-org/vllm-d3-edge-GGUF/"
                                                + "mmproj-vllm-d3-edge-BF16.gguf"))
                        .build()) {
            DecisionResponse response =
                    d3.decide(
                            DecisionRequest.builder()
                                    .input(
                                            "Customer: I was charged twice for my order last week"
                                                    + " and I want my money back today. This is the"
                                                    + " third time I am writing.")
                                    .question(
                                            "refund",
                                            YesNoQuestion.from(
                                                    "Is the customer asking for a refund?"))
                                    .question(
                                            "team",
                                            ChoiceQuestion.builder()
                                                    .text("Which team should handle this?")
                                                    .option("billing", "Charges and refunds")
                                                    .option("shipping", "Delivery problems")
                                                    .option("technical", "Product does not work")
                                                    .option("other")
                                                    .build())
                                    .question(
                                            "urgency",
                                            ScaleQuestion.from(
                                                    "How urgent is this?",
                                                    List.of(
                                                            "Can wait",
                                                            "This week",
                                                            "Today",
                                                            "Right now")))
                                    .build());
            assertEquals(0.97326, response.yesNo("refund").probability(), TOLERANCE);
            assertTrue(response.yesNo("refund").isYes(0.5));
            assertEquals(0.97421, response.choice("team").probabilityOf("billing"), TOLERANCE);
            assertEquals("billing", response.choice("team").value());
            // d3's own distribution: this ticket reads as the calmest level
            assertEquals(1.127, response.scale("urgency").mean(), 0.05);
        }
    }

    /** The framework's own contracts: typed accessors, metadata, no usage, no refusals. */
    @Test
    void theResponseHonoursTheFrameworksContracts() {
        String ticket =
                "Customer: I was charged twice for my order last week and I want my money back"
                        + " today.";
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .input(ticket)
                                .question(
                                        "team",
                                        ChoiceQuestion.builder()
                                                .text("Which team?")
                                                .option("billing", "Charges and refunds")
                                                .option("other")
                                                .build())
                                .question(
                                        "refund",
                                        YesNoQuestion.from("Is the customer asking for a refund?"))
                                .question(
                                        "urgency",
                                        ScaleQuestion.from(
                                                "How urgent?", List.of("Can wait", "Today", "Now")))
                                .build());
        assertEquals(
                List.of("team", "refund", "urgency"), List.copyOf(response.answers().keySet()));
        assertEquals("billing", response.answer("team", ChoiceAnswer.class).value());
        assertThrows(
                IllegalArgumentException.class, () -> response.answer("team", YesNoAnswer.class));
        assertTrue(!response.isRefused("team") && !response.isRefused("refund"));

        // a decision reads its input and generates nothing: the usage counts what it read. This
        // instance was built over a caller-loaded model, so it names none; from a path or a ref
        // the metadata carries the checkpoint
        assertTrue(response.tokenUsage().inputTokenCount() > 0, response.tokenUsage().toString());
        assertEquals(0, response.tokenUsage().outputTokenCount());
        assertNull(response.modelName());
        assertNull(response.metadata().modelName());
    }

    /** Contents of media alone: no text part, the question follows the picture directly. */
    @Test
    void contentsOfMediaAloneAreJudged() throws Exception {
        String cat =
                Base64.getEncoder()
                        .encodeToString(
                                Files.readAllBytes(TestModels.fixture("lfm3/images/cat.png")));
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .inputContents(ImageContent.from(cat, "image/png"))
                                .question(
                                        "cat", YesNoQuestion.from("Is there a cat in the picture?"))
                                .question(
                                        "shows",
                                        ChoiceQuestion.from(
                                                "What does the picture show?",
                                                List.of("cat", "dog", "dice", "landscape")))
                                .build());
        assertTrue(response.yesNo("cat").isYes(0.9), String.valueOf(response.yesNo("cat")));
        assertEquals("cat", response.choice("shows").value());
    }

    /** d3 has one prompt a question: its own way is each alone, the same as independent. */
    @Test
    void togetherOnAFamilyWithoutPackingAnswersEach() throws Exception {
        DecisionRequest request =
                DecisionRequest.builder()
                        .input("I was charged twice and I want my money back today.")
                        .question(
                                "refund",
                                YesNoQuestion.from("Is the customer asking for a refund?"))
                        .question(
                                "team",
                                ChoiceQuestion.builder()
                                        .text("Which team?")
                                        .option("billing", "Charges and refunds")
                                        .option("other")
                                        .build())
                        .build();
        Path d3 = TestModels.require("hf.co/ggml-org/vllm-d3-edge-GGUF/vllm-d3-edge-Q8_0.gguf");
        try (JinferDecisionModel each =
                        JinferDecisionModel.builder().modelPath(d3).independent(true).build();
                JinferDecisionModel packed = JinferDecisionModel.builder().modelPath(d3).build()) {
            // warm-up: a cold first call runs partly interpreted, and jam's int8 activations
            // magnify that past 1e-5
            each.decide(request);
            DecisionResponse one = each.decide(request);
            DecisionResponse all = packed.decide(request);
            assertEquals(
                    one.yesNo("refund").probability(), all.yesNo("refund").probability(), 1e-5);
            assertEquals(one.choice("team").value(), all.choice("team").value());
        }
    }

    /**
     * Eleven scale levels exceed what d1 can read (a digit a level): refused in the model's words.
     */
    @Test
    void elevenScaleLevelsAreRefused() {
        ScaleQuestion.Builder builder = ScaleQuestion.builder().text("?");
        for (int i = 0; i <= 10; i++) builder.level("level " + i);
        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.decide(
                                        DecisionRequest.builder()
                                                .input("x")
                                                .question("s", builder.build())
                                                .build()));
        assertTrue(refused.getMessage().contains("at most 10 levels"), refused.getMessage());
    }

    /** A choice stays in the model's code space: 200 options are all told apart. */
    @Test
    void aChoiceStaysInTheCodeSpace() {
        ChoiceQuestion.Builder builder = ChoiceQuestion.builder().text("Which one?");
        for (int i = 0; i < 200; i++) builder.option("option " + i);
        DecisionResponse response =
                model.decide(
                        DecisionRequest.builder()
                                .input("Which option is the last one?")
                                .question("c", builder.build())
                                .build());
        assertEquals(200, response.choice("c").probabilities().size());
        assertTrue(
                response.choice("c").probabilities().containsKey(response.choice("c").value()),
                response.choice("c").value());
    }

    /** The fixture tree beside the checkout, when scripts/download-models.sh filled it. */
    private static Path fixtureModels() {
        for (Path dir = Path.of("").toAbsolutePath(); dir != null; dir = dir.getParent())
            if (Files.exists(dir.resolve(".git"))) {
                Path models = dir.resolveSibling("models");
                return Files.isDirectory(models) ? models : null;
            }
        return null;
    }
}
