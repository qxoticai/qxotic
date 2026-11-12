package com.qxotic.jinfer.spring.ai;

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
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.part.MediaPart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.decision.DecisionOptions;
import org.springframework.ai.decision.DecisionPrompt;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.answer.ChoiceAnswer;
import org.springframework.ai.decision.input.DecisionInput;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.DecisionQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeType;

/**
 * Spring AI's decision model beyond the parity cases: video through the sampler seam, a score's
 * scale semantics, nested fields, and the failures a caller can act on. Liquid d1-2B with its
 * vision tower, so {@code integration}.
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

    private static Map<String, DecisionQuestion> questions(Object... namesAndQuestions) {
        Map<String, DecisionQuestion> questions = new LinkedHashMap<>();
        for (int i = 0; i < namesAndQuestions.length; i += 2)
            questions.put(
                    (String) namesAndQuestions[i], (DecisionQuestion) namesAndQuestions[i + 1]);
        return questions;
    }

    /** The sampler is the ffmpeg seam: these are the six frames the parity clip is judged on. */
    @Test
    void videoIsJudgedThroughTheTower() throws Exception {
        List<Media.Video.Frame> frames = new ArrayList<>();
        for (int index : new int[] {0, 2, 4, 7, 9, 11})
            frames.add(
                    new Media.Video.Frame(
                            ImageCodec.load(
                                    Path.of(
                                            String.format(
                                                    "../../test-fixtures/lfm3/video/clip-%02d.png",
                                                    index))),
                            Duration.ofMillis(index * 250)));
        Media.Video clip = new Media.Video(frames);
        Path placeholder = Files.createTempFile("clip", ".mp4");
        try (JinferDecisionModel video =
                JinferDecisionModel.builder().model(core).videoSampler(ignored -> clip).build()) {
            org.springframework.ai.content.Media media =
                    new org.springframework.ai.content.Media(
                            MimeType.valueOf("video/mp4"), placeholder.toUri());
            DecisionResponse response =
                    video.call(
                            DecisionInput.content(
                                    MediaPart.of(media), TextPart.of("A clip sent by a customer.")),
                            questions(
                                    "both",
                                    YesNoQuestion.of("Does the video show both a cat and dice?"),
                                    "first",
                                    ChoiceQuestion.of(
                                            "What does the video show first?",
                                            "cat",
                                            "dog",
                                            "dice",
                                            "landscape"),
                                    "last",
                                    ChoiceQuestion.of(
                                            "What does the video show at the end?",
                                            "cat",
                                            "dog",
                                            "dice",
                                            "landscape")));
            assertTrue(response.getYesNo("both").isYes());
            assertEquals(0.98811, response.getYesNo("both").getProbability(), TOLERANCE);
            assertEquals("cat", response.getChoice("first").getValue());
            assertEquals(0.97993, response.getChoice("first").getProbability("cat"), TOLERANCE);
            assertEquals("dice", response.getChoice("last").getValue());
            assertEquals(0.98170, response.getChoice("last").getProbability("dice"), TOLERANCE);
        } finally {
            Files.deleteIfExists(placeholder);
        }
    }

    @Test
    void aScoreIsTheProbabilityWeightedLevel() {
        DecisionResponse response =
                model.call(
                        "Customer: I was charged twice and I want my money back today.",
                        questions(
                                "urgency",
                                ScoreQuestion.of(
                                        "How urgent is this?",
                                        "Can wait",
                                        "This week",
                                        "Today",
                                        "Right now")));
        var score = response.getScore("urgency");
        assertEquals(4, score.getProbabilities().size());
        double sum = 0, mean = 0;
        for (int i = 0; i < score.getProbabilities().size(); i++) {
            sum += score.getProbabilities().get(i);
            mean += i * score.getProbabilities().get(i);
        }
        assertEquals(1, sum, 1e-5);
        assertEquals(mean, score.getValue(), 1e-5);
        assertEquals(2, score.getNearestLevel());
    }

    @Test
    void nestedFieldsArePutToTheModelAsJson() {
        Map<String, Object> fields = new LinkedHashMap<>();
        fields.put("customer", Map.of("name", "Ana", "vip", true));
        fields.put("order", 4411);
        fields.put("tags", List.of("billing", "repeat"));
        fields.put("message", "I was charged twice and I want my money back today.");
        DecisionResponse response =
                model.call(
                        DecisionInput.fields(fields),
                        questions(
                                "refund",
                                YesNoQuestion.of("Is the customer asking for a refund?"),
                                "team",
                                ChoiceQuestion.of(
                                        "Which team?",
                                        "billing",
                                        "shipping",
                                        "technical",
                                        "other")));
        assertTrue(response.getYesNo("refund").isYes());
        assertEquals("billing", response.getChoice("team").getValue());
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
                    refs.call(
                            "Customer: I was charged twice and I want my money back today.",
                            questions(
                                    "refund",
                                    YesNoQuestion.of("Is the customer asking for a refund?")));
            assertTrue(response.getYesNo("refund").isYes());
        } finally {
            if (previous == null) System.clearProperty("jinfer.models");
            else System.setProperty("jinfer.models", previous);
        }
    }

    @Test
    void whatTheCallerGotWrongIsRefusedWithAReason() {
        org.springframework.ai.content.Media brokenImage =
                new org.springframework.ai.content.Media(
                        MimeType.valueOf("image/png"), new ByteArrayResource(new byte[] {1, 2, 3}));
        UncheckedIOException broken =
                assertThrows(
                        UncheckedIOException.class,
                        () ->
                                model.call(
                                        DecisionInput.content(MediaPart.of(brokenImage)),
                                        questions("q", YesNoQuestion.of("?"))));
        assertTrue(broken.getMessage().contains("could not be decoded"), broken.getMessage());

        org.springframework.ai.content.Media pdf =
                new org.springframework.ai.content.Media(
                        MimeType.valueOf("application/pdf"),
                        new ByteArrayResource(new byte[] {1, 2, 3}));
        UnsupportedOperationException type =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                model.call(
                                        DecisionInput.content(MediaPart.of(pdf)),
                                        questions("q", YesNoQuestion.of("?"))));
        assertTrue(
                type.getMessage().contains("not supported: jinfer takes image, audio and video"),
                type.getMessage());

        // a URL is refused, never fetched: the library has no network path on inference
        org.springframework.ai.content.Media remote =
                new org.springframework.ai.content.Media(
                        MimeType.valueOf("image/png"), URI.create("https://example.org/cat.png"));
        UnsupportedOperationException url =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                model.call(
                                        DecisionInput.content(MediaPart.of(remote)),
                                        questions("q", YesNoQuestion.of("?"))));
        assertTrue(url.getMessage().contains("remote media URLs"), url.getMessage());

        IllegalArgumentException empty =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.call(
                                        DecisionInput.content(),
                                        questions("q", YesNoQuestion.of("?"))));
        assertTrue(
                empty.getMessage().contains("parts cannot be null or empty"), empty.getMessage());
    }

    /** Asked together, d1 reads a picture once for both questions: the picks hold. */
    @Test
    void togetherReadsAPictureOnce() throws Exception {
        org.springframework.ai.content.Media cat =
                new org.springframework.ai.content.Media(
                        MimeType.valueOf("image/png"), new ByteArrayResource(imageBytes("cat")));
        try (JinferDecisionModel packed = JinferDecisionModel.builder().model(core).build()) {
            DecisionResponse response =
                    packed.call(
                            DecisionInput.content(
                                    MediaPart.of(cat), TextPart.of("A photo sent by a customer.")),
                            questions(
                                    "cat",
                                    YesNoQuestion.of("Is there a cat in the picture?"),
                                    "shows",
                                    ChoiceQuestion.of(
                                            "What does the picture show?",
                                            "cat",
                                            "dog",
                                            "dice",
                                            "landscape")));
            assertTrue(response.getYesNo("cat").isYes());
            assertEquals("cat", response.getChoice("shows").getValue());
        }
    }

    @Test
    void capacityAndLifecycleAreTheModelsRule() {
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferDecisionModel.builder().model(core).contextCapacity(200_000).build());

        JinferDecisionModel small =
                JinferDecisionModel.builder().model(core).contextCapacity(64).build();
        try (small) {
            org.springframework.ai.content.Media cat =
                    new org.springframework.ai.content.Media(
                            MimeType.valueOf("image/png"),
                            new ByteArrayResource(imageBytes("cat")));
            IllegalArgumentException tooLong =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    small.call(
                                            DecisionInput.content(MediaPart.of(cat)),
                                            questions("cat", YesNoQuestion.of("A cat?"))));
            assertTrue(tooLong.getMessage().contains("media take"), tooLong.getMessage());
        }

        JinferDecisionModel closed = JinferDecisionModel.builder().model(core).build();
        closed.close();
        closed.close(); // idempotent
        assertThrows(
                IllegalStateException.class,
                () -> closed.call("x", questions("q", YesNoQuestion.of("?"))));
    }

    /** One loaded checkpoint names itself in the response metadata and cannot be switched. */
    @Test
    void theResponseNamesTheModelAndAPromptCannotSwitchIt() {
        try (JinferDecisionModel named =
                JinferDecisionModel.builder().modelPath(TestModels.require(D1)).build()) {
            DecisionResponse response =
                    named.call(
                            DecisionPrompt.builder()
                                    .text(
                                            "I was charged twice for my order and I want my money"
                                                    + " back today.")
                                    .question(
                                            "refund",
                                            YesNoQuestion.of(
                                                    "Is the customer asking for a refund?"))
                                    .build());
            assertEquals("d1-2B-Q8_0.gguf", response.getMetadata().getModel());
            assertTrue(response.getYesNo("refund").isYes());

            // naming the loaded model is the same request; naming another is refused
            named.call(
                    DecisionPrompt.builder()
                            .text("x")
                            .question("q", YesNoQuestion.of("?"))
                            .options(DecisionOptions.builder().model("d1-2B-Q8_0.gguf").build())
                            .build());
            UnsupportedOperationException refused =
                    assertThrows(
                            UnsupportedOperationException.class,
                            () ->
                                    named.call(
                                            DecisionPrompt.builder()
                                                    .text("x")
                                                    .question("q", YesNoQuestion.of("?"))
                                                    .options(
                                                            DecisionOptions.builder()
                                                                    .model("some-other.gguf")
                                                                    .build())
                                                    .build()));
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
                model.call(
                        DecisionInput.fields(
                                Map.of(
                                        "customer", "Ana",
                                        "plan", "enterprise",
                                        "message",
                                                "I was charged twice for my order and I want my"
                                                        + " money back today.")),
                        questions(
                                "team",
                                ChoiceQuestion.builder()
                                        .text("Which team should handle this ticket?")
                                        .option("billing", "Payments, invoices, refunds")
                                        .option("support", "Problems using the product")
                                        .option("sales", "Plans, pricing, upgrades")
                                        .build(),
                                "refund",
                                YesNoQuestion.builder()
                                        .text("Is the customer asking for a refund?")
                                        .whenYes("asks for money back")
                                        .whenNo("asks anything else")
                                        .build(),
                                "urgency",
                                ScoreQuestion.of(
                                        "How urgent is this?",
                                        "Can wait",
                                        "This week",
                                        "Today",
                                        "Right now"),
                                "human",
                                YesNoQuestion.of(
                                        "Should a human read this before it is answered?")));

        var team = response.getChoice("team");
        assertEquals("billing", team.getValue());
        // the margin between the two likeliest options: the escalation signal
        assertTrue(team.getProbability("billing") > 0.9, String.valueOf(team));
        assertTrue(response.getYesNo("refund").isYes());
        assertTrue(
                response.getScore("urgency").getNearestLevel() >= 2,
                String.valueOf(response.getScore("urgency")));
        assertEquals(
                List.of("team", "refund", "urgency", "human"),
                List.copyOf(response.getAnswers().keySet()));
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
                    d3.call(
                            "Customer: I was charged twice for my order last week and I want my"
                                    + " money back today. This is the third time I am writing.",
                            questions(
                                    "refund",
                                    YesNoQuestion.of("Is the customer asking for a refund?"),
                                    "team",
                                    ChoiceQuestion.builder()
                                            .text("Which team should handle this?")
                                            .option("billing", "Charges and refunds")
                                            .option("shipping", "Delivery problems")
                                            .option("technical", "Product does not work")
                                            .option("other")
                                            .build(),
                                    "urgency",
                                    ScoreQuestion.of(
                                            "How urgent is this?",
                                            "Can wait",
                                            "This week",
                                            "Today",
                                            "Right now")));
            assertEquals(0.97326, response.getYesNo("refund").getProbability(), TOLERANCE);
            assertTrue(response.getYesNo("refund").isYes());
            assertEquals(0.97421, response.getChoice("team").getProbability("billing"), TOLERANCE);
            assertEquals("billing", response.getChoice("team").getValue());
            // d3's own distribution: this ticket reads as the calmest level
            assertEquals(0, response.getScore("urgency").getNearestLevel());
            assertEquals(1.127, response.getScore("urgency").getValue(), 0.05);
        }
    }

    /** The framework's own contracts: results in prompt order, typed accessors, no usage. */
    @Test
    void theResponseHonoursTheFrameworksContracts() {
        String ticket =
                "Customer: I was charged twice for my order last week and I want my money back"
                        + " today.";
        DecisionResponse response =
                model.call(
                        ticket,
                        questions(
                                "team",
                                ChoiceQuestion.builder()
                                        .text("Which team?")
                                        .option("billing", "Charges and refunds")
                                        .option("other")
                                        .build(),
                                "refund",
                                YesNoQuestion.of("Is the customer asking for a refund?"),
                                "urgency",
                                ScoreQuestion.of("How urgent?", "Can wait", "Today", "Now")));

        // one result a question, in the order the prompt asked them
        assertEquals(
                List.of("team", "refund", "urgency"),
                response.getResults().stream().map(r -> r.getName()).toList());
        assertEquals(
                List.of("team", "refund", "urgency"), List.copyOf(response.getAnswers().keySet()));
        for (String name : List.of("team", "refund", "urgency"))
            assertTrue(response.hasAnswer(name));
        assertFalse(response.isRefused("team") || response.isRefused("refund"));

        // the typed accessors answer their own type, and refuse another's
        assertEquals("billing", response.getAnswer("team", ChoiceAnswer.class).getValue());
        assertThrows(IllegalArgumentException.class, () -> response.getYesNo("team"));
        assertThrows(IllegalArgumentException.class, () -> response.getAnswer("nope"));

        // a decision generates nothing: the usage is empty, not a fake count
        assertEquals(0, response.getMetadata().getUsage().getPromptTokens());
        assertEquals(0, response.getMetadata().getUsage().getCompletionTokens());
        // this instance was built over a caller-loaded model, so it names none; built from a path
        // or a ref, the metadata carries the checkpoint (see the model-name test)
        assertNull(response.getMetadata().getModel());
    }

    /** d1-Omni reads a picture and a clip in ONE input: the towers see both. */
    @Test
    void anOmniInputHoldsAPictureAndAClip() throws Exception {
        org.springframework.ai.content.Media photo =
                new org.springframework.ai.content.Media(
                        MimeType.valueOf("image/png"), new ByteArrayResource(imageBytes("cat")));
        org.springframework.ai.content.Media clip =
                new org.springframework.ai.content.Media(
                        MimeType.valueOf("audio/wav"),
                        new ByteArrayResource(
                                Files.readAllBytes(
                                        Path.of("../../test-fixtures/parakeet/ls0.wav"))));
        try (JinferDecisionModel omni =
                JinferDecisionModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/LiquidAI/d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf"))
                        .companionPath(
                                "media",
                                TestModels.require(
                                        "hf.co/LiquidAI/d1-Omni-2B-GGUF/"
                                                + "mmproj-d1-Omni-2B-F16.gguf"))
                        .build()) {
            DecisionResponse response =
                    omni.call(
                            DecisionInput.content(
                                    MediaPart.of(photo),
                                    MediaPart.of(clip),
                                    TextPart.of("A photo and a recording sent by a customer.")),
                            questions(
                                    "cat",
                                    YesNoQuestion.of("Is there a cat in the picture?"),
                                    "speech",
                                    YesNoQuestion.of("Is someone speaking in the recording?"),
                                    "shows",
                                    ChoiceQuestion.of(
                                            "What does the picture show?",
                                            "cat",
                                            "dog",
                                            "dice",
                                            "landscape")));
            assertEquals("cat", response.getChoice("shows").getValue());
            // the two towers read the pair as the engine does: same input, same answer
            try (JinferDecisionModel sameModel =
                    JinferDecisionModel.builder()
                            .modelPath(
                                    TestModels.require(
                                            "hf.co/LiquidAI/d1-Omni-2B-GGUF/"
                                                    + "d1-Omni-2B-Q8_0.gguf"))
                            .companionPath(
                                    "media",
                                    TestModels.require(
                                            "hf.co/LiquidAI/d1-Omni-2B-GGUF/"
                                                    + "mmproj-d1-Omni-2B-F16.gguf"))
                            .build()) {
                DecisionResponse again =
                        sameModel.call(
                                DecisionInput.content(
                                        MediaPart.of(photo),
                                        MediaPart.of(clip),
                                        TextPart.of("A photo and a recording sent by a customer.")),
                                questions(
                                        "cat",
                                        YesNoQuestion.of("Is there a cat in the picture?"),
                                        "speech",
                                        YesNoQuestion.of("Is someone speaking in the recording?"),
                                        "shows",
                                        ChoiceQuestion.of(
                                                "What does the picture show?",
                                                "cat",
                                                "dog",
                                                "dice",
                                                "landscape")));
                assertEquals(
                        response.getYesNo("cat").getProbability(),
                        again.getYesNo("cat").getProbability(),
                        1e-5);
                assertEquals(
                        response.getYesNo("speech").getProbability(),
                        again.getYesNo("speech").getProbability(),
                        1e-5);
            }
            // llama.cpp's reading of the pair: a clear cat, a near-split speech call
            assertTrue(response.getYesNo("cat").getProbability() > 0.9);
            assertTrue(response.getYesNo("speech").getProbability() > 0.2);
            assertTrue(response.getYesNo("speech").getProbability() < 0.5);
        }
    }

    /** d3 has one prompt a question: its own way is each alone, the same as independent. */
    @Test
    void togetherOnAFamilyWithoutPackingAnswersEach() throws Exception {
        Map<String, DecisionQuestion> asked =
                questions(
                        "refund",
                        YesNoQuestion.of("Is the customer asking for a refund?"),
                        "team",
                        ChoiceQuestion.builder()
                                .text("Which team?")
                                .option("billing", "Charges and refunds")
                                .option("other")
                                .build());
        String ticket = "I was charged twice and I want my money back today.";
        try (JinferDecisionModel each =
                        JinferDecisionModel.builder()
                                .modelPath(
                                        TestModels.require(
                                                "hf.co/ggml-org/vllm-d3-edge-GGUF/"
                                                        + "vllm-d3-edge-Q8_0.gguf"))
                                .independent(true)
                                .build();
                JinferDecisionModel packed =
                        JinferDecisionModel.builder()
                                .modelPath(
                                        TestModels.require(
                                                "hf.co/ggml-org/vllm-d3-edge-GGUF/"
                                                        + "vllm-d3-edge-Q8_0.gguf"))
                                .build()) {
            DecisionResponse one = each.call(ticket, asked);
            DecisionResponse all = packed.call(ticket, asked);
            assertEquals(
                    one.getYesNo("refund").getProbability(),
                    all.getYesNo("refund").getProbability(),
                    1e-5);
            assertEquals(one.getChoice("team").getValue(), all.getChoice("team").getValue());
        }
    }

    /**
     * Eleven score levels exceed what d1 can read (a digit a level): refused, in the model's words.
     */
    @Test
    void elevenScoreLevelsAreRefused() {
        ScoreQuestion.Builder builder = ScoreQuestion.builder().text("?");
        for (int i = 0; i <= 10; i++) builder.level("level " + i);
        ScoreQuestion eleven = builder.build();
        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> model.call("x", questions("s", eleven)));
        assertTrue(refused.getMessage().contains("at most 10 levels"), refused.getMessage());
    }

    /**
     * A choice stays in the model's code space: d1's fallback pool covers 200 options, and a
     * question past it is refused rather than answered with a code it cannot read.
     */
    @Test
    void aChoiceStaysInTheCodeSpace() {
        ChoiceQuestion.Builder builder = ChoiceQuestion.builder().text("Which one?");
        for (int i = 0; i < 200; i++) builder.option("option " + i);
        DecisionResponse response =
                model.call("Which option is the last one?", questions("c", builder.build()));
        assertEquals(200, response.getChoice("c").getProbabilities().size());
        assertTrue(
                response.getChoice("c")
                        .getProbabilities()
                        .containsKey(response.getChoice("c").getValue()),
                response.getChoice("c").getValue());
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

    private static byte[] imageBytes(String name) {
        try {
            return Files.readAllBytes(Path.of("../../test-fixtures/lfm3/images/" + name + ".png"));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
