package com.qxotic.jinfer.decision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.DecisionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SequencedMap;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The typed layer over the second decision family, vLLM Semantic Router's d3 (Qwen3.5): its own
 * prompt, readout and calibration, the same {@link Decisions} API. Numbers are llama.cpp's on the
 * same file (2026-11-10).
 */
@Tag("integration")
final class D3DecisionsIT {

    static final String D3 = "hf.co/ggml-org/vllm-d3-edge-GGUF/vllm-d3-edge-Q8_0.gguf";
    static final String MMPROJ = "hf.co/ggml-org/vllm-d3-edge-GGUF/mmproj-vllm-d3-edge-BF16.gguf";
    static final double TOLERANCE = 2e-2;

    static final String TICKET =
            "Customer: I was charged twice for my order last week and I want my money back today."
                    + " This is the third time I am writing.";

    enum Team {
        BILLING,
        SHIPPING,
        TECHNICAL,
        OTHER
    }

    static final Arena ARENA = Arena.ofShared();
    static DecisionModel d3;

    @BeforeAll
    static void load() throws Exception {
        d3 =
                Models.loadDecisionModel(
                        TestModels.require(D3), ARENA, Map.of("media", TestModels.require(MMPROJ)));
    }

    @AfterAll
    static void close() {
        ARENA.close();
    }

    @Test
    void aTicketIsRoutedAndRated() {
        try (Decisions decisions = Decisions.over(d3, 2048)) {
            var refund = Question.predicate("Is the customer asking for a refund?");
            var team = Question.choice("Which team should handle this?", Team.class);
            var urgency =
                    Question.rating(
                            "How urgent is this?", "Can wait", "This week", "Today", "Right now");
            Answers answers = decisions.askEach(TICKET, refund, team, urgency);
            assertEquals(0.97326, answers.get(refund).yes(), TOLERANCE);
            assertTrue(answers.get(refund).isYes());
            assertEquals(Team.BILLING, answers.get(team).selectedOption());
            assertTrue(answers.get(team).confidence() > 0.9f, "team " + answers.get(team));
            // d3 reads this ticket as the calmest level; the mean is its expected level
            assertEquals(1.127, answers.get(urgency).mean(), 0.05);

            // the same ticket with d3's own option descriptions: llama.cpp's number
            SequencedMap<String, String> described = new LinkedHashMap<>();
            described.put("billing", "Charges and refunds");
            described.put("shipping", "Delivery problems");
            described.put("technical", "Product does not work");
            described.put("other", null);
            Answer.Choice<String> routed =
                    decisions.ask(
                            TICKET, Question.choice("Which team should handle this?", described));
            assertEquals("billing", routed.selectedOption());
            assertEquals(0.97421f, routed.confidence(), TOLERANCE);
        }
    }

    /** d3 reads images, and only images: sound is refused naming what it takes. */
    @Test
    void mediaIsImagesOnly() throws Exception {
        try (Decisions decisions = Decisions.over(d3, 2048)) {
            var cat = Question.predicate("Is there a cat in the picture?");
            List<Content> photo =
                    List.of(
                            new Content.Media(
                                    ImageCodec.load(
                                            Path.of("../../test-fixtures/lfm3/images/cat.png"))),
                            new Content.Text("A photo sent by a customer."));
            assertTrue(decisions.ask(photo, cat).isYes(), "the cat");

            List<Content> clip =
                    List.of(
                            new Content.Media(
                                    AudioCodec.decode(
                                            Files.readAllBytes(
                                                    Path.of(
                                                            "../../test-fixtures/parakeet/ls0"
                                                                    + ".wav")))),
                            new Content.Text("A recording sent by a customer."));
            UnsupportedOperationException refused =
                    assertThrows(
                            UnsupportedOperationException.class, () -> decisions.ask(clip, cat));
            assertTrue(refused.getMessage().contains("takes no audio"), refused.getMessage());
        }
    }

    /** An empty picture list and a video without frames fail before any inference. */
    @Test
    void anUnusableMediaIsRefused() throws Exception {
        try (Decisions decisions = Decisions.over(d3, 2048)) {
            var cat = Question.predicate("Is there a cat?");
            List<Content> video =
                    List.of(
                            new Content.Media(new Media.Video(List.of())),
                            new Content.Text("A clip."));
            UnsupportedOperationException refused =
                    assertThrows(
                            UnsupportedOperationException.class, () -> decisions.ask(video, cat));
            assertTrue(refused.getMessage().contains("takes no video"), refused.getMessage());
        }
    }
}
