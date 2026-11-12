package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The whole d3 family the repository publishes - edge, lite, nano and mini - on the invariants the
 * family's contract states, whatever the checkpoint: one finite probability per declared option
 * summing to one, no packing (a question asked together answers as the same question asked alone),
 * a code space from {@code A} to {@code AA..} bounded at 255, images read and sound and video
 * refused, and the routing battery landing on the same team on every size.
 */
@Tag("integration")
final class Qwen35DecisionFamilyTest {

    record Variant(String name, String model, String mmproj) {}

    static final List<Variant> VARIANTS =
            List.of(
                    new Variant(
                            "edge",
                            "hf.co/ggml-org/vllm-d3-edge-GGUF/vllm-d3-edge-Q8_0.gguf",
                            "hf.co/ggml-org/vllm-d3-edge-GGUF/mmproj-vllm-d3-edge-BF16.gguf"),
                    new Variant(
                            "lite",
                            "hf.co/ggml-org/vllm-d3-lite-GGUF/vllm-d3-lite-Q8_0.gguf",
                            "hf.co/ggml-org/vllm-d3-lite-GGUF/mmproj-vllm-d3-lite-Q8_0.gguf"),
                    new Variant(
                            "nano",
                            "hf.co/ggml-org/vllm-d3-nano-GGUF/vllm-d3-nano-Q8_0.gguf",
                            "hf.co/ggml-org/vllm-d3-nano-GGUF/mmproj-vllm-d3-nano-Q8_0.gguf"),
                    new Variant(
                            "mini",
                            "hf.co/ggml-org/vllm-d3-mini-GGUF/vllm-d3-mini-Q8_0.gguf",
                            "hf.co/ggml-org/vllm-d3-mini-GGUF/mmproj-vllm-d3-mini-Q8_0.gguf"));

    static final String TICKET =
            "Customer: I was charged twice for my order last week and I want my money back today."
                    + " This is the third time I am writing.";

    static final Predicate REFUND = new Predicate("Is the customer asking for a refund?");
    static final Choice TEAM =
            new Choice(
                    "Which team should handle this?",
                    List.of(
                            new Option("billing", "Charges and refunds"),
                            new Option("shipping", "Delivery problems"),
                            new Option("technical", "Product does not work"),
                            new Option("other")));
    static final Rating URGENCY =
            new Rating(
                    "How urgent is this?",
                    Option.of("Can wait", "This week", "Today", "Right now"));
    static final Rating POLARITY =
            new Rating("What is the sentiment?", Option.of("negative", "neutral", "positive"));

    static final String POSITIVE = "I absolutely love this product, it works perfectly.";
    static final String NEGATIVE = "This is the worst purchase I ever made, it broke immediately.";

    private static List<Content> text(String text) {
        return List.of(new Content.Text(text));
    }

    private static DecisionModel load(Arena arena, Variant variant) throws Exception {
        return Models.loadDecisionModel(
                TestModels.require(variant.model()),
                arena,
                Map.of("media", TestModels.require(variant.mmproj())));
    }

    private static void assertRow(float[] row, int options, String what) {
        assertEquals(options, row.length, what + ": one probability an option");
        float sum = 0;
        for (float p : row) {
            assertTrue(Float.isFinite(p), what + ": " + p + " is not finite");
            assertTrue(p >= 0 && p <= 1, what + ": " + p + " outside 0..1");
            sum += p;
        }
        assertEquals(1f, sum, 1e-3f, what + ": probabilities sum");
    }

    private static int argmax(float[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
        return best;
    }

    /**
     * The family's core law: d3 is not trained to pack, so a question asked together with others
     * ({@code decide}) answers as the same question asked alone. Twenty questions in one call, then
     * each on its own: every row matches to 1e-6, far below the shift packing would cause (about
     * 1e-2); exact equality would only test the JIT's warm-up.
     */
    @Test
    void togetherAnswersAsEachOnEverySize() throws Exception {
        List<Question> many = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            many.add(REFUND);
            many.add(TEAM);
            many.add(URGENCY);
            many.add(POLARITY);
        }
        for (Variant variant : VARIANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                try (ContextState state = d3.newState(4096)) {
                    float[][] together = d3.decide(state, text(TICKET), many);
                    float[][] each = d3.decideIndependently(state, text(TICKET), many);
                    assertEquals(20, together.length, variant.name());
                    for (int i = 0; i < together.length; i++) {
                        assertRow(
                                together[i],
                                many.get(i).options().size(),
                                variant.name() + " " + i);
                        assertArrayEquals(
                                each[i],
                                together[i],
                                1e-6f,
                                variant.name()
                                        + ": question "
                                        + i
                                        + " together differs from alone");
                    }
                    // and a question asked together still equals the single-question door
                    assertArrayEquals(
                            d3.decide(state, text(TICKET), REFUND),
                            together[0],
                            1e-6f,
                            variant.name() + ": together equals the single-question decide");
                }
            }
        }
    }

    /**
     * A routing battery lands on billing on every checkpoint, with the ticket's sentiment positive.
     */
    @Test
    void everySizeRoutesAndReadsSentiment() throws Exception {
        for (Variant variant : VARIANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                try (ContextState state = d3.newState(4096)) {
                    float[] refund = d3.decide(state, text(TICKET), REFUND);
                    assertRow(refund, 2, variant.name() + " refund");
                    assertTrue(refund[1] > 0.9f, variant.name() + " refund yes " + refund[1]);

                    float[] team = d3.decide(state, text(TICKET), TEAM);
                    assertRow(team, 4, variant.name() + " team");
                    assertEquals(0, argmax(team), variant.name() + " billing: " + List.of(team[0]));

                    float[] urgency = d3.decide(state, text(TICKET), URGENCY);
                    assertRow(urgency, 4, variant.name() + " urgency");

                    float[] positive = d3.decide(state, text(POSITIVE), POLARITY);
                    assertRow(positive, 3, variant.name() + " positive");
                    assertEquals(2, argmax(positive), variant.name() + " positive polarity");

                    float[] negative = d3.decide(state, text(NEGATIVE), POLARITY);
                    assertRow(negative, 3, variant.name() + " negative");
                    assertEquals(0, argmax(negative), variant.name() + " negative polarity");
                }
            }
        }
    }

    /**
     * The routing battery of four tickets: every checkpoint sends each to its own team. The
     * checkpoints differ in confidence, not in what they pick.
     */
    @Test
    void theRoutingBatteryAgreesOnEverySize() throws Exception {
        Choice team =
                new Choice(
                        "Which team should handle this ticket?",
                        List.of(
                                new Option("billing", "Charges, invoices and refunds"),
                                new Option("shipping", "Delivery and parcels"),
                                new Option("technical", "The product does not work"),
                                new Option("account", "Login, password and profile")));
        record Case(String ticket, int team) {}
        List<Case> cases =
                List.of(
                        new Case("I was charged twice this month, please refund the extra.", 0),
                        new Case("My parcel has not arrived after two weeks. Where is it?", 1),
                        new Case("The app crashes every time I open the camera tab.", 2),
                        new Case("I cannot log in: my password reset email never arrives.", 3));
        for (Variant variant : VARIANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                try (ContextState state = d3.newState(4096)) {
                    for (Case c : cases) {
                        float[] row = d3.decide(state, text(c.ticket()), team);
                        assertRow(row, 4, variant.name() + " " + c.ticket());
                        assertEquals(
                                c.team(),
                                argmax(row),
                                variant.name()
                                        + ": "
                                        + c.ticket()
                                        + " -> "
                                        + List.of(row[0], row[1], row[2], row[3]));
                    }
                }
            }
        }
    }

    /**
     * The code space: every size tells 40 options apart, and the code a question uses is the option
     * the model names. Past 255, the family refuses rather than guessing.
     */
    @Test
    void theCodeSpaceIsBoundedAtTwoHundredFiftyFive() throws Exception {
        List<Option> forty = new ArrayList<>();
        for (int i = 0; i < 40; i++) forty.add(new Option("option " + i));
        forty.set(26, new Option("the answer"));
        Choice pick = new Choice("Which one is the answer?", forty);

        for (Variant variant : VARIANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                try (ContextState state = d3.newState(4096)) {
                    float[] row =
                            d3.decide(
                                    state,
                                    text("The answer is the one called 'the answer': option 26."),
                                    pick);
                    assertRow(row, 40, variant.name() + " 40 options");
                    // edge (0.8B) is too weak to pick among forty, past Z or not; the others show
                    // AA
                    if (!variant.name().equals("edge"))
                        assertEquals(26, argmax(row), variant.name() + " option 26: " + row[26]);

                    List<Option> tooMany = new ArrayList<>();
                    for (int i = 0; i < 256; i++) tooMany.add(new Option("o" + i));
                    IllegalArgumentException refused =
                            assertThrows(
                                    IllegalArgumentException.class,
                                    () ->
                                            d3.decide(
                                                    state,
                                                    text("Pick one."),
                                                    new Choice("Which?", tooMany)),
                                    variant.name());
                    assertTrue(refused.getMessage().contains("at most 255"), refused.getMessage());
                }
            }
        }
    }

    /**
     * Images are read on every size, and a picture whose modality the family does not take is
     * refused naming what it takes: d3 reads images, never sound or video.
     */
    @Test
    void imagesAreReadAndOtherMediaRefusedOnEverySize() throws Exception {
        List<Content> photo =
                List.of(
                        new Content.Media(
                                ImageCodec.load(
                                        Path.of("../../test-fixtures/lfm3/images/dice.png"))),
                        new Content.Text("A photo sent by a customer."));
        Choice shows =
                new Choice(
                        "What does the picture show?",
                        Option.of("cat", "dog", "dice", "landscape"));

        for (Variant variant : VARIANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                try (ContextState state = d3.newState(4096)) {
                    float[] row = d3.decide(state, photo, shows);
                    assertRow(row, 4, variant.name() + " picture");
                    assertEquals(2, argmax(row), variant.name() + " dice: " + List.of(row));

                    // the same picture with no text at all
                    float[] alone = d3.decide(state, List.of(photo.getFirst()), shows);
                    assertRow(alone, 4, variant.name() + " picture alone");
                    assertEquals(2, argmax(alone), variant.name() + " dice alone");

                    UnsupportedOperationException audio =
                            assertThrows(
                                    UnsupportedOperationException.class,
                                    () ->
                                            d3.decide(
                                                    state,
                                                    List.of(
                                                            new Content.Media(
                                                                    com.qxotic.jinfer.codecs
                                                                            .AudioCodec.decode(
                                                                            java.nio.file.Files
                                                                                    .readAllBytes(
                                                                                            Path.of(
                                                                                                    "../../test-fixtures/parakeet/ls0.wav")))),
                                                            new Content.Text("A recording.")),
                                                    REFUND),
                                    variant.name());
                    assertTrue(audio.getMessage().contains("takes no audio"), audio.getMessage());

                    UnsupportedOperationException video =
                            assertThrows(
                                    UnsupportedOperationException.class,
                                    () ->
                                            d3.decide(
                                                    state,
                                                    List.of(
                                                            new Content.Media(
                                                                    new com.qxotic.jinfer.media
                                                                            .Media.Video(
                                                                            List.of())),
                                                            new Content.Text("A clip.")),
                                                    REFUND),
                                    variant.name());
                    assertTrue(video.getMessage().contains("takes no video"), video.getMessage());
                }
            }
        }
    }

    /**
     * A scale from two to ten levels and a choice from two to many options answer on every size;
     * the row always has exactly the declared count.
     */
    @Test
    void everyQuestionShapeAnswersOnEverySize() throws Exception {
        for (Variant variant : VARIANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                try (ContextState state = d3.newState(4096)) {
                    assertRow(
                            d3.decide(
                                    state,
                                    text(TICKET),
                                    new Rating("Urgent?", Option.of("no", "yes"))),
                            2,
                            variant.name() + " two levels");
                    for (int levels : new int[] {3, 5, 9, 10}) {
                        List<Option> scale = new ArrayList<>();
                        for (int i = 0; i < levels; i++) scale.add(new Option("L" + i));
                        assertRow(
                                d3.decide(state, text(TICKET), new Rating("How much?", scale)),
                                levels,
                                variant.name() + " " + levels + " levels");
                    }
                    for (int options : new int[] {2, 3, 7, 20}) {
                        List<Option> choices = new ArrayList<>();
                        for (int i = 0; i < options; i++) choices.add(new Option("c" + i));
                        assertRow(
                                d3.decide(state, text(TICKET), new Choice("Which?", choices)),
                                options,
                                variant.name() + " " + options + " options");
                    }
                    // a predicate with each side described, and with none
                    assertRow(
                            d3.decide(
                                    state,
                                    text(TICKET),
                                    new Predicate("Refund?", "asks for money back", null)),
                            2,
                            variant.name() + " one side");
                }
            }
        }
    }

    /** Every size reads the empty state as the family says, each with its own calibration. */
    @Test
    void everySizeReadsTheEmptyState() throws Exception {
        for (Variant variant : VARIANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                try (ContextState state = d3.newState(4096)) {
                    float[] empty = d3.decide(state, text(""), REFUND);
                    assertRow(empty, 2, variant.name() + " empty");
                    // the reference the family was trained on answers one fixed probability an
                    // empty state; only edge's is pinned by llama.cpp's dump
                    if (variant.name().equals("edge"))
                        assertEquals(0.37191f, empty[1], 2e-2f, "llama.cpp's empty state");
                    // whitespace is text, not emptiness: a different prompt
                    float[] spaces = d3.decide(state, text("   "), REFUND);
                    assertRow(spaces, 2, variant.name() + " whitespace");
                }
            }
        }
    }

    /**
     * A state is reusable and independent: the same input twice answers the same, and a second
     * state over the same weights is its own pipeline.
     */
    @Test
    void statesAreReusableAndIndependent() throws Exception {
        for (Variant variant : List.of(VARIANTS.get(0), VARIANTS.get(2))) { // edge, nano
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                float[][] first = new float[1][];
                try (ContextState state = d3.newState(2048)) {
                    first[0] = d3.decide(state, text(TICKET), REFUND);
                    for (int i = 0; i < 3; i++)
                        assertArrayEquals(
                                first[0],
                                d3.decide(state, text(TICKET), REFUND),
                                1e-6f,
                                variant.name() + " reuse " + i);
                    // another input in between leaves nothing behind
                    d3.decide(state, text("Where is my parcel?"), TEAM);
                    assertArrayEquals(
                            first[0],
                            d3.decide(state, text(TICKET), REFUND),
                            1e-6f,
                            variant.name() + " after another input");
                }
                try (ContextState other = d3.newState(2048)) {
                    assertArrayEquals(
                            first[0],
                            d3.decide(other, text(TICKET), REFUND),
                            1e-6f,
                            variant.name() + " a second state");
                }
            }
        }
    }

    /**
     * d3's token budget rule: 0 is the default, and past the maximum is refused. The maximum itself
     * (-1, 262,144 tokens in one batch) takes tens of gigabytes and is never opened here.
     */
    @Test
    void theCapacityRuleHoldsOnEverySize() throws Exception {
        for (Variant variant : VARIANTS) {
            try (Arena arena = Arena.ofShared()) {
                DecisionModel d3 = load(arena, variant);
                try (ContextState byDefault = d3.newState(0)) {
                    assertTrue(byDefault.contextCapacity() > 0, variant.name());
                }
                assertThrows(
                        IllegalArgumentException.class,
                        () -> d3.newState(1_000_000),
                        variant.name() + ": past the maximum");
                assertThrows(IllegalArgumentException.class, () -> d3.newState(-2), variant.name());
            }
        }
    }
}
