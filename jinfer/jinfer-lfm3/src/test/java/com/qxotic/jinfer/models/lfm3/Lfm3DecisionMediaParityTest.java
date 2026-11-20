package com.qxotic.jinfer.models.lfm3;

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
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * d1-2B and d1-Omni-2B with their towers against llama.cpp's {@code /v1/systemone} on the same
 * files (Q8_0 and the F16 mmproj, llama.cpp 2026-11-10): pictures, clips and a video ahead of the
 * text, the probabilities llama.cpp answered. The fixtures are {@code test-fixtures/lfm3/images},
 * {@code test-fixtures/audio} and {@code test-fixtures/lfm3/video}: the six frames llama.cpp
 * samples of a 12-frame clip at 4 frames a second.
 */
@Tag("integration")
final class Lfm3DecisionMediaParityTest {

    static final String D1 = Lfm3DecisionParityTest.D1;
    static final String MMPROJ = "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf";
    static final String OMNI = "hf.co/LiquidAI/d1-Omni-2B-GGUF/d1-Omni-2B-Q8_0.gguf";
    static final String OMNI_MMPROJ = "hf.co/LiquidAI/d1-Omni-2B-GGUF/mmproj-d1-Omni-2B-F16.gguf";

    /** Two engines' Q8_0 kernels, and their towers: {@link Lfm3DecisionParityTest#TOLERANCE}. */
    static final float TOLERANCE = Lfm3DecisionParityTest.TOLERANCE;

    static final Content PHOTO = new Content.Text("A photo sent by a customer.");
    static final Content RECORDING = new Content.Text("A recording sent by a customer.");

    static final Predicate CAT = new Predicate("Is there a cat in the picture?");
    static final List<Option> THINGS =
            List.of(
                    new Option("cat"),
                    new Option("dog"),
                    new Option("dice"),
                    new Option("landscape"));
    static final Choice SHOWS = new Choice("What does the picture show?", THINGS);
    static final Rating SHARP =
            new Rating(
                    "How sharp is the picture?",
                    List.of(new Option("Blurry"), new Option("Fair"), new Option("Sharp")));

    static final Predicate SPEECH = new Predicate("Is someone speaking in the recording?");
    static final Choice HOLDS =
            new Choice(
                    "What does the recording hold?",
                    List.of(
                            new Option("speech"),
                            new Option("music"),
                            new Option("a_steady_tone"),
                            new Option("silence")));
    static final Rating CLEAR =
            new Rating(
                    "How clear is the recording?",
                    List.of(new Option("Poor"), new Option("Fair"), new Option("Clear")));

    static final Predicate BOTH = new Predicate("Does the video show both a cat and dice?");
    static final Choice FIRST = new Choice("What does the video show first?", THINGS);
    static final Choice LAST = new Choice("What does the video show at the end?", THINGS);
    static final Rating MOVING =
            new Rating(
                    "How much does the camera move?",
                    List.of(new Option("Not at all"), new Option("A little"), new Option("A lot")));

    private static DecisionModel load(Arena arena, String model, String mmproj) throws Exception {
        return Models.loadDecisionModel(
                TestModels.require(model), arena, Map.of("media", TestModels.require(mmproj)));
    }

    private static Media.Image image(String name) throws Exception {
        return ImageCodec.load(TestModels.fixture("lfm3/images/" + name + ".png"));
    }

    private static Content picture(String name) throws Exception {
        return new Content.Media(image(name));
    }

    private static Content clip(String path) throws Exception {
        return new Content.Media(Lfm3AudioParityTest.wav(TestModels.fixture(path)));
    }

    /** The frames llama.cpp samples of the clip: {@code round(linspace(0, 11, 6))}. */
    private static Content video() throws Exception {
        List<Media.Video.Frame> frames = new ArrayList<>();
        for (int index : new int[] {0, 2, 4, 7, 9, 11})
            frames.add(
                    new Media.Video.Frame(
                            ImageCodec.load(
                                    TestModels.fixture(
                                            String.format("lfm3/video/clip-%02d.png", index))),
                            Duration.ofMillis(index * 250)));
        return new Content.Media(new Media.Video(frames));
    }

    private static void check(
            DecisionModel model,
            ContextState state,
            String name,
            List<Content> input,
            List<? extends Question> questions,
            float[]... expected) {
        float[][] got = model.decideIndependently(state, input, questions);
        for (int i = 0; i < expected.length; i++) {
            // a predicate is recorded as its yes
            float[] row = expected[i].length == 1 ? new float[] {got[i][1]} : got[i];
            assertArrayEquals(expected[i], row, TOLERANCE, name + ", question " + i);
        }
    }

    private static float[] p(float... values) {
        return values;
    }

    @Test
    void picturesMatchLlamaCpp() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = load(arena, D1, MMPROJ);
            List<Question> three = List.of(CAT, SHOWS, SHARP);
            try (ContextState state = d1.newState(4096)) {
                check(
                        d1,
                        state,
                        "cat",
                        List.of(picture("cat"), PHOTO),
                        three,
                        p(0.98768f),
                        p(0.99832f, 0.00109f, 0.00017f, 0.00042f),
                        p(0.19020f, 0.70843f, 0.10137f));
                check(
                        d1,
                        state,
                        "dice (RGBA)",
                        List.of(picture("dice"), PHOTO),
                        three,
                        p(0.00525f),
                        p(0.00037f, 0.00048f, 0.99865f, 0.00051f),
                        p(0.48925f, 0.41759f, 0.09316f));
                check(
                        d1,
                        state,
                        "small, scaled up to the tower's least",
                        List.of(picture("small"), PHOTO),
                        three,
                        p(0.01484f),
                        p(0.09808f, 0.02916f, 0.12493f, 0.74783f),
                        p(0.53032f, 0.17717f, 0.29250f));
                check(
                        d1,
                        state,
                        "wide",
                        List.of(picture("wide"), PHOTO),
                        three,
                        p(0.01301f),
                        p(0.05023f, 0.02956f, 0.29143f, 0.62878f),
                        p(0.45194f, 0.25120f, 0.29687f));
                check(
                        d1,
                        state,
                        "two pictures, in their order",
                        List.of(
                                picture("dice"),
                                picture("cat"),
                                new Content.Text("Two photos sent by a customer.")),
                        List.of(
                                new Choice(
                                        "What does the first picture show?", THINGS.subList(0, 3)),
                                new Choice(
                                        "What does the second picture show?",
                                        THINGS.subList(0, 3))),
                        p(0.00139f, 0.00068f, 0.99793f),
                        p(0.99726f, 0.00063f, 0.00211f));
                check(
                        d1,
                        state,
                        "a picture and an empty text",
                        List.of(picture("cat"), new Content.Text("")),
                        List.of(CAT, SHOWS),
                        p(0.99122f),
                        p(0.99795f, 0.00145f, 0.00017f, 0.00044f));
                check(
                        d1,
                        state,
                        "a picture and no text: the question follows it directly",
                        List.of(picture("cat")),
                        List.of(CAT, SHOWS),
                        p(0.99076f),
                        p(0.99800f, 0.00139f, 0.00018f, 0.00044f));
                check(
                        d1,
                        state,
                        "video",
                        List.of(video(), new Content.Text("A clip sent by a customer.")),
                        List.of(BOTH, FIRST, LAST, MOVING),
                        p(0.98811f),
                        p(0.97993f, 0.00097f, 0.01785f, 0.00125f),
                        p(0.01378f, 0.00208f, 0.98170f, 0.00244f),
                        p(0.46570f, 0.40343f, 0.13087f));
                check(
                        d1,
                        state,
                        "a video and no text",
                        List.of(video()),
                        List.of(FIRST),
                        p(0.98985f, 0.00070f, 0.00865f, 0.00081f));
            }
        }
    }

    @Test
    void omniMatchesLlamaCpp() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            DecisionModel omni = load(arena, OMNI, OMNI_MMPROJ);
            List<Question> heard = List.of(SPEECH, HOLDS, CLEAR);
            try (ContextState state = omni.newState(4096)) {
                check(
                        omni,
                        state,
                        "text",
                        List.of(new Content.Text(Lfm3DecisionParityTest.STATE)),
                        List.of(Lfm3DecisionParityTest.REFUND, Lfm3DecisionParityTest.URGENCY),
                        p(0.95194f),
                        p(0.02427f, 0.02224f, 0.63657f, 0.31692f));
                check(
                        omni,
                        state,
                        "picture",
                        List.of(picture("cat"), PHOTO),
                        List.of(CAT, SHOWS),
                        p(0.93478f),
                        p(0.95280f, 0.02622f, 0.00740f, 0.01358f));
                check(
                        omni,
                        state,
                        "speech",
                        List.of(clip("parakeet/ls0.wav"), RECORDING),
                        heard,
                        p(0.53227f),
                        p(0.74172f, 0.11477f, 0.07548f, 0.06802f),
                        p(0.06402f, 0.40587f, 0.53011f));
                check(
                        omni,
                        state,
                        "a tone",
                        List.of(clip("audio/sine440-3s.wav"), RECORDING),
                        heard,
                        p(0.45412f),
                        p(0.11440f, 0.06767f, 0.21295f, 0.60498f),
                        p(0.51821f, 0.31553f, 0.16626f));
                check(
                        omni,
                        state,
                        "an odd length",
                        List.of(clip("audio/odd-1p37s.wav"), RECORDING),
                        heard,
                        p(0.05986f),
                        p(0.01159f, 0.06599f, 0.85738f, 0.06503f),
                        p(0.60378f, 0.23192f, 0.16430f));
                check(
                        omni,
                        state,
                        "35 seconds",
                        List.of(clip("audio/long-35s.wav"), RECORDING),
                        heard,
                        p(0.06282f),
                        p(0.01507f, 0.05169f, 0.86264f, 0.07059f),
                        p(0.37687f, 0.35055f, 0.27258f));
                check(
                        omni,
                        state,
                        "a clip and no text",
                        List.of(clip("parakeet/ls0.wav")),
                        heard,
                        p(0.50960f),
                        p(0.57873f, 0.15026f, 0.17728f, 0.09373f),
                        p(0.02535f, 0.41602f, 0.55863f));
                check(
                        omni,
                        state,
                        "a picture and a clip",
                        List.of(
                                picture("cat"),
                                clip("parakeet/ls0.wav"),
                                new Content.Text("A photo and a recording.")),
                        List.of(CAT, SHOWS, SPEECH),
                        p(0.90702f),
                        p(0.95365f, 0.02240f, 0.00558f, 0.01837f),
                        p(0.36273f));
                // asked together, the clip is read once for the three: the same picks
                List<Content> speech = List.of(clip("parakeet/ls0.wav"), RECORDING);
                float[][] each = omni.decideIndependently(state, speech, heard);
                float[][] together = omni.decide(state, speech, heard);
                for (int i = 1; i < heard.size(); i++) // the yes/no is near even: no pick to keep
                assertEquals(argmax(each[i]), argmax(together[i]), "question " + i);

                check(
                        omni,
                        state,
                        "video: its frames, as the reference takes it",
                        List.of(video(), new Content.Text("A clip sent by a customer.")),
                        List.of(BOTH, FIRST, LAST, MOVING),
                        p(0.91751f),
                        p(0.75687f, 0.01658f, 0.21341f, 0.01314f),
                        p(0.08410f, 0.01587f, 0.88293f, 0.01709f),
                        p(0.44874f, 0.45760f, 0.09366f));
            }
        }
    }

    /** What holds whatever the reference: where the parts sit, how they are asked, what fits. */
    @Test
    void mediaLaws() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            DecisionModel d1 = load(arena, D1, MMPROJ);
            List<Question> three = List.of(CAT, SHOWS, SHARP);
            try (ContextState state = d1.newState(4096)) {
                List<Content> input = List.of(picture("cat"), PHOTO);
                float[][] each = d1.decideIndependently(state, input, three);
                for (int i = 0; i < three.size(); i++)
                    assertArrayEquals(
                            d1.decide(state, input, three.get(i)), each[i], 1e-4f, "alone " + i);

                // the checkpoint reads its media first: where a picture sits among the text is not
                // read
                assertArrayEquals(
                        each[0], d1.decide(state, List.of(PHOTO, picture("cat")), CAT), 1e-4f);

                // together the picture is read once for the three; one question alone is the same
                float[][] together = d1.decide(state, input, three);
                for (int i = 0; i < three.size(); i++) {
                    float sum = 0;
                    for (float v : together[i]) sum += v;
                    assertEquals(1f, sum, 1e-4f);
                    assertEquals(argmax(each[i]), argmax(together[i]), "question " + i);
                }
                assertArrayEquals(each[0], d1.decide(state, input, List.of(CAT))[0], 1e-4f);

                // a model with no audio tower says so; so does a video without frames
                assertTrue(
                        assertThrows(
                                        UnsupportedOperationException.class,
                                        () ->
                                                d1.decide(
                                                        state,
                                                        List.of(clip("audio/sine440-3s.wav")),
                                                        SPEECH))
                                .getMessage()
                                .contains("takes no audio"));
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                d1.decide(
                                        state,
                                        List.of(new Content.Media(new Media.Video(List.of()))),
                                        CAT));
            }

            // a video of an odd count reads its last frame twice: as if it were given twice
            try (ContextState state = d1.newState(4096)) {
                List<Media.Video.Frame> six =
                        ((Media.Video) ((Content.Media) video()).media()).frames();
                List<Media.Video.Frame> five = six.subList(0, 5);
                List<Media.Video.Frame> doubled = new ArrayList<>(five);
                doubled.add(five.get(4));
                assertArrayEquals(
                        d1.decide(
                                state, List.of(new Content.Media(new Media.Video(doubled))), FIRST),
                        d1.decide(state, List.of(new Content.Media(new Media.Video(five))), FIRST),
                        1e-5f);
                // its frames are of one size
                List<Media.Video.Frame> mixed = new ArrayList<>(five);
                mixed.add(new Media.Video.Frame(image("small"), Duration.ofSeconds(3)));
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                d1.decide(
                                        state,
                                        List.of(new Content.Media(new Media.Video(mixed))),
                                        FIRST));
            }

            // a picture over the checkpoint's 1,638,400 pixels is scaled down to them: the cat
            // three times as large, 2.8 megapixels, takes 1,600 rows at most, not its 2,744
            Media.Image cat = image("cat");
            Media.Image large = scaled(cat, 3);
            try (ContextState state = d1.newState(1700)) {
                float[] seen = d1.decide(state, List.of(new Content.Media(large), PHOTO), CAT);
                assertTrue(seen[1] > 0.9f, "the cat, large: " + seen[1]);
            }
            // what does not fit is refused: the picture before its tower runs, the prompt after
            try (ContextState small = d1.newState(256)) {
                IllegalArgumentException refused =
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        d1.decide(
                                                small,
                                                List.of(new Content.Media(cat), PHOTO),
                                                CAT));
                assertTrue(
                        refused.getMessage().contains("media take 304 tokens"),
                        refused.getMessage());
            }
            try (ContextState tight = d1.newState(320)) {
                IllegalArgumentException refused =
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        d1.decide(
                                                tight,
                                                List.of(new Content.Media(cat), PHOTO),
                                                CAT));
                assertTrue(
                        refused.getMessage().contains("304 of them its media"),
                        refused.getMessage());
            }
        }
    }

    @Test
    void mediaNeedTheTowers() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            DecisionModel bare = Models.loadDecisionModel(TestModels.require(D1), arena);
            try (ContextState state = bare.newState(256)) {
                UnsupportedOperationException refused =
                        assertThrows(
                                UnsupportedOperationException.class,
                                () -> bare.decide(state, List.of(picture("cat"), PHOTO), CAT));
                assertTrue(refused.getMessage().contains("mmproj"), refused.getMessage());
            }
        }
    }

    private static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    /** {@code image} with every pixel {@code times} wide and high. */
    private static Media.Image scaled(Media.Image image, int times) {
        int w = image.width(), h = image.height(), c = image.channels();
        float[] out = new float[w * times * h * times * c];
        for (int y = 0; y < h * times; y++)
            for (int x = 0; x < w * times; x++)
                System.arraycopy(
                        image.values(),
                        ((y / times) * w + x / times) * c,
                        out,
                        (y * w * times + x) * c,
                        c);
        return new Media.Image(out, h * times, w * times, c);
    }
}
