package com.qxotic.jinfer.models.gemmaembedding2;

import static com.qxotic.jinfer.models.gemmaembedding2.GemmaEmbedding2.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.codecs.VideoCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The README's task examples, run: each task with its prompt on the text and images, audio and
 * video bare, as the card prescribes, against real fixtures. Every assertion is a ranking a user
 * would rely on. Model-gated via TestModels; the video example needs ffmpeg.
 */
@Tag("integration")
class GemmaEmbedding2ExamplesTest {

    static final Path IMAGES = Path.of("../jinfer-langchain4j/src/test/resources/kit-images");

    static Arena weights;
    static LoadedEmbedder<?> embedder;
    static ContextState state;
    static Media.Image cat, dice;
    static Media.Audio speech; // "Hello world one two three four five"

    @BeforeAll
    static void load() throws IOException {
        weights = Arena.ofShared();
        embedder =
                Models.loadEmbedder(
                        GemmaEmbedding2ParityTest.path(
                                "model", GemmaEmbedding2ParityTest.MODEL_REF),
                        weights,
                        Map.of(
                                "media",
                                GemmaEmbedding2ParityTest.path(
                                        "mmproj", GemmaEmbedding2ParityTest.MMPROJ_REF)),
                        null);
        state = embedder.model().newState(8192);
        cat = ImageCodec.load(IMAGES.resolve("cat.png"));
        dice = ImageCodec.load(IMAGES.resolve("dice.png"));
        speech = AudioCodec.load(TestModels.fixture("audio/speech-tts.wav"));
    }

    @AfterAll
    static void close() {
        if (state != null) state.close();
        if (weights != null) weights.close();
    }

    /** One vector for one input: Strings and media, in order. */
    static float[] embed(Object... parts) {
        float[][] v = new float[1][];
        embedder.embed(state, List.of(parts), embedder.dimension(), x -> v[0] = x);
        return v[0];
    }

    /** Vectors are L2-normalized: the dot product is the cosine. */
    static float score(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    /** The index of the item scoring highest against the query. */
    static int best(float[] query, List<float[]> items) {
        int best = 0;
        for (int i = 1; i < items.size(); i++)
            if (score(query, items.get(i)) > score(query, items.get(best))) best = i;
        return best;
    }

    @Test
    void searchOneTextQueryOverAMixedIndex(@TempDir Path dir) throws IOException {
        List<float[]> index =
                List.of(
                        embed(document("Trip notes", "We hiked to the glacier at sunrise.")),
                        embed(cat),
                        embed(speech),
                        embed(VideoCodec.ffmpeg().every(video(dir), Duration.ofSeconds(1), 32)));

        assertEquals(0, best(embed(searchQuery("a glacier hike")), index));
        assertEquals(1, best(embed(searchQuery("a cat")), index));
        assertEquals(2, best(embed(searchQuery("a person counting from one to five")), index));
        assertEquals(3, best(embed(searchQuery("dice")), index));
    }

    @Test
    void questionAnsweringOverPageScans() throws IOException {
        List<float[]> pages =
                List.of(
                        embed(
                                page(
                                        "Quarterly report",
                                        "Revenue in the third quarter",
                                        "was 4.2 million euros.")),
                        embed(
                                page(
                                        "Staff notice",
                                        "The office picnic is on Friday",
                                        "at noon in the park.")),
                        embed(
                                page(
                                        "Cafeteria menu",
                                        "Tomato soup, fresh bread",
                                        "and apple pie.")));

        assertEquals(0, best(embed(question("What was the revenue in the third quarter?")), pages));
        assertEquals(1, best(embed(question("When is the office picnic?")), pages));
        assertEquals(2, best(embed(question("What dessert is served today?")), pages));
    }

    @Test
    void factCheckingAgainstPhotosAndReports() {
        List<float[]> evidence =
                List.of(
                        embed(cat),
                        embed(dice),
                        embed(
                                document(
                                        "City report",
                                        "The bridge reopened in March after repairs.")));

        assertEquals(2, best(embed(claim("The bridge was closed all winter.")), evidence));
        assertEquals(0, best(embed(claim("The photo shows a cat.")), evidence));
    }

    @Test
    void mediaAsTheQuery() {
        List<float[]> entries =
                List.of(
                        embed(document("Cat", "A small domesticated feline with soft fur.")),
                        embed(document("Dice", "Small cubes with numbered faces, used in games.")),
                        embed(document("Glacier", "A slowly moving mass of ice.")));
        assertEquals(0, best(embed(cat), entries));
        assertEquals(1, best(embed(dice), entries));

        List<float[]> transcripts =
                List.of(
                        embed(document(null, "The weather will be sunny tomorrow.")),
                        embed(document(null, "Hello world, one two three four five.")));
        assertEquals(1, best(embed(speech), transcripts));
    }

    @Test
    void classificationOfImagesAgainstTextLabels() {
        List<float[]> labels =
                List.of(
                        embed(classification("a photo of a cat")),
                        embed(classification("a photo of dice")),
                        embed(classification("a photo of a mountain")));

        assertEquals(0, best(embed(cat), labels));
        assertEquals(1, best(embed(dice), labels));
    }

    @Test
    void clusteringCaptionsAndPhotosTogether() {
        List<float[]> captions =
                List.of(
                        embed(clustering("A cat sitting and looking at the camera.")),
                        embed(clustering("A pair of dice on a table.")));

        // a photo lands next to its own caption: one cluster each
        assertEquals(0, best(embed(cat), captions));
        assertEquals(1, best(embed(dice), captions));
    }

    @Test
    void sentenceSimilarity() {
        float[] guitar = embed(similarity("A man is playing a guitar."));
        float[] strums = embed(similarity("Someone strums an instrument."));
        float[] stocks = embed(similarity("The stock market fell today."));

        assertTrue(score(guitar, strums) > score(guitar, stocks) + 0.1f);
    }

    @Test
    void interleavedTextAndMediaMakeOneVector() {
        // one input, no prompt, as in the card's example
        List<float[]> listings =
                List.of(
                        embed("Our pet of the month: ", cat, " Ready for adoption."),
                        embed("Game night supplies: ", dice, " Sold in pairs."));

        assertEquals(0, best(embed(searchQuery("a cat to adopt")), listings));
        assertEquals(1, best(embed(searchQuery("board game accessories")), listings));
    }

    /** A three-second clip of the dice photo, as a recorded video would arrive. */
    static Path video(Path dir) throws IOException {
        Path mp4 = dir.resolve("dice.mp4");
        try {
            Process ffmpeg =
                    new ProcessBuilder(
                                    "ffmpeg",
                                    "-loglevel",
                                    "error",
                                    "-y",
                                    "-loop",
                                    "1",
                                    "-i",
                                    IMAGES.resolve("dice.png").toString(),
                                    "-t",
                                    "3",
                                    "-vf",
                                    "scale=640:-2,format=yuv420p",
                                    "-r",
                                    "5",
                                    mp4.toString())
                            .inheritIO()
                            .start();
            assertEquals(0, ffmpeg.waitFor(), "ffmpeg");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(e);
        }
        return mp4;
    }

    /** A scanned page: a heading and two lines of text, black on white. */
    static Media.Image page(String heading, String... lines) throws IOException {
        BufferedImage image = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        g.setRenderingHint(
                RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 800, 600);
        g.setColor(Color.BLACK);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 48));
        g.drawString(heading, 60, 120);
        g.setFont(new Font(Font.SANS_SERIF, Font.PLAIN, 36));
        for (int i = 0; i < lines.length; i++) g.drawString(lines[i], 60, 240 + 60 * i);
        g.dispose();
        ByteArrayOutputStream png = new ByteArrayOutputStream();
        ImageIO.write(image, "png", png);
        return ImageCodec.decode(png.toByteArray());
    }
}
