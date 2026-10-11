package com.qxotic.jinfer.models.gemmaembedding2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.MediaProjector;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Parity with the official pipeline (sentence-transformers over {@code google/embeddinggemma-2}):
 * pooled embeddings and media projector rows against a reference directory ({@code
 * -Djinfer.gemmaEmbedding2.reference}) that {@code scripts/gemma-embedding2-reference.py} writes:
 * {@code cases.tsv} (name, text with {@code {m}} media markers, comma-separated media files),
 * {@code <name>.emb} (the L2-normalized pooled vector, little-endian f32) and, for single-media
 * cases, {@code <name>.proj} (i32 rows, i32 width, f32 rows). The reference runs in float32, so run
 * this against F32 weights ({@code -Djinfer.gemmaEmbedding2.model}/{@code .mmproj}) to isolate the
 * port from quantization noise; with quantized weights relax the cosine gates ({@code .textCosine},
 * {@code .cosine}) to 0.999.
 */
@Tag("integration")
final class GemmaEmbedding2ParityTest {
    static final String MODEL_REF =
            "hf.co/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf";
    static final String MMPROJ_REF = "hf.co/unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf";

    @Test
    void embeddingsAndProjectionsMatchTheReference() throws IOException {
        String dir = System.getProperty("jinfer.gemmaEmbedding2.reference");
        assumeTrue(dir != null, "no reference: -Djinfer.gemmaEmbedding2.reference");
        Path reference = Path.of(dir);
        // F32 text agrees to rounding (a window off by one key already costs 1e-6); images differ
        // by Pillow's bicubic against torchvision's (2/255 on 0.2% of pixels)
        double textCosine = gate("textCosine", "0.999999"), mediaCosine = gate("cosine", "0.99999");
        try (Arena arena = Arena.ofShared()) {
            GemmaEmbedding2 model =
                    GemmaEmbedding2.loadModel(path("model", MODEL_REF), arena)
                            .withMedia(path("mmproj", MMPROJ_REF), arena);
            try (GemmaEmbedding2.State state = model.newState(8_192, 512)) {
                for (String line : Files.readAllLines(reference.resolve("cases.tsv"))) {
                    String[] cell = line.split("\t", -1);
                    List<Media> media = new ArrayList<>();
                    if (!cell[2].isEmpty())
                        for (String file : cell[2].split(",")) media.add(decode(Path.of(file)));
                    Path dump = reference.resolve(cell[0] + ".proj");
                    // a video's frames are framed one by one: its pooled embedding is the check
                    if (Files.exists(dump) && !(media.getFirst() instanceof Media.Video))
                        assertProjection(cell[0], model, media.getFirst(), dump);

                    float[] expected = floats(reference.resolve(cell[0] + ".emb"), 0);
                    float[] actual = embed(model, state, cell[1], media);
                    double cosine = cosine(expected, actual);
                    System.out.printf(
                            "%-9s embedding: cosine=%.9f max|diff|=%.3g%n",
                            cell[0], cosine, maxDiff(expected, actual));
                    assertTrue(
                            cosine > (media.isEmpty() ? textCosine : mediaCosine),
                            cell[0] + " cosine " + cosine);
                }
            }
        }
    }

    /** The medium's projector rows against the reference's: same count, every row aligned. */
    private static void assertProjection(String name, GemmaEmbedding2 model, Media media, Path dump)
            throws IOException {
        ByteBuffer header =
                ByteBuffer.wrap(Files.readAllBytes(dump), 0, 8).order(ByteOrder.LITTLE_ENDIAN);
        int rows = header.getInt(), width = header.getInt();
        float[] expected = floats(dump, 8);
        float[] actual = project(model, media);
        assertEquals(rows * width, actual.length, name + " projected rows x width");
        double worstCosine = 1;
        for (int r = 0; r < rows; r++)
            worstCosine =
                    Math.min(
                            worstCosine,
                            cosine(
                                    Arrays.copyOfRange(expected, r * width, (r + 1) * width),
                                    Arrays.copyOfRange(actual, r * width, (r + 1) * width)));
        double cosine = cosine(expected, actual);
        System.out.printf(
                "%-9s projector: %d rows, cosine=%.9f worst row=%.6f max|diff|=%.3g%n",
                name, rows, cosine, worstCosine, maxDiff(expected, actual));
        // a ViT's few outlier tokens amplify the resamplers' pixel differences over 16 layers (on
        // identical pixels the rows agree to 1e-7): the whole projection is the tight gate, a
        // single row only has to stay aligned
        assertTrue(cosine > 0.99995, name + " projection cosine " + cosine);
        assertTrue(worstCosine > 0.99, name + " worst projector row cosine " + worstCosine);
    }

    @SuppressWarnings("unchecked")
    private static float[] project(GemmaEmbedding2 model, Media media) {
        MediaProjector<Media> projector =
                model.projector((Class<Media>) media.getClass()).orElseThrow();
        float[] all =
                new float[projector.positions(media) * model.configuration().embeddingLength()];
        int[] at = {0};
        projector.project(
                media,
                Integer.MAX_VALUE,
                rows -> {
                    int n = Math.toIntExact(rows.shape().size());
                    Views.copyToArray(
                            Views.castToSegmentBacked(rows, "rows"), 0, all, at[0], n, "rows");
                    at[0] += n;
                });
        return all;
    }

    /** The text cut at each {@code {m}} marker, with the next medium in between. */
    private static float[] embed(
            GemmaEmbedding2 model, GemmaEmbedding2.State state, String text, List<Media> media) {
        List<Object> parts = new ArrayList<>();
        String[] pieces = text.split("\\{m}", -1);
        for (int i = 0; i < pieces.length; i++) {
            if (!pieces[i].isEmpty()) parts.add(pieces[i]);
            if (i < media.size()) parts.add(media.get(i));
        }
        return embed(model, state, parts);
    }

    static float[] embed(GemmaEmbedding2 model, GemmaEmbedding2.State state, List<?> parts) {
        float[][] out = new float[1][];
        model.embed(state, parts, view -> out[0] = toFloats(view));
        return out[0];
    }

    private static float[] toFloats(MemoryView<?> view) {
        return Views.toFloatArray(Views.castToSegmentBacked(view, "embedding"), "embedding");
    }

    /** A WAV, an image, or {@code +}-joined frames as a video sampled one a second. */
    private static Media decode(Path file) throws IOException {
        String name = file.toString();
        if (name.contains("+")) {
            List<Media.Video.Frame> frames = new ArrayList<>();
            for (String frame : name.split("\\+"))
                frames.add(
                        new Media.Video.Frame(
                                ImageCodec.load(Path.of(frame)),
                                Duration.ofSeconds(frames.size())));
            return new Media.Video(frames);
        }
        return name.endsWith(".wav") ? AudioCodec.load(file) : ImageCodec.load(file);
    }

    private static double gate(String name, String fallback) {
        return Double.parseDouble(System.getProperty("jinfer.gemmaEmbedding2." + name, fallback));
    }

    static Path path(String what, String ref) {
        String override = System.getProperty("jinfer.gemmaEmbedding2." + what);
        return override != null ? Path.of(override) : TestModels.require(ref);
    }

    private static float[] floats(Path file, int skip) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        bytes.position(skip);
        float[] values = new float[bytes.remaining() / Float.BYTES];
        bytes.asFloatBuffer().get(values);
        return values;
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            aa += (double) a[i] * a[i];
            bb += (double) b[i] * b[i];
        }
        return dot / Math.sqrt(aa * bb);
    }

    static double maxDiff(float[] a, float[] b) {
        double max = 0;
        for (int i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
        return max;
    }
}
