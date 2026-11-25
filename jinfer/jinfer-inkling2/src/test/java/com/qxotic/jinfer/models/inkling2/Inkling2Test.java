package com.qxotic.jinfer.models.inkling2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.CheckpointCodec;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.MediaProjectorContract;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The whole forward pass and the vision tower on {@link TinyInkling2}: no model file needed. */
final class Inkling2Test {
    /** One token per byte, longer than the window and the bias extent. */
    private static final String PROMPT =
            "Hello world, this is a tiny test of the looped model. It keeps going past the window!";

    private static Arena arena;
    private static LoadedModel<?> loaded;
    private static Inkling2 model;
    private static Inkling2Vision vision;
    private static ContentKey seed;
    private static int[] tokens;

    @BeforeAll
    static void load(@TempDir Path directory) throws Exception {
        arena = Arena.ofShared();
        loaded =
                Models.load(
                        TinyInkling2.write(directory.resolve("tiny-inkling2.gguf")),
                        arena,
                        Map.of(
                                "media",
                                TinyInkling2.writeVision(directory.resolve("mmproj.gguf"))));
        seed = loaded.seed();
        model = assertInstanceOf(Inkling2.class, loaded.model());
        vision =
                assertInstanceOf(
                        Inkling2Vision.class, model.projector(Media.Image.class).orElseThrow());
        assertInstanceOf(Inkling2ChatTemplate.class, loaded.template().orElseThrow());
        assertEquals(
                "<|content_model_end_sampling|>",
                loaded.tokenizer().vocabulary().token(loaded.stopTokens().iterator().next()));
        tokens = model.tokenizer().encodeToArray(PROMPT);
        byte[] bytes = PROMPT.getBytes(StandardCharsets.US_ASCII);
        for (int i = 0; i < bytes.length; i++) assertEquals(bytes[i], tokens[i], "token " + i);
    }

    @AfterAll
    static void close() {
        arena.close();
    }

    @Test
    void theLoopRepeatsItsLayersOverTheirOwnCacheSlots() {
        Inkling2.Configuration c = model.configuration();
        assertEquals(16, c.slots());
        assertEquals(1, c.slot(0, 1));
        assertEquals(3, c.slot(0, 3));
        assertEquals(8, c.slot(1, 2));
        assertEquals(15, c.slot(2, 5));
        assertEquals(7, c.slot(2, 7), "the layers after the loop keep their own slot");
    }

    /**
     * llama.cpp cbb1c1c50 on the same file and prompt, {@code llama-debug --save-logits -fa off
     * -ctk f32 -ctv f32}: FP32 history on both sides, as one batch attends here.
     */
    @Test
    void matchesLlamaCpp() {
        int[] ids = {202, 197, 14, 92, 0, 50, 100, 150, 250, 299};
        float[] reference = {
            10.1393f, 9.8679f, 9.1450f, 9.0120f, 1.3104f, -4.0290f, 0.2439f, -1.6400f, 6.7221f,
            0.5411f
        };
        try (Inkling2.State state = model.newState(128, tokens.length)) {
            model.ingest(state, Batch.prefill(tokens));
            float[] logits = logits(state);
            assertEquals(TinyInkling2.VOCABULARY, logits.length, "the padding rows are dropped");
            assertEquals(ids[0], argmax(logits));
            for (int i = 0; i < ids.length; i++)
                assertEquals(reference[i], logits[ids[i]], 5e-3f, "token " + ids[i]);
        }
    }

    /** History read back from the FP16 caches and the convolution state, against one batch. */
    @Test
    void chunksAndStepsAgreeWithOneBatch() {
        float[] batched = run(tokens, tokens.length);
        for (int chunk : new int[] {1, 7}) {
            double cosine = cosine(batched, run(tokens, chunk));
            assertTrue(cosine > 0.9999, "chunks of " + chunk + ": cosine " + cosine);
        }
    }

    @Test
    void aCheckpointResumesExactly() {
        int split = 40;
        CheckpointCodec<Inkling2.State> codec = model.checkpointCodec().orElseThrow();
        MemorySegment blob = arena.allocate(codec.byteSize(split), 64);
        float[] straight;
        try (Inkling2.State state = model.newState(128, 16)) {
            for (int at = 0; at < split; at += 8)
                model.ingest(state, Batch.prefill(Arrays.copyOfRange(tokens, at, at + 8)));
            codec.capture(state, 0, split, blob);
            for (int at = split; at < tokens.length; at++)
                model.ingest(state, Batch.step(tokens[at]));
            straight = logits(state);

            state.reset();
            for (int at = 0; at < split; at += 8)
                model.ingest(state, Batch.prefill(Arrays.copyOfRange(tokens, at, at + 8)));
            for (int at = split; at < tokens.length; at++)
                model.ingest(state, Batch.step(tokens[at]));
            assertArrayEquals(straight, logits(state), "a reset replays");
        }
        try (Inkling2.State state = model.newState(128, 16)) {
            codec.restore(state, 0, split, blob);
            state.resumeAt(split);
            for (int at = split; at < tokens.length; at++)
                model.ingest(state, Batch.step(tokens[at]));
            assertArrayEquals(straight, logits(state));
        }
    }

    /**
     * With a checkpoint overhead past its limit the cache keeps no blocks while serving: a session
     * whose next prompt departs from its reply rewinds to the last prompt boundary, with only the
     * convolution state restored. The rows before the boundary must have survived the reply, here a
     * reply several windows long.
     */
    @Test
    void aRetainedSessionRewindsToItsPromptBoundary() {
        int boundary = 20;
        Batch shared = Batch.prefill(Arrays.copyOf(tokens, boundary));
        Batch turn = Batch.prefill(Arrays.copyOfRange(tokens, boundary, 24));
        Batch echo = Batch.prefill(Arrays.copyOfRange(tokens, 40, 50));
        PromptCache.Options options =
                PromptCache.Options.DEFAULTS
                        .withRetainedSessions(1)
                        .withContextCapacity(128)
                        .withMaxCheckpointOverhead(0);
        try (PromptCache<Inkling2.State> cache = PromptCache.of(model, seed, options)) {
            cache.serve(
                    List.of(shared, turn),
                    (state, serving) -> {
                        for (int at = 24; at < tokens.length; at++) {
                            model.ingest(state, Batch.step(tokens[at]));
                            serving.tail(tokens[at]);
                        }
                        return null;
                    });
            float[] rewound =
                    cache.serve(
                            List.of(shared, echo),
                            (state, serving) -> {
                                assertEquals(PromptCache.Tier.SESSION, serving.tier());
                                assertEquals(boundary, serving.restored());
                                return logits(state);
                            });
            try (Inkling2.State state = model.newState(128, boundary)) {
                model.ingest(state, shared);
                model.ingest(state, echo);
                assertArrayEquals(logits(state), rewound);
            }
        }
    }

    /**
     * llama.cpp cbb1c1c50 on the same mmproj, {@code llama-mtmd-debug -p encode --image rainbow -n
     * 24}: a 6 x 6 patch grid over the 4 x 4 position table, the pixels given as they are. ggml
     * reads the patches and the attention's K and V in FP16.
     */
    @Test
    void theVisionTowerMatchesLlamaCpp() {
        int size = 24, patch = TinyInkling2.PATCH, grid = size / patch;
        float[] pixels = rainbow(size), patches = new float[pixels.length];
        int at = 0;
        for (int py = 0; py < grid; py++)
            for (int px = 0; px < grid; px++)
                for (int c = 0; c < 3; c++)
                    for (int y = py * patch; y < (py + 1) * patch; y++)
                        for (int x = px * patch; x < (px + 1) * patch; x++)
                            patches[at++] = pixels[(y * size + x) * 3 + c];
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            float[] rows = Views.toFloatArray(vision.encode(patches, grid, grid, scratch), "rows");
            int dim = TinyInkling2.DIM, last = rows.length - dim;
            assertEquals(9 * dim, rows.length);
            float[] first = {-0.9339f, 1.4129f, -3.1150f, 4.2582f, 4.7964f, -8.7942f};
            float[] ninth = {4.4001f, 1.9899f, -0.2372f, 7.3700f, 5.4701f, -16.0012f};
            for (int i = 0; i < 3; i++) {
                assertEquals(first[i], rows[i], 5e-3f, "row 0 lane " + i);
                assertEquals(first[3 + i], rows[dim - 3 + i], 5e-3f, "row 0 lane -" + (3 - i));
                assertEquals(ninth[i], rows[last + i], 5e-3f, "row 8 lane " + i);
                assertEquals(ninth[3 + i], rows[last + dim - 3 + i], 5e-3f, "row 8 lane -" + i);
            }
            double sum = 0;
            for (float value : rows) sum += value;
            assertEquals(462.1457, sum, .2);
        } finally {
            Arenas.close(scratch);
        }
    }

    /** {@code llama-mtmd-debug}'s rainbow: hue by angle, saturation by distance to the centre. */
    private static float[] rainbow(int size) {
        float[] pixels = new float[size * size * 3];
        float centre = size / 2f, reach = (float) Math.sqrt(2 * centre * centre);
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) {
                float dx = x - centre, dy = y - centre;
                float hue = (float) Math.atan2(dy, dx) / (2f * 3.14159265f);
                if (hue < 0) hue += 1f;
                float saturation = Math.min(1f, (float) Math.sqrt(dx * dx + dy * dy) / reach);
                float sixth = hue * 6f, f = sixth - (int) sixth;
                float p = 1f - saturation, q = 1f - saturation * f;
                float t = 1f - saturation * (1f - f);
                float[] rgb =
                        switch ((int) sixth % 6) {
                            case 0 -> new float[] {1, t, p};
                            case 1 -> new float[] {q, 1, p};
                            case 2 -> new float[] {p, 1, t};
                            case 3 -> new float[] {p, q, 1};
                            case 4 -> new float[] {t, p, 1};
                            default -> new float[] {1, p, q};
                        };
                System.arraycopy(rgb, 0, pixels, (y * size + x) * 3, 3);
            }
        return pixels;
    }

    /** An image keeps its size, padded to whole 8 x 8 cells, up to the cap of 4096 rows. */
    @Test
    void anImageIsPaddedAndOnlyAHugeOneIsResized() {
        assertArrayEquals(new int[] {10, 7}, vision.size(image(10, 7)));
        assertEquals(2, vision.positions(image(10, 7)));
        assertEquals(1, vision.positions(image(1, 1)));
        assertArrayEquals(new int[] {656, 392}, vision.size(image(1000, 600)));
        assertEquals(82 * 49, vision.positions(image(1000, 600)));
        MediaProjectorContract.assertContract(vision, image(10, 7), TinyInkling2.DIM);
    }

    /** An image is a block of its own in its turn, between the text blocks around it. */
    @Test
    void anImageTurnIsIngestedAsBlocks() {
        Message turn =
                new Message(
                        Role.USER,
                        List.of(
                                new Content.Text("a"),
                                new Content.Media(image(10, 7)),
                                new Content.Text("b")));
        StringBuilder framing = new StringBuilder();
        try (Inkling2.State state = model.newState(128, 16)) {
            loaded.template()
                    .orElseThrow()
                    .encode(
                            new Conversation(List.of(turn)),
                            16,
                            batch -> {
                                if (batch.input() instanceof Batch.Input.Tokens tokens)
                                    for (int i = 0; i < batch.count(); i++)
                                        framing.append(
                                                loaded.tokenizer()
                                                        .vocabulary()
                                                        .token(tokens.ids()[i]));
                                else framing.append("[row]".repeat(batch.count()));
                                model.ingest(state, batch);
                            });
            for (float logit : logits(state)) assertTrue(Float.isFinite(logit));
        }
        assertEquals(
                "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"
                        + "<|message_user|><|content_text|>a<|end_message|>"
                        + "<|message_user|><|content_image|>[row][row]<|end_message|>"
                        + "<|message_user|><|content_text|>b<|end_message|>",
                framing.toString().replace('\u0120', ' '));
    }

    private static Media.Image image(int width, int height) {
        float[] values = new float[width * height * 3];
        for (int i = 0; i < values.length; i++) values[i] = i * 37 % 256 / 255f;
        return new Media.Image(values, height, width, 3);
    }

    private static float[] run(int[] prompt, int chunk) {
        try (Inkling2.State state = model.newState(128, chunk)) {
            for (int at = 0; at < prompt.length; at += chunk) {
                int[] part = Arrays.copyOfRange(prompt, at, Math.min(prompt.length, at + chunk));
                model.ingest(state, part.length == 1 ? Batch.step(part[0]) : Batch.prefill(part));
            }
            return logits(state);
        }
    }

    private static float[] logits(Inkling2.State state) {
        return Views.toFloatArray(
                Views.castToSegmentBacked(model.logits(state), "logits"), "logits");
    }

    static int argmax(float[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
        return best;
    }

    static double cosine(float[] left, float[] right) {
        double dot = 0, leftSquares = 0, rightSquares = 0;
        for (int i = 0; i < left.length; i++) {
            dot += (double) left[i] * right[i];
            leftSquares += (double) left[i] * left[i];
            rightSquares += (double) right[i] * right[i];
        }
        return dot / Math.sqrt(leftSquares * rightSquares);
    }
}
