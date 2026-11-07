package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryAllocators;
import dev.langchain4j.data.message.AudioContent;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.ContentType;
import dev.langchain4j.data.message.ImageContent;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.embedding.request.EmbeddingInput;
import dev.langchain4j.model.embedding.request.EmbeddingInputType;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * EmbeddingGemma 2 with its projector attached as {@code companion("media", ...)}: image, audio and
 * interleaved inputs embed into the space text queries search, without the text prompt whatever the
 * input type, a mixed batch equals its inputs one by one, and a load without the projector stays
 * text-only at langchain4j's own content check.
 */
@Tag("integration")
class EmbeddingGemma2MediaIT {

    static final String MODEL = "hf.co/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf";
    static final String MMPROJ = "hf.co/unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf";

    static JinferEmbeddingModel model;
    static URI cat, dice, speech;

    @BeforeAll
    static void load() throws Exception {
        Path weights = TestModels.require(MODEL), projector = TestModels.require(MMPROJ);
        cat = EmbeddingGemma2MediaIT.class.getResource("/kit-images/cat.png").toURI();
        dice = EmbeddingGemma2MediaIT.class.getResource("/kit-images/dice.png").toURI();
        speech = fixture("audio/speech-tts.wav").toUri(); // "Hello world one two three four five"
        model =
                JinferEmbeddingModel.builder()
                        .modelPath(weights)
                        .companionPath("media", projector)
                        .contextCapacity(2048)
                        .build();
        for (int warm = 0; warm < 3; warm++)
            embed(EmbeddingInputType.DOCUMENT, ImageContent.from(cat));
    }

    @AfterAll
    static void close() {
        if (model != null) model.close();
    }

    @Test
    void theProjectorDeclaresItsMedia() {
        assertEquals(
                Set.of(ContentType.TEXT, ContentType.IMAGE, ContentType.AUDIO, ContentType.VIDEO),
                model.supportedContentTypes());
    }

    @Test
    void textQueriesFindTheirMedium() {
        List<float[]> documents =
                List.of(
                        embed(EmbeddingInputType.DOCUMENT, ImageContent.from(cat)),
                        embed(EmbeddingInputType.DOCUMENT, ImageContent.from(dice)),
                        embed(EmbeddingInputType.DOCUMENT, AudioContent.from(speech)));
        assertEquals(0, nearest(query("a photo of a cat"), documents));
        assertEquals(1, nearest(query("dice on a table"), documents));
        assertEquals(2, nearest(query("a person saying hello world and counting"), documents));
    }

    /**
     * The model card: "Prefixes apply to text only. Pass images, video, and audio without any
     * prefix." Whatever the input type, an input carrying media embeds exactly as the core embedder
     * embeds its parts bare.
     */
    @Test
    void mediaEmbedsWithoutThePrompt() throws IOException {
        Object catImage = ImageCodec.decode(Files.readAllBytes(Path.of(cat)));
        Object diceImage = ImageCodec.decode(Files.readAllBytes(Path.of(dice)));
        Object speechAudio = AudioCodec.decode(Files.readAllBytes(Path.of(speech)));
        List<float[]> bare =
                bare(
                        List.of(catImage),
                        List.of(speechAudio),
                        List.of("A photo of dice:", diceImage));
        for (EmbeddingInputType type :
                Arrays.asList(EmbeddingInputType.DOCUMENT, EmbeddingInputType.QUERY, null)) {
            assertTrue(cosine(embed(type, ImageContent.from(cat)), bare.get(0)) > 0.99999, "image");
            assertTrue(
                    cosine(embed(type, AudioContent.from(speech)), bare.get(1)) > 0.99999, "audio");
            assertTrue(
                    cosine(
                                    embed(
                                            type,
                                            TextContent.from("A photo of dice:"),
                                            ImageContent.from(dice)),
                                    bare.get(2))
                            > 0.99999,
                    "interleaved");
        }
    }

    @Test
    void aMixedBatchEqualsItsInputsOneByOne() {
        List<EmbeddingInput> inputs =
                List.of(
                        EmbeddingInput.from("A tabby cat looking at the camera."),
                        EmbeddingInput.from(
                                TextContent.from("A photo of dice:"), ImageContent.from(dice)),
                        EmbeddingInput.from(AudioContent.from(speech)));
        EmbeddingResponse batch =
                model.embed(
                        EmbeddingRequest.builder()
                                .inputs(inputs)
                                .inputType(EmbeddingInputType.DOCUMENT)
                                .build());
        assertEquals(3, batch.embeddings().size());
        for (int i = 0; i < inputs.size(); i++) {
            float[] alone =
                    embed(
                            EmbeddingInputType.DOCUMENT,
                            inputs.get(i).contents().toArray(Content[]::new));
            assertTrue(cosine(batch.embeddings().get(i).vector(), alone) > 0.99999, "input " + i);
        }
        assertTrue(batch.metadata().tokenUsage().inputTokenCount() > 280, "media rows count");
    }

    @Test
    void withoutTheProjectorMediaIsRefusedBeforeAnyDecode() {
        try (var bare =
                JinferEmbeddingModel.builder()
                        .modelPath(TestModels.require(MODEL))
                        .contextCapacity(2048)
                        .build()) {
            assertEquals(Set.of(ContentType.TEXT), bare.supportedContentTypes());
            assertThrows(
                    UnsupportedFeatureException.class,
                    () ->
                            bare.embed(
                                    EmbeddingRequest.builder()
                                            .input(ImageContent.from(cat))
                                            .build()));
        }
    }

    static float[] embed(EmbeddingInputType type, Content... parts) {
        return model.embed(
                        EmbeddingRequest.builder()
                                .input(EmbeddingInput.from(parts))
                                .inputType(type)
                                .build())
                .embeddings()
                .getFirst()
                .vector();
    }

    /** The core embedder's vectors for the given inputs, from a load of its own. */
    @SafeVarargs
    static List<float[]> bare(List<Object>... inputs) throws IOException {
        try (Arena arena = Arena.ofShared()) {
            LoadedEmbedder<?> core =
                    Models.loadEmbedder(
                            TestModels.require(MODEL),
                            arena,
                            Map.of("media", TestModels.require(MMPROJ)),
                            null);
            ContextState state = core.model().newState(2048, MemoryAllocators.ofArena(arena));
            List<float[]> vectors = new ArrayList<>();
            for (List<Object> parts : inputs)
                core.embed(state, parts, core.dimension(), vectors::add);
            return vectors;
        }
    }

    static float[] query(String text) {
        return embed(EmbeddingInputType.QUERY, TextContent.from(text));
    }

    static int nearest(float[] query, List<float[]> documents) {
        int best = 0;
        for (int i = 1; i < documents.size(); i++)
            if (cosine(query, documents.get(i)) > cosine(query, documents.get(best))) best = i;
        return best;
    }

    static double cosine(float[] a, float[] b) {
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.length; i++) {
            ab += a[i] * b[i];
            aa += a[i] * a[i];
            bb += b[i] * b[i];
        }
        return ab / Math.sqrt(aa * bb);
    }

    private static Path fixture(String name) {
        for (Path directory = Path.of("").toAbsolutePath();
                directory != null;
                directory = directory.getParent()) {
            Path candidate = directory.resolve("test-fixtures").resolve(name);
            if (Files.isRegularFile(candidate)) return candidate;
        }
        assumeTrue(false, "test-fixtures/" + name + " not found");
        throw new AssertionError();
    }
}
