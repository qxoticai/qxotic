package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryAllocators;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.content.Media;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.TokenCountBatchingStrategy;
import org.springframework.util.MimeType;

/**
 * EmbeddingGemma 2 with its projector attached as {@code companion("media", ...)}: a media {@link
 * Document} embeds into the space text queries search, without the text prompt, a mixed batch keeps
 * document order and equals its documents one by one, a PDF is refused by its type, an input over
 * the context is refused naming the positions it needs, and a load without the projector refuses
 * media naming the companion. Model-gated via {@link TestModels}. Run: {@code mvn test
 * -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class EmbeddingGemma2MediaIT {

    static final String MODEL = "hf.co/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf";
    static final String MMPROJ = "hf.co/unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf";

    static JinferEmbeddingModel model;
    static Document speech, chirp;

    @BeforeAll
    static void load() {
        Path weights = TestModels.require(MODEL), projector = TestModels.require(MMPROJ);
        speech =
                audio(
                        TestModels.fixture(
                                "audio/speech-tts.wav")); // "Hello world one two three four five"
        chirp = audio(TestModels.fixture("audio/chirp-5s.wav"));
        model =
                JinferEmbeddingModel.builder()
                        .modelPath(weights)
                        .companionPath("media", projector)
                        .contextCapacity(2048)
                        .build();
        for (int warm = 0; warm < 3; warm++) model.embed(speech);
    }

    @AfterAll
    static void close() {
        if (model != null) model.close();
    }

    @Test
    void aTextQueryFindsTheSpokenDocument() {
        float[] query = model.embed("a person saying hello world and counting");
        assertTrue(cosine(query, model.embed(speech)) > cosine(query, model.embed(chirp)));
    }

    @Test
    void aNullQueryIsRefusedRatherThanEmbeddedAsText() {
        IllegalArgumentException failure =
                assertThrows(IllegalArgumentException.class, () -> model.embed((String) null));
        assertEquals("Text must not be null", failure.getMessage());
        assertEquals(model.dimensions(), model.embed("null").length);
    }

    /**
     * The model card: "Prefixes apply to text only. Pass images, video, and audio without any
     * prefix." A media document embeds exactly as the core embedder embeds the media bare, alone
     * and in the batched route vector stores take.
     */
    @Test
    void mediaEmbedsWithoutThePrompt() throws IOException {
        List<float[]> bare = new ArrayList<>();
        try (Arena arena = Arena.ofShared()) {
            LoadedEmbedder<?> core =
                    Models.loadEmbedder(
                            TestModels.require(MODEL),
                            arena,
                            Map.of("media", TestModels.require(MMPROJ)),
                            null);
            ContextState state = core.model().newState(2048, MemoryAllocators.ofArena(arena));
            for (String wav : List.of("audio/speech-tts.wav", "audio/chirp-5s.wav")) {
                Object audio = AudioCodec.decode(Files.readAllBytes(TestModels.fixture(wav)));
                core.embed(state, List.of(audio), core.dimension(), bare::add);
            }
        }
        assertTrue(cosine(model.embed(speech), bare.get(0)) > 0.99999, "speech");
        assertTrue(cosine(model.embed(chirp), bare.get(1)) > 0.99999, "chirp");
        List<float[]> batch =
                model.embed(
                        List.of(new Document("A tone rising in pitch."), speech, chirp),
                        EmbeddingOptions.builder().build(),
                        new TokenCountBatchingStrategy());
        assertTrue(cosine(batch.get(1), bare.get(0)) > 0.99999, "speech in a batch");
        assertTrue(cosine(batch.get(2), bare.get(1)) > 0.99999, "chirp in a batch");
    }

    /**
     * A model-wide width reaches media as it reaches text: with {@code dimensions(256)} and no
     * width in the options, a media document embeds as the core embedder's bare 256 vector, alone
     * and in the batched route vector stores take.
     */
    @Test
    void aModelWideWidthReachesMedia() throws IOException {
        Object audio =
                AudioCodec.decode(Files.readAllBytes(TestModels.fixture("audio/speech-tts.wav")));
        try (Arena arena = Arena.ofShared()) {
            LoadedEmbedder<?> core =
                    Models.loadEmbedder(
                            TestModels.require(MODEL),
                            arena,
                            Map.of("media", TestModels.require(MMPROJ)),
                            null);
            ContextState state = core.model().newState(2048, MemoryAllocators.ofArena(arena));
            float[][] bare = new float[1][];
            core.embed(state, List.of(audio), 256, v -> bare[0] = v);
            try (JinferEmbeddingModel narrow =
                    JinferEmbeddingModel.builder()
                            .model(core)
                            .contextCapacity(2048)
                            .dimensions(256)
                            .build()) {
                assertEquals(256, narrow.dimensions());
                float[] alone = narrow.embed(speech);
                assertEquals(256, alone.length);
                assertTrue(cosine(alone, bare[0]) > 0.99999, "speech");
                List<float[]> batch =
                        narrow.embed(
                                List.of(new Document("Someone counts from one to five."), speech),
                                EmbeddingOptions.builder().build(),
                                new TokenCountBatchingStrategy());
                assertEquals(256, batch.get(0).length);
                assertTrue(cosine(batch.get(1), bare[0]) > 0.99999, "speech in a batch");
            }
        }
    }

    @Test
    void aMixedBatchKeepsOrderAndEqualsItsDocumentsOneByOne() {
        List<Document> documents =
                List.of(
                        new Document("Someone counts from one to five."),
                        speech,
                        new Document("A tone rising in pitch."),
                        chirp);
        List<float[]> batch =
                model.embed(
                        documents,
                        EmbeddingOptions.builder().build(),
                        new TokenCountBatchingStrategy());
        assertEquals(documents.size(), batch.size());
        for (int i = 0; i < documents.size(); i++)
            assertTrue(cosine(batch.get(i), model.embed(documents.get(i))) > 0.99999, "doc " + i);
    }

    /** A type that is not media is refused as such: the projector it already has is no fix. */
    @Test
    void aPdfIsRefusedByItsTypeNotByTheProjector() {
        Document pdf =
                Document.builder()
                        .media(
                                Media.builder()
                                        .mimeType(MimeType.valueOf("application/pdf"))
                                        .data(new byte[] {'%', 'P', 'D', 'F'})
                                        .build())
                        .build();
        UnsupportedOperationException e =
                assertThrows(UnsupportedOperationException.class, () -> model.embed(pdf));
        assertEquals(
                "media type application/pdf is not supported: jinfer takes image, audio and video",
                e.getMessage());
    }

    /** An input over the context is refused up front, naming the positions it needs. */
    @Test
    void anInputOverTheContextIsRefusedWithThePositionsItNeeds() {
        try (var small =
                JinferEmbeddingModel.builder()
                        .modelPath(TestModels.require(MODEL))
                        .companionPath("media", TestModels.require(MMPROJ))
                        .contextCapacity(64)
                        .build()) {
            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> small.embed(speech));
            assertTrue(
                    e.getMessage()
                            .matches(
                                    "input exceeds context capacity \\(\\d+ positions, 64"
                                            + " available - raise contextCapacity\\)"),
                    e.getMessage());
        }
    }

    @Test
    void withoutTheProjectorMediaIsRefusedNamingTheCompanion() {
        try (var bare =
                JinferEmbeddingModel.builder()
                        .modelPath(TestModels.require(MODEL))
                        .contextCapacity(2048)
                        .build()) {
            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> bare.embed(speech));
            assertTrue(e.getMessage().contains("embeds text only"), e.getMessage());
            assertTrue(
                    e.getMessage().contains("spring.ai.jinfer.embedding.companions.media"),
                    e.getMessage());
        }
    }

    static Document audio(Path wav) {
        return Document.builder()
                .media(new Media(MimeType.valueOf("audio/wav"), wav.toUri()))
                .build();
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
}
