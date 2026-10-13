package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
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
 * Document} embeds into the space text queries search, a mixed batch keeps document order and
 * equals its documents one by one, and a load without the projector refuses media naming the
 * companion. Model-gated via {@link TestModels}. Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl jinfer-spring-ai}
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
        speech = audio(fixture("audio/speech-tts.wav")); // "Hello world one two three four five"
        chirp = audio(fixture("audio/chirp-5s.wav"));
        model =
                JinferEmbeddingModel.builder()
                        .modelPath(weights)
                        .companionPath("media", projector)
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

    @Test
    void withoutTheProjectorMediaIsRefusedNamingTheCompanion() {
        try (var bare =
                JinferEmbeddingModel.builder().modelPath(TestModels.require(MODEL)).build()) {
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
