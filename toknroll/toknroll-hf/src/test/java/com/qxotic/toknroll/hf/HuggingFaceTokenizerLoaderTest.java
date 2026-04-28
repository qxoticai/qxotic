package com.qxotic.toknroll.hf;

import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildBpeModel;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildByteLevelVocab;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildTokenizerJson;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceTokenizerLoaderTest {

    @TempDir Path tempDir;

    @Test
    void loadFromTokenizerJsonPath_buildsTokenizer() throws IOException {
        Path tokenizerJson =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        buildByteLevelTokenizerJsonWithAbToken(),
                        StandardCharsets.UTF_8);

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);

        assertArrayEquals(new int[] {256}, tokenizer.encode("ab").toArray());
    }

    @Test
    void loadFromDirectory_usesTokenizerJson() throws IOException {
        Path modelDir = Files.createDirectory(tempDir.resolve("model"));
        Files.writeString(
                modelDir.resolve("tokenizer.json"),
                buildByteLevelTokenizerJson(true),
                StandardCharsets.UTF_8);

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(modelDir);
        assertEquals(257, tokenizer.vocabulary().size());
    }

    @Test
    void ignoreMergesTrue_prefersExistingWholeToken() throws IOException {
        Path tokenizerJson =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        buildByteLevelTokenizerJson(true),
                        StandardCharsets.UTF_8);

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {256}, tokenizer.encode("ab").toArray());
    }

    @Test
    void ignoreMergesFalse_keepsBytePairPathOnly() throws IOException {
        Path tokenizerJson =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        buildByteLevelTokenizerJson(false),
                        StandardCharsets.UTF_8);

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {97, 98}, tokenizer.encode("ab").toArray());
    }

    @Test
    void ignoreMergesTrue_stillAppliesRegularMergesWhenWholeTokenMissing() throws IOException {
        Path tokenizerJson =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        buildByteLevelTokenizerJsonWithAbMerge(true),
                        StandardCharsets.UTF_8);

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {256, 99}, tokenizer.encode("abc").toArray());
    }

    private static String buildByteLevelTokenizerJson(boolean ignoreMerges) {
        return buildTokenizerJson(
                buildBpeModel(
                        buildByteLevelVocab(Map.of("ab", 256)),
                        "[]",
                        ",\"ignore_merges\":" + ignoreMerges));
    }

    private static String buildByteLevelTokenizerJsonWithAbMerge(boolean ignoreMerges) {
        return buildTokenizerJson(
                buildBpeModel(
                        buildByteLevelVocab(Map.of("ab", 256)),
                        "[[\"a\",\"b\"]]",
                        ",\"ignore_merges\":" + ignoreMerges));
    }

    private static String buildByteLevelTokenizerJsonWithAbToken() {
        return buildTokenizerJson(
                buildBpeModel(
                        buildByteLevelVocab(Map.of("ab", 256)),
                        "[[\"a\",\"b\"]]",
                        ",\"ignore_merges\":false"));
    }

    @Test
    void unsupportedModelType_throws() throws IOException {
        Path tokenizerJson =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        "{\"model\":{\"type\":\"WordPiece\",\"vocab\":{\"a\":0}}}",
                        StandardCharsets.UTF_8);

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(error.getMessage().contains("model.type"));
    }

    @Test
    void unsupportedPreTokenizer_throws() throws IOException {
        Path tokenizerJson =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        "{"
                                + "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]},"
                                + "\"pre_tokenizer\":{\"type\":\"Whitespace\"}"
                                + "}",
                        StandardCharsets.UTF_8);

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(error.getMessage().contains("pre_tokenizer.type"));
    }

    @Test
    void missingTokenizerJsonInDirectory_throws() throws IOException {
        Path emptyDir = Files.createDirectory(tempDir.resolve("empty"));
        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(emptyDir));
        assertTrue(error.getMessage().contains("Missing tokenizer.json"));
    }
}
