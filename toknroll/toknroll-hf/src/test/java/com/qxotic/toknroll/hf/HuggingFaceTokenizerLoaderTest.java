package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceTokenizerLoaderTest {

    @TempDir Path tempDir;

    @Test
    void loadFromTokenizerJsonPath_buildsTokenizer() throws IOException {
        Path tokenizerJson =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        "{\"model\":{\"type\":\"BPE\","
                            + "\"vocab\":{\"a\":0,\"b\":1,\"ab\":2,\"<|end|>\":3},"
                            + "\"merges\":[[\"a\",\"b\"]]},\"pre_tokenizer\":{\"type\":\"Split\","
                            + "\"pattern\":{\"Regex\":\"[ab<|end|>]+\"},\"behavior\":\"Removed\","
                            + "\"invert\":true},"
                            + "\"added_tokens\":[{\"id\":3,\"content\":\"<|end|>\",\"special\":true}]"
                            + "}",
                        StandardCharsets.UTF_8);

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);

        assertArrayEquals(new int[] {2}, tokenizer.encode("ab").toArray());
    }

    @Test
    void loadFromDirectory_usesTokenizerJson() throws IOException {
        Path modelDir = Files.createDirectory(tempDir.resolve("model"));
        Files.writeString(
                modelDir.resolve("tokenizer.json"),
                "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}",
                StandardCharsets.UTF_8);

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(modelDir);
        assertEquals(1, tokenizer.vocabulary().size());
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
        StringBuilder vocab = new StringBuilder();
        for (int b = 0; b < 256; b++) {
            if (b > 0) {
                vocab.append(',');
            }
            vocab.append('"')
                    .append(escapeJsonChar(ByteLevel.encodeSingle((byte) b)))
                    .append('"')
                    .append(':')
                    .append(b);
        }
        vocab.append(",\"ab\":256");
        return "{\"model\":{\"type\":\"BPE\",\"ignore_merges\":"
                + ignoreMerges
                + ",\"vocab\":{"
                + vocab
                + "},\"merges\":[]}}";
    }

    private static String buildByteLevelTokenizerJsonWithAbMerge(boolean ignoreMerges) {
        StringBuilder vocab = new StringBuilder();
        for (int b = 0; b < 256; b++) {
            if (b > 0) {
                vocab.append(',');
            }
            vocab.append('"')
                    .append(escapeJsonChar(ByteLevel.encodeSingle((byte) b)))
                    .append('"')
                    .append(':')
                    .append(b);
        }
        vocab.append(",\"ab\":256");
        return "{\"model\":{\"type\":\"BPE\",\"ignore_merges\":"
                + ignoreMerges
                + ",\"vocab\":{"
                + vocab
                + "},\"merges\":[[\"a\",\"b\"]]}}";
    }

    private static String escapeJsonChar(char c) {
        switch (c) {
            case '"':
                return "\\\"";
            case '\\':
                return "\\\\";
            default:
                return String.valueOf(c);
        }
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
