package com.qxotic.toknroll.hf;

import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildByteLevelVocab;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.Tokenizer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceTokenizerLoaderValidationTest {

    @TempDir Path tempDir;

    @Test
    void fromLocalRejectsNonExistentPath() {
        assertThrows(
                IllegalArgumentException.class,
                () -> HuggingFaceTokenizerLoader.fromLocal(Path.of("/nonexistent/path")));
    }

    @Test
    void fromLocalRejectsDirectoryWithoutTokenizerJson() {
        assertThrows(RuntimeException.class, () -> HuggingFaceTokenizerLoader.fromLocal(tempDir));
    }

    @Test
    void fromLocalRejectsNullPath() {
        assertThrows(NullPointerException.class, () -> HuggingFaceTokenizerLoader.fromLocal(null));
    }

    @Test
    void fromHuggingFaceRejectsBlankUser() {
        assertThrows(
                RuntimeException.class,
                () -> HuggingFaceTokenizerLoader.fromHuggingFace("", "repo"));
        assertThrows(
                RuntimeException.class,
                () -> HuggingFaceTokenizerLoader.fromHuggingFace(null, "repo"));
    }

    @Test
    void fromHuggingFaceRejectsBlankRepo() {
        assertThrows(
                RuntimeException.class,
                () -> HuggingFaceTokenizerLoader.fromHuggingFace("user", ""));
    }

    @Test
    void emptyVocabIsRejected() throws Exception {
        String json = "{\"model\":{\"type\":\"BPE\",\"vocab\":{},\"merges\":[]}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        assertThrows(RuntimeException.class, () -> HuggingFaceTokenizerLoader.fromLocal(file));
    }

    @Test
    void anIdThatIsNoWholeNumberInRangeIsRefusedByName() throws Exception {
        // the cast these replace truncated 257.9 to 257, wrapped 2^32 to 0, failed -1 without a
        // message and sized the vocabulary by the largest id
        for (String id : new String[] {"-1", "0.5", "256.9", "2147483648", "4294967296", "1e100"}) {
            String vocab =
                    "{\"version\":\"1.0\",\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":"
                            + id
                            + "},\"merges\":[]}}";
            String added =
                    "{\"version\":\"1.0\",\"added_tokens\":[{\"id\":"
                            + id
                            + ",\"content\":\"<x>\"}],"
                            + "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
            assertRefused(vocab, "tokenizer.json:model.vocab['a'] is ");
            assertRefused(added, "tokenizer.json:added_tokens['<x>'].id is ");
        }
    }

    private void assertRefused(String json, String where) throws Exception {
        Path file = Files.writeString(tempDir.resolve("tokenizer.json"), json);
        Throwable t =
                assertThrows(
                        RuntimeException.class, () -> HuggingFaceTokenizerLoader.fromLocal(file));
        String all = "";
        for (; t != null; t = t.getCause()) all += t.getMessage() + " | ";
        assertTrue(
                all.contains(where) && all.contains(", not an id: a whole number from 0 to"), all);
    }

    @Test
    void nonBpeModelTypeIsRejected() throws Exception {
        String json = "{\"model\":{\"type\":\"Unigram\",\"vocab\":{\"h\":0},\"merges\":[]}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        assertThrows(RuntimeException.class, () -> HuggingFaceTokenizerLoader.fromLocal(file));
    }

    @Test
    void untaggedModelWithMergesIsBpe() throws Exception {
        // gpt2's tokenizer.json: a BPE model with no "type" field.
        String json =
                "{\"model\":{\"vocab\":"
                        + buildByteLevelVocab(Map.of("hi", 256))
                        + ",\"merges\":[\"h i\"]}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(file);
        assertArrayEquals(new int[] {256}, tokenizer.encode("hi").toArray());
    }

    @Test
    void untaggedModelWithoutMergesIsRejected() throws Exception {
        String json = "{\"model\":{\"vocab\":{\"h\":0}}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        assertThrows(RuntimeException.class, () -> HuggingFaceTokenizerLoader.fromLocal(file));
    }

    @Test
    void missingModelFieldIsRejected() throws Exception {
        String json = "{\"normalizer\":{\"type\":\"NFC\"}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        assertThrows(RuntimeException.class, () -> HuggingFaceTokenizerLoader.fromLocal(file));
    }

    @Test
    void nonObjectRootIsRejected() throws Exception {
        String json = "[]";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        assertThrows(RuntimeException.class, () -> HuggingFaceTokenizerLoader.fromLocal(file));
    }

    @Test
    void nonStringVocabValueIsRejected() throws Exception {
        String json =
                "{\"model\":{\"type\":\"BPE\",\"vocab\":{\"h\":\"not-a-number\"},\"merges\":[]}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        assertThrows(RuntimeException.class, () -> HuggingFaceTokenizerLoader.fromLocal(file));
    }

    private static String byteLevelVocabJson() {
        StringBuilder vocab = new StringBuilder();
        for (int i = 0; i < 256; i++) {
            if (i > 0) vocab.append(",");
            vocab.append(jsonEscape(ByteLevel.encode(new byte[] {(byte) i}))).append(":").append(i);
        }
        return vocab.toString();
    }

    private static String jsonEscape(String s) {
        StringBuilder sb = new StringBuilder("\"");
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '\\' || c == '"') sb.append('\\').append(c);
            else if (c >= 0x20 && c < 0x7F) sb.append(c);
            else sb.append("\\u").append(String.format("%04x", (int) c));
        }
        return sb.append("\"").toString();
    }

    @Test
    void byteLevelTokenizerJsonLoads() throws Exception {
        String json =
                "{\"model\":{"
                        + "\"type\":\"BPE\","
                        + "\"vocab\":{"
                        + byteLevelVocabJson()
                        + "},"
                        + "\"merges\":[]"
                        + "}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        Tokenizer t = HuggingFaceTokenizerLoader.fromLocal(file);
        assertNotNull(t);
    }

    @Test
    void normalizerNfcLoads() throws Exception {
        String json =
                "{"
                        + "\"normalizer\":{\"type\":\"NFC\"},"
                        + "\"model\":{"
                        + "\"type\":\"BPE\","
                        + "\"vocab\":{"
                        + byteLevelVocabJson()
                        + "},"
                        + "\"merges\":[]"
                        + "}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        Tokenizer t = HuggingFaceTokenizerLoader.fromLocal(file);
        assertNotNull(t);
    }

    @Test
    void normalizerLowercaseLoads() throws Exception {
        String json =
                "{"
                        + "\"normalizer\":{\"type\":\"Lowercase\"},"
                        + "\"model\":{"
                        + "\"type\":\"BPE\","
                        + "\"vocab\":{"
                        + byteLevelVocabJson()
                        + "},"
                        + "\"merges\":[]"
                        + "}}";
        Path file = tempDir.resolve("tokenizer.json");
        Files.writeString(file, json);
        Tokenizer t = HuggingFaceTokenizerLoader.fromLocal(file);
        assertNotNull(t);
    }
}
