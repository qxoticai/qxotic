package com.qxotic.toknroll.hf;

import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.MISTRAL_DECODER;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildBpeModel;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildByteLevelVocab;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildSentencePieceVocab;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildTokenizerJson;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.UnsupportedTokenizerException;
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
            assertRefused(added, "tokenizer.json:added_tokens[0].id is ");
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
    void mergesHuggingFaceRefusesAreRefusedByIndex() throws Exception {
        // tokenizers 0.23.3 refuses every one of these files; the loader used to skip the merge
        String vocab = buildByteLevelVocab(Map.of("he", 256));
        String list = "tokenizer.json:model.merges must be a list";
        assertRefusedAs("{\"type\":\"BPE\",\"vocab\":" + vocab + "}", list);
        for (String merges : new String[] {"null", "\"h e\"", "5"}) {
            assertRefusedAs(buildBpeModel(vocab, merges, ""), list);
        }
        String[] shapes = {
            "\"h\"",
            "\"\"",
            "\"h  e\"",
            "\"h e \"",
            "\"h e x\"",
            "123",
            "[\"h\"]",
            "[\"h\",\"e\",\"l\"]",
            "[\"h\",1]"
        };
        for (String bad : shapes) {
            assertRefusedAs(
                    buildBpeModel(vocab, "[\"h e\"," + bad + "]", ""),
                    "tokenizer.json:model.merges[1] is ");
        }
        assertRefusedAs( // a string quoted, so an empty one or a doubled space shows
                buildBpeModel(vocab, "[\"h  e\"]", ""),
                "tokenizer.json:model.merges[0] is 'h  e', not \"left right\" or [\"left\","
                        + " \"right\"]");
        String absent = "', which is not in the vocabulary";
        assertRefusedAs(
                buildBpeModel(vocab, "[\"h e\",\"h zz\"]", ""),
                "tokenizer.json:model.merges[1] names 'zz" + absent);
        assertRefusedAs( // the merged token, too
                buildBpeModel(vocab, "[\"h e\",[\"e\",\"l\"]]", ""),
                "tokenizer.json:model.merges[1] names 'el" + absent);
        assertRefusedAs( // and on the SentencePiece path
                buildBpeModel(
                        buildSentencePieceVocab(Map.of("▁", 256)),
                        "[\"▁ zz\"]",
                        ",\"byte_fallback\":true"),
                "tokenizer.json:model.merges[0] names 'zz" + absent,
                METASPACE_PRE_TOKENIZER,
                MISTRAL_DECODER);
    }

    private static final String METASPACE_PRE_TOKENIZER =
            "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}";

    /**
     * BERT's files name no model type and have no merges: a WordPiece model, unsupported, not
     * malformed.
     */
    @Test
    void aModelWithoutTypeOrMergesIsUnsupportedNotMalformed() throws Exception {
        Path file =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        "{\"version\":\"1.0\",\"model\":{\"vocab\":{\"[UNK]\":0,\"hello\":1},"
                                + "\"unk_token\":\"[UNK]\"},\"added_tokens\":[]}");
        UnsupportedTokenizerException refused =
                assertThrows(
                        UnsupportedTokenizerException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(file));
        assertTrue(
                refused.getMessage().contains("tokenizer.json:model.type: WordPiece, untyped"),
                refused.getMessage());
    }

    /** T5's file names no type either; its vocabulary is a scored list: a Unigram model. */
    @Test
    void aModelWithoutTypeAndAScoredVocabularyIsUnigram() throws Exception {
        Path file =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"),
                        "{\"version\":\"1.0\",\"model\":{\"unk_id\":0,\"vocab\":[[\"<unk>\",0.0],"
                                + "[\"▁hello\",-1.5]]},\"added_tokens\":[]}");
        UnsupportedTokenizerException refused =
                assertThrows(
                        UnsupportedTokenizerException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(file));
        assertTrue(
                refused.getMessage().contains("tokenizer.json:model.type: Unigram, untyped"),
                refused.getMessage());
    }

    /** A well-formed option that would make another tokenizer is refused, never read as absent. */
    @Test
    void aBehaviorChangingModelOptionIsRefused() throws Exception {
        String byteLevel = buildByteLevelVocab(Map.of("he", 256));
        assertRefusedAs( // HF tokenizers 1.0 refuses any dropout; 0 is the no-op it loads
                buildBpeModel(byteLevel, "[\"h e\"]", ",\"dropout\":1.0"),
                "Unsupported tokenizer config at tokenizer.json:model.dropout: 1.0"
                        + " (supported: null or 0)");
        assertNotNull(
                HuggingFaceTokenizerLoader.fromLocal(
                        Files.writeString(
                                tempDir.resolve("tokenizer.json"),
                                buildTokenizerJson(
                                        buildBpeModel(byteLevel, "[\"h e\"]", ",\"dropout\":0.0"),
                                        "\"pre_tokenizer\":{\"type\":\"ByteLevel\"}",
                                        "\"decoder\":{\"type\":\"ByteLevel\"}"))));
        String sentencePiece = buildSentencePieceVocab(Map.of("▁", 256));
        for (String fallback : new String[] {",\"byte_fallback\":false", ""}) { // absent is false
            assertRefusedAs( // HF emits <unk> where the bytes would be spelled
                    buildBpeModel(sentencePiece, "[]", fallback),
                    "Unsupported tokenizer config at tokenizer.json:model.byte_fallback: "
                            + (fallback.isEmpty() ? "null" : "false")
                            + " (supported: true for a vocabulary with byte tokens)",
                    METASPACE_PRE_TOKENIZER,
                    MISTRAL_DECODER);
        }
        assertRefusedAs( // a Metaspace decoder of another replacement would leave ▁ in the text
                buildBpeModel(sentencePiece, "[]", ",\"byte_fallback\":true"),
                "Unsupported tokenizer config at tokenizer.json:decoder: ",
                METASPACE_PRE_TOKENIZER,
                "\"decoder\":{\"type\":\"Metaspace\",\"replacement\":\"_\"}");
    }

    private void assertRefusedAs(String model, String message, String... rootFields)
            throws Exception {
        Path file =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"), buildTokenizerJson(model, rootFields));
        int merges = model.indexOf("\"merges\"");
        String loaded =
                (merges < 0 ? "no merges" : model.substring(merges)) + String.join(",", rootFields);
        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(file),
                        loaded);
        assertTrue(refused.getMessage().startsWith(message), refused.getMessage());
    }

    @Test
    void aByteLevelModelRefusesADecoderThatReadsItsIdsAsOtherText() throws Exception {
        // Hugging Face decodes [he, l] as "he l" under WordPiece or no decoder; toknroll as "hel"
        String model = buildBpeModel(buildByteLevelVocab(Map.of("he", 256)), "[\"h e\"]", "");
        String byteLevel = "{\"type\":\"ByteLevel\"}";
        String[][] others = { // the decoder, as the refusal names it
            {"{\"type\":\"WordPiece\",\"prefix\":\"##\"}", "WordPiece"},
            {"{\"type\":\"Metaspace\",\"replacement\":\"▁\"}", "Metaspace"},
            {"{\"type\":\"NoSuchDecoder\"}", "NoSuchDecoder"},
            {
                "{\"type\":\"Sequence\",\"decoders\":[" + byteLevel + ",{\"type\":\"Fuse\"}]}",
                "ByteLevel+Fuse"
            },
            {"{\"type\":\"Sequence\",\"decoders\":[]}", "an empty Sequence"}
        };
        for (String[] other : others) {
            assertRefusedAs(
                    model,
                    "Unsupported tokenizer config at tokenizer.json:decoder: "
                            + other[1]
                            + " (supported: ByteLevel for a byte-level model)",
                    "\"decoder\":" + other[0]);
        }
        // the bytes decoded: ByteLevel, alone in a Sequence, or no decoder stated
        String[] bytes = {
            byteLevel, "{\"type\":\"Sequence\",\"decoders\":[" + byteLevel + "]}", "null"
        };
        for (String decoder : bytes) {
            Path file =
                    Files.writeString(
                            tempDir.resolve("tokenizer.json"),
                            buildTokenizerJson(model, "\"decoder\":" + decoder));
            Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(file);
            assertEquals("hel", tokenizer.decode(tokenizer.encode("hel")), decoder);
        }
    }

    @Test
    void aValidTokenizerToknrollDoesNotImplementIsUnsupportedAMalformedOneIsNot() throws Exception {
        // unsupported says another library may load it; a plain refusal, that the file is broken
        String bpe = buildBpeModel(buildByteLevelVocab(Map.of("he", 256)), "[\"h e\"]", "");
        assertUnsupported(true, "{\"type\":\"Unigram\",\"vocab\":[[\"h\",0.0]]}");
        assertUnsupported(true, bpe, "\"decoder\":{\"type\":\"WordPiece\",\"prefix\":\"##\"}");
        assertUnsupported(true, bpe, "\"normalizer\":{\"type\":\"Precompiled\"}");
        // a byte-level model whose text never becomes bytes: Hugging Face drops what is not a token
        assertUnsupported(
                true,
                bpe,
                "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\" \"},"
                        + "\"behavior\":\"Isolated\"}");
        // a prefix space toknroll never adds
        assertUnsupported(
                true, bpe, "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":true}");
        assertUnsupported(false, bpe, "\"normalizer\":5");
        assertUnsupported(false, buildBpeModel(buildByteLevelVocab(Map.of()), "[\"h\"]", ""));
    }

    private void assertUnsupported(boolean unsupported, String model, String... rootFields)
            throws Exception {
        Path file =
                Files.writeString(
                        tempDir.resolve("tokenizer.json"), buildTokenizerJson(model, rootFields));
        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(file));
        assertEquals(
                unsupported,
                refused instanceof UnsupportedTokenizerException,
                refused.getMessage());
    }

    @Test
    void addedTokensAndNormalizersHuggingFaceRefusesAreRefusedByIndex() throws Exception {
        // tokenizers 0.23.3 refuses each file; the loader used to skip the entry, or failed later
        String bpe = buildBpeModel(buildByteLevelVocab(Map.of("he", 256)), "[\"h e\"]", "");
        assertRefusedAs(
                bpe,
                "tokenizer.json:normalizer.normalizers[1] is null",
                "\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"NFC\"},null]}");
        for (String notAList : new String[] {"null", "{}", "\"x\""}) {
            assertRefusedAs(
                    bpe,
                    "tokenizer.json:added_tokens must be a list",
                    "\"added_tokens\":" + notAList);
        }
        assertRefusedAs(
                bpe, "tokenizer.json:added_tokens[0] must be an object", "\"added_tokens\":[5]");
        assertRefusedAs(
                bpe,
                "tokenizer.json:added_tokens[0].id must be numeric",
                "\"added_tokens\":[{\"content\":\"<x>\"}]");
        assertRefusedAs(
                bpe,
                "tokenizer.json:added_tokens[0].content must be a string",
                "\"added_tokens\":[{\"id\":257}]");
    }

    @Test
    void anAddedTokenMustAgreeWithEveryOtherSpellingOfItsIdAndIdOfItsSpelling() throws Exception {
        // where they disagree Hugging Face renumbers or drops the token: refused, never guessed
        String bpe = buildBpeModel(buildByteLevelVocab(Map.of("he", 256)), "[\"h e\"]", "");
        assertRefusedAs(
                bpe,
                "tokenizer.json:added_tokens[0] gives id 256 to '<x>', which is 'he'",
                "\"added_tokens\":[" + added(256, "<x>") + "]");
        assertRefusedAs(
                bpe,
                "tokenizer.json:added_tokens[0] gives 'he' id 257, which is id 256",
                "\"added_tokens\":[" + added(257, "he") + "]");
        assertRefusedAs(
                bpe,
                "tokenizer.json:added_tokens[1] gives id 257 to '<y>', which is '<x>'",
                "\"added_tokens\":[" + added(257, "<x>") + "," + added(257, "<y>") + "]");
        assertRefusedAs(
                bpe,
                "tokenizer.json:added_tokens[1] gives '<x>' id 258, which is id 257",
                "\"added_tokens\":[" + added(257, "<x>") + "," + added(258, "<x>") + "]");
        // agreeing loads: a repeat (Yi-6B lists <fim_suffix> twice), a vocabulary token's own id,
        // empty content, and no added_tokens at all
        for (String agreeing :
                new String[] {
                    "\"added_tokens\":[" + added(257, "<x>") + "," + added(257, "<x>") + "]",
                    "\"added_tokens\":[" + added(256, "he") + "]",
                    "\"added_tokens\":[" + added(257, "") + "]",
                    "\"pre_tokenizer\":null"
                }) {
            Path file =
                    Files.writeString(
                            tempDir.resolve("tokenizer.json"), buildTokenizerJson(bpe, agreeing));
            assertNotNull(HuggingFaceTokenizerLoader.fromLocal(file), agreeing);
        }
    }

    private static String added(int id, String content) {
        return "{\"id\":" + id + ",\"content\":\"" + content + "\",\"special\":true}";
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
