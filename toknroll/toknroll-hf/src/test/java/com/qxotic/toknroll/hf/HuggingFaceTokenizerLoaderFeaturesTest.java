package com.qxotic.toknroll.hf;

import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildBpeModel;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildByteLevelVocab;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildTokenizerJson;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HuggingFaceTokenizerLoaderFeaturesTest {

    @TempDir Path tempDir;

    @Test
    void supportsMergesAsStringSpecs() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("ab", 256)),
                                        "[\"a b\"]",
                                        ",\"ignore_merges\":false")));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {256}, tokenizer.encode("ab").toArray());
    }

    @Test
    void supportsAddedSpecialTokensAndTokenTypes() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("<|end|>", 256)), "[]", ""),
                                "\"added_tokens\":[{\"id\":256,\"content\":\"<|end|>\",\"special\":true}]"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertEquals("<|end|>", tokenizer.vocabulary().token(256));
        assertTrue(tokenizer.vocabulary().isTokenOfType(256, StandardTokenType.CONTROL));
    }

    @Test
    void supportsUnicodeNfcNormalizer() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("é", 256)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"normalizer\":{\"type\":\"NFC\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(tokenizer.encode("é").toArray(), tokenizer.encode("e\u0301").toArray());
    }

    @Test
    void supportsReplaceNormalizer() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("é", 256)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\"x\"},\"content\":\"é\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        // With ignore_merges=true, "é" is encoded as individual byte tokens
        // since no merges are applied. The token 256 exists but won't be used.
        IntSequence encoded = tokenizer.encode("x");
        assertEquals("é", tokenizer.decode(encoded));
    }

    @Test
    void supportsSequenceNormalizer() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("é", 256)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"normalizer\":{\"type\":\"Sequence\",\"normalizers\":[{\"type\":\"Lowercase\"},{\"type\":\"NFC\"}]}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        // With ignore_merges=true, "É" -> "é" via normalizer, then encoded as byte tokens
        IntSequence encoded = tokenizer.encode("É");
        assertEquals("é", tokenizer.decode(encoded));
    }

    @Test
    void supportsSplitStringIsolated() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("a,b", 300)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\",\"},\"behavior\":\"Isolated\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {97, 44, 98}, tokenizer.encode("a,b").toArray());
    }

    @Test
    void supportsSplitStringRemoved() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(buildByteLevelVocab(Map.of()), "[]", ""),
                                "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\",\"},\"behavior\":\"Removed\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {97, 98}, tokenizer.encode("a,b").toArray());
    }

    @Test
    void supportsSplitStringMergedWithPrevious() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("a,", 301)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\",\"},\"behavior\":\"MergedWithPrevious\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {301, 98}, tokenizer.encode("a,b").toArray());
    }

    @Test
    void supportsSplitStringMergedWithNext() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of(",b", 302)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\",\"},\"behavior\":\"MergedWithNext\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {97, 302}, tokenizer.encode("a,b").toArray());
    }

    @Test
    void supportsSplitRegexWithInvert() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("ab", 256, "cd", 257)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"Regex\":\"[a-z]+\"},\"behavior\":\"Removed\",\"invert\":true}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {256, 257}, tokenizer.encode("ab,cd").toArray());
    }

    @Test
    void supportsSequencePreTokenizer() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(buildByteLevelVocab(Map.of()), "[]", ""),
                                "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
                                    + "{\"type\":\"Split\",\"pattern\":{\"String\":\",\"},\"behavior\":\"Isolated\"},{\"type\":\"Split\",\"pattern\":{\"String\":\":\"},\"behavior\":\"Isolated\"}]"
                                    + " }"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(
                new int[] {97, 58, 98, 44, 99, 58, 100}, tokenizer.encode("a:b,c:d").toArray());
    }

    @Test
    void supportsByteLevelPreTokenizer() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("ab", 256)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"pre_tokenizer\":{\"type\":\"ByteLevel\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertArrayEquals(new int[] {256}, tokenizer.encode("ab").toArray());
    }

    @Test
    void supportsMetaspacePreTokenizerDecodeWrapper() throws IOException {
        // Metaspace tokens use U+2581 which is not a valid byte-level symbol.
        // This test uses sentencePieceStyle path (Replace normalizer with ▁)
        // which does not require byte-level validation.
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel("{\"▁hi\":0}", "[]", ""),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertEquals("hi", tokenizer.decode(IntSequence.of(0)));
    }

    @Test
    void supportsSentencePieceStyleDecodeWrapper() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel("{\"▁hi\":0}", "[]", ""),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                        + " \"},\"content\":\"▁\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertEquals(" hi", tokenizer.decode(IntSequence.of(0)));
    }

    @Test
    void throwsOnUnsupportedNormalizerType() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel("{\"a\":0}", "[]", ""),
                                "\"normalizer\":{\"type\":\"BertNormalizer\"}"));

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(error.getMessage().contains("normalizer.type"));
    }

    @Test
    void throwsOnUnsupportedReplacePatternShape() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel("{\"a\":0}", "[]", ""),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"Regex\":\"a+\"},\"content\":\"b\"}"));

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(error.getMessage().contains("normalizer.pattern"));
    }

    @Test
    void throwsOnUnsupportedPreTokenizerBehavior() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(buildByteLevelVocab(Map.of()), "[]", ""),
                                "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\",\"},\"behavior\":\"Contiguous\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        IllegalArgumentException error =
                assertThrows(IllegalArgumentException.class, () -> tokenizer.encode("a,b"));
        assertTrue(error.getMessage().contains("pre_tokenizer.behavior"));
    }

    @Test
    void throwsOnUnsupportedSplitPatternShape() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel("{\"a\":0}", "[]", ""),
                                "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"Unknown\":\",\"}}"));

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(error.getMessage().contains("pre_tokenizer.pattern"));
    }

    @Test
    void throwsOnUnsupportedPreTokenizerType() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel("{\"a\":0}", "[]", ""),
                                "\"pre_tokenizer\":{\"type\":\"Whitespace\"}"));

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(error.getMessage().contains("pre_tokenizer.type"));
    }

    @Test
    void throwsOnUnsupportedModelType() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        "{\"model\":{\"type\":\"Unigram\",\"vocab\":{\"a\":0},\"merges\":[]}}");

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(error.getMessage().contains("model.type"));
    }

    @Test
    void throwsOnEmptyVocab() throws IOException {
        Path tokenizerJson = writeTokenizerJson(buildTokenizerJson(buildBpeModel("{}", "[]", "")));

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(error.getMessage().contains("vocab is empty"));
    }

    private Path writeTokenizerJson(String json) throws IOException {
        return Files.writeString(tempDir.resolve("tokenizer.json"), json, StandardCharsets.UTF_8);
    }
}
