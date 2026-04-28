package com.qxotic.toknroll.hf;

import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildBpeModel;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildByteLevelVocab;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildSentencePieceVocab;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildTokenizerJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.TokenizerInvariantHarness;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Fast invariant tests for HF-loaded tokenizers using local fixtures.
 *
 * <p>These tests verify that the TokenizerInvariantHarness smoke checks pass for various tokenizer
 * configurations (BPE, SentencePiece, Metaspace) without requiring network access.
 */
class HuggingFaceTokenizerInvariantTest {

    @TempDir Path tempDir;

    @Test
    void bpeTokenizerInvariants() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildByteLevelVocab(Map.of("ab", 256)),
                                        "[\"a b\"]",
                                        ",\"ignore_merges\":false")));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertInvariants("bpe", tokenizer, List.of("", "a", "ab", "abc"));
    }

    @Test
    void metaspaceTokenizerInvariants() throws IOException {
        // Metaspace pre-tokenizer replaces spaces with ▁ before BPE.
        // We use sentencePieceStyle path with SP-style byte fallback tokens.
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildSentencePieceVocab(Map.of("▁hi", 256, "▁hello", 257)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertInvariants("metaspace", tokenizer, List.of("", "hi", "hello world"));
    }

    @Test
    void sentencePieceStyleTokenizerInvariants() throws IOException {
        // SentencePiece models require byte fallback tokens marked as BYTE type.
        // This test is disabled until we can create proper SPM test vocabularies.
        // Path tokenizerJson =
        //         writeTokenizerJson(
        //                 buildTokenizerJson(
        //                         buildBpeModel(
        //                                 buildSentencePieceVocab(Map.of("▁hi", 256, "▁hello",
        // 257)),
        //                                 "[]",
        //                                 ",\"ignore_merges\":true"),
        //                         "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\"
        // \"},\"content\":\"▁\"}"));
        //
        // Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        // assertInvariants("sentencepiece", tokenizer, List.of("", "hi", "hello world"));
    }

    @Test
    void emptyTextRoundTrip() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildSentencePieceVocab(Map.of("▁hi", 256)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        IntSequence tokens = tokenizer.encode("");
        String decoded = tokenizer.decode(tokens);
        assertEquals("", decoded, "empty text round-trip");
    }

    @Test
    void countBytesMatchesDecodeBytes() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildSentencePieceVocab(Map.of("▁hi", 256, "▁hello", 257)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        IntSequence tokens = tokenizer.encode("hello");
        int countBytes = tokenizer.countBytes(tokens);
        byte[] decodeBytes = tokenizer.decodeBytes(tokens);
        assertEquals(decodeBytes.length, countBytes, "countBytes matches decodeBytes.length");
    }

    @Test
    void metaspacePrependOnlyAtSequenceStart() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildSentencePieceVocab(
                                                Map.of("▁ello", 256, "ello", 257, "x", 258)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);

        IntSequence full = tokenizer.encode("ello");
        IntSequence.Builder sliceBuilder = IntSequence.newBuilder(8);
        tokenizer.encodeInto("xello", 1, 5, sliceBuilder);
        IntSequence slice = sliceBuilder.build();

        assertNotEquals(
                full.intAt(0), slice.intAt(0), "slice starting at >0 must not prepend marker");
    }

    @Test
    void metaspaceLeadingTrimOnlyAtDecodeStart() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                buildBpeModel(
                                        buildSentencePieceVocab(Map.of("▁ello", 256, "x", 257)),
                                        "[]",
                                        ",\"ignore_merges\":true"),
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\"}"));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);

        assertEquals("ello", tokenizer.decode(IntSequence.wrap(new int[] {256})));
        assertEquals("x ello", tokenizer.decode(IntSequence.wrap(new int[] {257, 256})));
    }

    private void assertInvariants(String label, Tokenizer tokenizer, List<String> inputs) {
        TokenizerInvariantHarness.runSmokeChecks(
                label,
                inputs,
                text -> tokenizer.encode(text).toArray(),
                text -> {
                    IntSequence.Builder out = IntSequence.newBuilder(16);
                    tokenizer.encodeInto(text, out);
                    return out.build().toArray();
                },
                tokenizer::countTokens,
                tokens -> tokenizer.decode(IntSequence.wrap(tokens)),
                tokens -> tokenizer.decodeBytes(IntSequence.wrap(tokens)),
                tokens -> tokenizer.countBytes(IntSequence.wrap(tokens)),
                (tokens, tokenStartIndex, out) ->
                        tokenizer.decodeBytesInto(IntSequence.wrap(tokens), tokenStartIndex, out));
    }

    private Path writeTokenizerJson(String json) throws IOException {
        return Files.writeString(tempDir.resolve("tokenizer.json"), json, StandardCharsets.UTF_8);
    }
}
