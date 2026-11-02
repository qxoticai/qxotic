package com.qxotic.toknroll.hf;

import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.GEMMA_DECODER;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.MISTRAL_DECODER;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.SENTENCEPIECE_MODEL;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildBpeModel;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildByteLevelVocab;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildTokenizerJson;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
                                SENTENCEPIECE_MODEL,
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}",
                                MISTRAL_DECODER));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertInvariants("metaspace", tokenizer, List.of("", "hi", "hello world"));
    }

    /** Gemma's shape: a normalizer spells the spaces, no Metaspace and no stripping decoder. */
    @Test
    void sentencePieceStyleTokenizerInvariants() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                SENTENCEPIECE_MODEL,
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                        + " \"},\"content\":\"▁\"}",
                                GEMMA_DECODER));
        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
        assertInvariants("sentencepiece", tokenizer, List.of("", "hi", "hello world", " x"));
        assertEquals(" ello", tokenizer.decode(IntSequence.wrap(new int[] {266})));
    }

    @Test
    void emptyTextRoundTrip() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                SENTENCEPIECE_MODEL,
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}",
                                MISTRAL_DECODER));

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
                                SENTENCEPIECE_MODEL,
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}",
                                MISTRAL_DECODER));

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
                                SENTENCEPIECE_MODEL,
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\",\"prepend_scheme\":\"first\"}",
                                MISTRAL_DECODER));

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
                                SENTENCEPIECE_MODEL,
                                "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\""
                                    + " \"},\"content\":\"▁\"},\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}",
                                MISTRAL_DECODER));

        Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);

        assertEquals("ello", tokenizer.decode(IntSequence.wrap(new int[] {266})));
        assertEquals("x ello", tokenizer.decode(IntSequence.wrap(new int[] {262, 266})));
    }

    @Test
    void sentencePieceWithIgnoreMergesIsRefused() throws IOException {
        Path tokenizerJson =
                writeTokenizerJson(
                        buildTokenizerJson(
                                SENTENCEPIECE_MODEL.replace(
                                        ",\"byte_fallback\"",
                                        ",\"ignore_merges\":true,\"byte_fallback\""),
                                "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}"));
        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(tokenizerJson));
        assertTrue(refused.getMessage().contains("ignore_merges"), refused.getMessage());
    }

    /**
     * Ranked merging, as Hugging Face: merges b+c, a+b, ab+c, abc+d make "abcd" a, bc, d, though
     * "abcd" is a token; only ignore_merges takes a word that is a token whole. The external review
     * of 2026-11-02 found the whole-token shortcut taken regardless on main.
     */
    @Test
    void mergesKeepTheirRankEvenWhenTheWordIsAToken() throws IOException {
        for (boolean ignoreMerges : new boolean[] {false, true}) {
            Path tokenizerJson =
                    writeTokenizerJson(
                            buildTokenizerJson(
                                    buildBpeModel(
                                            buildByteLevelVocab(
                                                    Map.of(
                                                            "bc", 256, "ab", 257, "abc", 258,
                                                            "abcd", 259)),
                                            "[\"b c\",\"a b\",\"ab c\",\"abc d\"]",
                                            ",\"ignore_merges\":" + ignoreMerges),
                                    "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false,"
                                        + "\"use_regex\":true}"));
            Tokenizer tokenizer = HuggingFaceTokenizerLoader.fromLocal(tokenizerJson);
            // Hugging Face tokenizers 0.23.2 on the same file
            List<String> abc = ignoreMerges ? List.of("abc") : List.of("a", "bc");
            List<String> abcd = ignoreMerges ? List.of("abcd") : List.of("a", "bc", "d");
            assertEquals(abc, spellings(tokenizer, "abc"), "ignore_merges " + ignoreMerges);
            assertEquals(abcd, spellings(tokenizer, "abcd"), "ignore_merges " + ignoreMerges);
            assertEquals(List.of("a", "bc", "d", "x"), spellings(tokenizer, "abcdx"));
        }
    }

    private static List<String> spellings(Tokenizer tokenizer, String text) {
        return tokenizer.encode(text).stream().mapToObj(tokenizer.vocabulary()::token).toList();
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

        // Slicing invariants: 4-arg encodeInto and 3-arg countTokens
        TokenizerInvariantHarness.runSlicingInvariants(
                label,
                inputs,
                text -> tokenizer.encode(text).toArray(),
                (text, start, end) -> {
                    IntSequence.Builder out = IntSequence.newBuilder(16);
                    tokenizer.encodeInto(text, start, end, out);
                    return out.build().toArray();
                },
                tokenizer::countTokens,
                tokenizer::countTokens);

        // Convenience overload parity
        TokenizerInvariantHarness.runConvenienceOverloadInvariants(
                label,
                inputs,
                text -> tokenizer.encode(text).toArray(),
                tokenizer::encodeToArray,
                tokens -> tokenizer.decode(IntSequence.wrap(tokens)),
                tokenizer::decode,
                tokens -> tokenizer.decodeBytes(IntSequence.wrap(tokens)),
                tokenizer::decodeBytes,
                tokens -> tokenizer.countBytes(IntSequence.wrap(tokens)),
                tokenizer::countBytes);

        TokenizerInvariantHarness.assertExpectedTokensPerCharRange(
                label, tokenizer.expectedTokensPerChar());
    }

    private Path writeTokenizerJson(String json) throws IOException {
        return Files.writeString(tempDir.resolve("tokenizer.json"), json, StandardCharsets.UTF_8);
    }
}
