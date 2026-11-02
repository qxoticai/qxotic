package com.qxotic.toknroll.hf;

import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.GEMMA_DECODER;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.MISTRAL_DECODER;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.SENTENCEPIECE_MODEL;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildBpeModel;
import static com.qxotic.toknroll.hf.HuggingFaceTokenizerTestFixtures.buildTokenizerJson;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * What a SentencePiece tokenizer.json may hold: the pre-tokenizer stages and decoder steps toknroll
 * reproduces load, with Hugging Face's ids (tokenizers 0.23.2), and any other is refused rather
 * than dropped.
 */
class HuggingFaceSentencePieceRulesTest {

    private static final String SPELL_SPACES =
            "\"normalizer\":{\"type\":\"Replace\",\"pattern\":{\"String\":\" \"},"
                    + "\"content\":\"▁\"}";

    @TempDir Path tempDir;

    @Test
    void metaspaceBesideAnotherStageIsRefused() throws IOException {
        assertRefused(
                "pre_tokenizer",
                SENTENCEPIECE_MODEL,
                "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
                        + "{\"type\":\"Metaspace\",\"replacement\":\"▁\"},"
                        + "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"\\\\p{N}\"},"
                        + "\"behavior\":\"Isolated\",\"invert\":false}]}",
                MISTRAL_DECODER);
    }

    @Test
    void byteLevelAfterASpaceSpellingNormalizerIsRefused() throws IOException {
        assertRefused(
                "pre_tokenizer",
                SENTENCEPIECE_MODEL,
                SPELL_SPACES,
                "\"pre_tokenizer\":{\"type\":\"ByteLevel\",\"add_prefix_space\":false}",
                GEMMA_DECODER);
    }

    /** Without the normalizer, a Split at ' ' would cut the text before Metaspace. */
    @Test
    void splitAtSpaceWithoutTheNormalizerIsRefused() throws IOException {
        assertRefused(
                "pre_tokenizer",
                SENTENCEPIECE_MODEL,
                "\"pre_tokenizer\":{\"type\":\"Sequence\",\"pretokenizers\":["
                        + "{\"type\":\"Split\",\"pattern\":{\"String\":\" \"},"
                        + "\"behavior\":\"MergedWithPrevious\",\"invert\":false},"
                        + "{\"type\":\"Metaspace\",\"replacement\":\"▁\"}]}",
                MISTRAL_DECODER);
    }

    @Test
    void metaspaceWithAnotherReplacementIsRefused() throws IOException {
        assertRefused(
                "replacement",
                SENTENCEPIECE_MODEL,
                "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"_\"}",
                MISTRAL_DECODER);
    }

    /** Gemma's shape: the Split at ' ' never matches once the normalizer spelled the spaces. */
    @Test
    void gemmasSplitAtSpaceLoads() throws IOException {
        Tokenizer tokenizer =
                load(
                        SENTENCEPIECE_MODEL,
                        SPELL_SPACES,
                        "\"pre_tokenizer\":{\"type\":\"Split\",\"pattern\":{\"String\":\" \"},"
                                + "\"behavior\":\"MergedWithPrevious\",\"invert\":false}",
                        GEMMA_DECODER);
        assertEncodes(tokenizer, "hi ello", 260, 261, 266);
        assertEncodes(tokenizer, "ello hi", 265, 268);
        assertEncodes(tokenizer, " hi", 268);
        assertEquals(" hi", tokenizer.decode(IntSequence.of(268)));
    }

    @Test
    void prependSchemeNeverPrependsNothing() throws IOException {
        Tokenizer tokenizer =
                load(
                        SENTENCEPIECE_MODEL,
                        "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\","
                                + "\"prepend_scheme\":\"never\"}",
                        MISTRAL_DECODER);
        assertEncodes(tokenizer, "hi ello", 260, 261, 266);
        assertEncodes(tokenizer, "ello hi", 265, 268);
        assertEncodes(tokenizer, " hi", 268);
    }

    /** The spelling before prepend_scheme: false is never, true is always. */
    @Test
    void addPrefixSpaceIsTheOlderPrependScheme() throws IOException {
        Tokenizer never =
                load(
                        SENTENCEPIECE_MODEL,
                        "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\","
                                + "\"add_prefix_space\":false}",
                        MISTRAL_DECODER);
        assertEncodes(never, "hi ello", 260, 261, 266);
        assertEncodes(never, "ello hi", 265, 268);
        Tokenizer always =
                load(
                        SENTENCEPIECE_MODEL,
                        "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\","
                                + "\"add_prefix_space\":true}",
                        MISTRAL_DECODER);
        assertEncodes(always, "hi ello", 268, 266);
        assertEncodes(always, "ello hi", 266, 268);
        assertEncodes(always, " hi", 268);
    }

    /**
     * With split false Hugging Face merges the text whole, so a token may span a space: "a▁b" is
     * one token here, where split true cuts before every ▁.
     */
    @Test
    void metaspaceSplitFalseMergesAcrossSpaces() throws IOException {
        String model =
                buildBpeModel(
                        "{\"<unk>\":0,\"▁\":1,\"a\":2,\"b\":3,\"a▁\":4,\"a▁b\":5,\"▁a\":6,"
                                + "\"▁b\":7}",
                        "[\"a ▁\",\"a▁ b\",\"▁ a\",\"▁ b\"]",
                        ",\"unk_token\":\"<unk>\"");
        String metaspace =
                "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\","
                        + "\"prepend_scheme\":\"first\",\"split\":";
        Tokenizer whole = load(model, metaspace + "false}", MISTRAL_DECODER);
        assertEncodes(whole, "a b", 1, 5);
        assertEncodes(whole, "a b a b", 1, 5, 1, 5);
        assertEquals("a b", whole.decode(IntSequence.of(1, 5)));
        Tokenizer split = load(model, metaspace + "true}", MISTRAL_DECODER);
        assertEncodes(split, "a b", 6, 7);
        assertEncodes(split, "a b a b", 6, 7, 6, 7);
    }

    @Test
    void stripOfTheEndIsRefused() throws IOException {
        assertRefusedDecoder(
                "{\"type\":\"Replace\",\"pattern\":{\"String\":\"▁\"},\"content\":\" \"},"
                        + "{\"type\":\"Fuse\"},"
                        + "{\"type\":\"Strip\",\"content\":\" \",\"start\":1,\"stop\":1}");
    }

    @Test
    void stripOfMoreThanOneLeadingSpaceIsRefused() throws IOException {
        assertRefusedDecoder(
                "{\"type\":\"Replace\",\"pattern\":{\"String\":\"▁\"},\"content\":\" \"},"
                        + "{\"type\":\"Fuse\"},"
                        + "{\"type\":\"Strip\",\"content\":\" \",\"start\":2,\"stop\":0}");
    }

    /** Before a Fuse, Hugging Face strips every token's leading space, not the text's. */
    @Test
    void stripBeforeFuseIsRefused() throws IOException {
        assertRefusedDecoder(
                "{\"type\":\"Replace\",\"pattern\":{\"String\":\"▁\"},\"content\":\" \"},"
                        + "{\"type\":\"Strip\",\"content\":\" \",\"start\":1,\"stop\":0},"
                        + "{\"type\":\"Fuse\"}");
    }

    @Test
    void replaceWithOtherContentIsRefused() throws IOException {
        assertRefusedDecoder(
                "{\"type\":\"Replace\",\"pattern\":{\"String\":\"▁\"},\"content\":\"_\"},"
                        + "{\"type\":\"Fuse\"}");
    }

    @Test
    void anotherDecoderTypeIsRefused() throws IOException {
        assertRefusedDecoder(
                "{\"type\":\"Replace\",\"pattern\":{\"String\":\"▁\"},\"content\":\" \"},"
                        + "{\"type\":\"WordPiece\",\"prefix\":\"##\",\"cleanup\":true}");
    }

    /** Without a decoder Hugging Face joins the tokens with spaces and keeps every ▁. */
    @Test
    void missingDecoderIsRefused() throws IOException {
        assertRefused(
                "decoder",
                SENTENCEPIECE_MODEL,
                "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}");
    }

    @Test
    void metaspaceDecoderDropsThePrependedSpace() throws IOException {
        Tokenizer tokenizer =
                load(
                        SENTENCEPIECE_MODEL,
                        "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}",
                        "\"decoder\":{\"type\":\"Sequence\",\"decoders\":["
                                + "{\"type\":\"Metaspace\",\"replacement\":\"▁\","
                                + "\"prepend_scheme\":\"always\"},"
                                + "{\"type\":\"ByteFallback\"},{\"type\":\"Fuse\"}]}");
        assertEquals("hi ello", tokenizer.decode(IntSequence.of(268, 266)));
    }

    private void assertRefusedDecoder(String steps) throws IOException {
        assertRefused(
                "decoder",
                SENTENCEPIECE_MODEL,
                "\"pre_tokenizer\":{\"type\":\"Metaspace\",\"replacement\":\"▁\"}",
                "\"decoder\":{\"type\":\"Sequence\",\"decoders\":[" + steps + "]}");
    }

    private void assertRefused(String where, String model, String... rootFields)
            throws IOException {
        Path file = write(model, rootFields);
        IllegalArgumentException refused =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> HuggingFaceTokenizerLoader.fromLocal(file));
        assertTrue(refused.getMessage().contains(where), refused.getMessage());
    }

    private static void assertEncodes(Tokenizer tokenizer, String text, int... ids) {
        assertArrayEquals(ids, tokenizer.encode(text).toArray(), text);
    }

    private Tokenizer load(String model, String... rootFields) throws IOException {
        return HuggingFaceTokenizerLoader.fromLocal(write(model, rootFields));
    }

    private Path write(String model, String... rootFields) throws IOException {
        return Files.writeString(
                tempDir.resolve("tokenizer.json"),
                buildTokenizerJson(model, rootFields),
                StandardCharsets.UTF_8);
    }
}
