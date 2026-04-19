package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.impl.TiktokenFiles;
import com.qxotic.toknroll.impl.TiktokenReconstruction;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.text.Normalizer.Form;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class TokenizersApiTest {

    private static final String R50K_NAME = "r50k_base";
    private static final String R50K_FILE = "r50k_base.tiktoken";
    private static final String R50K_HASH =
            "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";
    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";
    private static final Map<String, Integer> R50K_SPECIALS = Map.of("<|endoftext|>", 50256);

    @Test
    void advancedPipelineBuilderIsAvailableViaFacade() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        assertNotNull(tokenizer);
    }

    @Test
    void tokenizerArrayConvenienceMethodsMatchSequenceMethods() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        String text = "Hello tokenizer API";

        IntSequence sequenceTokens = tokenizer.encode(text);
        int[] arrayTokens = tokenizer.encodeToArray(text);
        assertArrayEquals(sequenceTokens.toArray(), arrayTokens);

        assertEquals(tokenizer.decode(sequenceTokens), tokenizer.decode(arrayTokens));
        assertArrayEquals(
                tokenizer.decodeBytes(sequenceTokens), tokenizer.decodeBytes(arrayTokens));
    }

    @Test
    void encodeIntoAppendsToBuilder() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        String text = "encode into";

        IntSequence.Builder out = IntSequence.newBuilder();
        tokenizer.encodeInto(text, out);

        assertArrayEquals(tokenizer.encode(text).toArray(), out.build().toArray());
        assertThrows(NullPointerException.class, () -> tokenizer.encodeInto(text, null));
        assertThrows(
                NullPointerException.class, () -> tokenizer.encodeInto((CharSequence) null, out));
    }

    @Test
    void pipelineFacadeBuildsWorkingTokenizer() throws Exception {
        Map<String, Integer> mergeableRanks =
                TiktokenFiles.loadMergeableRanks(resourcePath(R50K_FILE).toString(), R50K_HASH);

        Tokenizer tokenizer =
                createTikTokenTokenizer(
                        mergeableRanks,
                        R50K_SPECIALS,
                        Splitter.regex(
                                Pattern.compile(R50K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS)));

        String text = "Tokenizer facade test";
        IntSequence tokens = tokenizer.encode(text);
        assertEquals(text, tokenizer.decode(tokens));
        assertEquals(tokens.length(), tokenizer.countTokens(text));
    }

    @Test
    void pipelineWithRegexSplitterWorks() throws Exception {
        Map<String, Integer> mergeableRanks =
                TiktokenFiles.loadMergeableRanks(resourcePath(R50K_FILE).toString(), R50K_HASH);

        Tokenizer tokenizer =
                createTikTokenTokenizer(
                        mergeableRanks,
                        R50K_SPECIALS,
                        Splitter.regex(
                                Pattern.compile(R50K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS)));
        String text = "Classic overload";
        assertEquals(text, tokenizer.decode(tokenizer.encode(text)));
    }

    @Test
    void tikTokenModelFacadeBuildsWorkingTokenizer() throws Exception {
        Map<String, Integer> mergeableRanks =
                TiktokenFiles.loadMergeableRanks(resourcePath(R50K_FILE).toString(), R50K_HASH);

        Tokenizer tokenizer =
                createTikTokenTokenizer(
                        mergeableRanks,
                        R50K_SPECIALS,
                        Splitter.regex(
                                Pattern.compile(R50K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS)));

        String text = "Fast tokenizer facade test";
        IntSequence tokens = tokenizer.encode(text);
        assertEquals(text, tokenizer.decode(tokens));
        assertEquals(tokens.length(), tokenizer.countTokens(text));
    }

    @Test
    void tokenizersFactoryMethodsValidateNulls() {
        Vocabulary vocabulary = Tokenizers.vocabulary("a");
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        assertThrows(NullPointerException.class, () -> Tokenizers.tikTokenModel(null, List.of()));
        assertThrows(
                NullPointerException.class,
                () -> Tokenizers.tikTokenModel(vocabulary, (List<Tokenizers.MergeRule>) null));
        assertThrows(NullPointerException.class, () -> Tokenizers.pipeline(null));
        assertThrows(NullPointerException.class, () -> tokenizer.decode((IntSequence) null));
    }

    @Test
    void vocabularyVarargsAssignsIdsByPosition() {
        Vocabulary vocabulary = Tokenizers.vocabulary("a", "b", "c");

        assertEquals(3, vocabulary.size());
        assertEquals(0, vocabulary.id("a"));
        assertEquals(1, vocabulary.id("b"));
        assertEquals(2, vocabulary.id("c"));
        assertEquals("b", vocabulary.token(1));
    }

    @Test
    void vocabularyVarargsWithSpecialTokensIncludesControls() {
        Vocabulary vocabulary = Tokenizers.vocabulary(Map.of("<|eot|>", 10), "a", "b", "c");

        assertEquals(4, vocabulary.size());
        assertEquals(10, vocabulary.id("<|eot|>"));
        assertEquals("<|eot|>", vocabulary.token(10));
        assertEquals(2, vocabulary.id("c"));
    }

    @Test
    void vocabularyVarargsRejectsDuplicatesAndSpecialOverlaps() {
        assertThrows(IllegalArgumentException.class, () -> Tokenizers.vocabulary("a", "a"));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tokenizers.vocabulary(Map.of("a", 99), "a", "b"));
        assertThrows(
                IllegalArgumentException.class,
                () -> Tokenizers.vocabulary(Map.of("<|eot|>", 1), "a", "b"));
    }

    @Test
    void tokenizerIsDeterministicAndLosslessForTextRoundTrip() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        String text = "  Keep all whitespace\n\tand symbols <>[]{}  ";

        IntSequence first = tokenizer.encode(text);
        IntSequence second = tokenizer.encode(text);

        assertArrayEquals(first.toArray(), second.toArray());
        assertEquals(text, tokenizer.decode(first));
        assertEquals(first.length(), tokenizer.countTokens(text));
    }

    @Test
    void decodeBytesMatchesUtf8ForEncodedText() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        String text = "Fidelity: 😀 data\nline2";
        IntSequence tokens = tokenizer.encode(text);

        assertArrayEquals(text.getBytes(StandardCharsets.UTF_8), tokenizer.decodeBytes(tokens));
    }

    @Test
    void countBytesMatchesDecodedByteLength() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        String text = "Count decoded bytes 😀 across APIs";
        IntSequence tokens = tokenizer.encode(text);

        assertEquals(tokenizer.decodeBytes(tokens).length, tokenizer.countBytes(tokens));
        assertEquals(tokenizer.decodeBytes(tokens).length, tokenizer.countBytes(tokens.toArray()));
        assertThrows(NullPointerException.class, () -> tokenizer.countBytes((IntSequence) null));
        assertThrows(NullPointerException.class, () -> tokenizer.countBytes((int[]) null));
    }

    @Test
    void decodeBytesIntoAllowsChunkedDecodeWithoutCopies() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        String text = "Chunked decode bytes 😀 with multiple tokens and symbols!";
        IntSequence tokens = tokenizer.encode(text);

        ByteBuffer out = ByteBuffer.allocate(tokenizer.decodeBytes(tokens).length);
        int index = 0;
        while (index < tokens.length()) {
            int consumed = tokenizer.decodeBytesInto(tokens, index, out);
            assertTrue(consumed > 0);
            index += consumed;
        }
        assertEquals(0, tokenizer.decodeBytesInto(tokens, tokens.length(), out));

        assertArrayEquals(tokenizer.decodeBytes(tokens), out.array());
    }

    @Test
    void decodeBytesIntoThrowsWhenBufferCannotFitNextToken() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        IntSequence tokens = tokenizer.encode("buffer overflow contract");
        ByteBuffer out = ByteBuffer.allocate(0);

        assertThrows(
                IllegalArgumentException.class, () -> tokenizer.decodeBytesInto(tokens, 0, out));
        assertEquals(0, out.position());
    }

    @Test
    void lossyTransformIsExplicitAndSeparateFromBaseTokenizer() throws Exception {
        Map<String, Integer> mergeableRanks =
                TiktokenFiles.loadMergeableRanks(resourcePath(R50K_FILE).toString(), R50K_HASH);
        Vocabulary vocabulary = TiktokenReconstruction.vocabulary(mergeableRanks, R50K_SPECIALS);
        TokenizationModel model =
                Tokenizers.tikTokenModel(
                        vocabulary, TiktokenReconstruction.mergeRules(mergeableRanks));

        Tokenizer base =
                TokenizationPipeline.builder(model)
                        .splitter(
                                Splitter.regex(
                                        Pattern.compile(
                                                R50K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS)))
                        .build();
        Tokenizer nfcTransformed =
                TokenizationPipeline.builder(model)
                        .normalizer(Normalizer.unicode(Form.NFC))
                        .splitter(
                                Splitter.regex(
                                        Pattern.compile(
                                                R50K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS)))
                        .build();

        String decomposed = "e\u0301";
        String composed = "é";

        assertEquals(decomposed, base.decode(base.encode(decomposed)));
        assertEquals(composed, nfcTransformed.decode(nfcTransformed.encode(decomposed)));
        assertArrayEquals(
                nfcTransformed.encodeToArray(decomposed), nfcTransformed.encodeToArray(composed));
    }

    @Test
    void coreCountingApisAreAvailable() {
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        String text = "core api check";
        IntSequence tokens = tokenizer.encode(text);

        assertEquals(tokens.length(), tokenizer.countTokens(text));
        assertEquals(tokenizer.decodeBytes(tokens).length, tokenizer.countBytes(tokens));
    }

    private static Tokenizer createTokenizer(
            String name, String file, String hash, String pattern, Map<String, Integer> specials) {
        try {
            Map<String, Integer> mergeableRanks =
                    TiktokenFiles.loadMergeableRanks(resourcePath(file).toString(), hash);
            return createTikTokenTokenizer(
                    mergeableRanks,
                    specials,
                    Splitter.regex(Pattern.compile(pattern, Pattern.UNICODE_CHARACTER_CLASS)));
        } catch (Exception e) {
            throw new IllegalStateException("Failed to create tokenizer " + name, e);
        }
    }

    private static Tokenizer createTikTokenTokenizer(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specials, Splitter splitter) {
        Vocabulary vocabulary = TiktokenReconstruction.vocabulary(mergeableRanks, specials);
        TokenizationModel model =
                Tokenizers.tikTokenModel(
                        vocabulary, TiktokenReconstruction.mergeRules(mergeableRanks));
        return Tokenizers.pipeline(model).splitter(splitter).build();
    }

    private static Path resourcePath(String fileName) {
        try {
            return Path.of(
                    TokenizersApiTest.class
                            .getClassLoader()
                            .getResource("tiktoken/" + fileName)
                            .toURI());
        } catch (URISyntaxException e) {
            throw new IllegalStateException("Failed to resolve " + fileName, e);
        }
    }
}
