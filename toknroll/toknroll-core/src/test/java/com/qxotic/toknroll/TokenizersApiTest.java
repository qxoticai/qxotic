package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.impl.ClassicBPE;
import com.qxotic.toknroll.impl.RegexSplitter;
import java.net.URISyntaxException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.text.Normalizer.Form;
import java.util.Map;
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
    void classicBpeFacadeBuildsWorkingTokenizer() throws Exception {
        Map<String, Integer> mergeableRanks =
                ClassicBPE.loadMergeableRanks(resourcePath(R50K_FILE).toString(), R50K_HASH);

        Tokenizer tokenizer =
                Tokenizers.classicBpe(
                        mergeableRanks, R50K_SPECIALS, RegexSplitter.create(R50K_PATTERN));

        String text = "Tokenizer facade test";
        IntSequence tokens = tokenizer.encode(text);
        assertEquals(text, tokenizer.decode(tokens));
        assertEquals(tokens.length(), tokenizer.countTokens(text));
    }

    @Test
    void classicBpeRegexStringOverloadWorks() throws Exception {
        Map<String, Integer> mergeableRanks =
                ClassicBPE.loadMergeableRanks(resourcePath(R50K_FILE).toString(), R50K_HASH);

        Tokenizer tokenizer = Tokenizers.classicBpe(mergeableRanks, R50K_SPECIALS, R50K_PATTERN);
        String text = "Classic overload";
        assertEquals(text, tokenizer.decode(tokenizer.encode(text)));
    }

    @Test
    void fastBpeFacadeBuildsWorkingTokenizer() throws Exception {
        Map<String, Integer> mergeableRanks =
                ClassicBPE.loadMergeableRanks(resourcePath(R50K_FILE).toString(), R50K_HASH);

        Tokenizer tokenizer =
                Tokenizers.fastBpe(
                        mergeableRanks, R50K_SPECIALS, RegexSplitter.create(R50K_PATTERN));

        String text = "Fast tokenizer facade test";
        IntSequence tokens = tokenizer.encode(text);
        assertEquals(text, tokenizer.decode(tokens));
        assertEquals(tokens.length(), tokenizer.countTokens(text));
    }

    @Test
    void tokenizersFactoryMethodsValidateNulls() {
        Map<String, Integer> ranks = Map.of("a", 0);
        Tokenizer tokenizer =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        assertThrows(
                NullPointerException.class,
                () -> Tokenizers.classicBpe(ranks, Map.of(), (String) null));
        assertThrows(
                NullPointerException.class,
                () -> Tokenizers.fastBpe(ranks, Map.of(), (String) null));
        assertThrows(
                NullPointerException.class,
                () -> Tokenizers.withTextTransform(null, Normalizer.identity()));
        assertThrows(
                NullPointerException.class, () -> Tokenizers.withTextTransform(tokenizer, null));
        assertThrows(NullPointerException.class, () -> tokenizer.decode((IntSequence) null));
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
    void lossyTransformIsExplicitAndSeparateFromBaseTokenizer() {
        Tokenizer base =
                createTokenizer(R50K_NAME, R50K_FILE, R50K_HASH, R50K_PATTERN, R50K_SPECIALS);
        Tokenizer nfcTransformed = Tokenizers.withTextTransform(base, Normalizer.unicode(Form.NFC));

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
                    ClassicBPE.loadMergeableRanks(resourcePath(file).toString(), hash);
            return Tokenizers.fastBpe(mergeableRanks, specials, pattern);
        } catch (Exception e) {
            throw new IllegalStateException("Failed to create tokenizer " + name, e);
        }
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
