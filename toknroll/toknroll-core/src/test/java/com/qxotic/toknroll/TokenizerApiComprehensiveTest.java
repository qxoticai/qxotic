package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerApiComprehensiveTest {

    @ParameterizedTest(name = "empty input {0}")
    @MethodSource("tokenizers")
    void emptyInputRoundTrips(String name, Tokenizer tokenizer) {
        IntSequence tokens = tokenizer.encode("");
        assertEquals(0, tokens.length(), name + " empty token length");
        assertArrayEquals(new int[0], tokenizer.encodeToArray(""), name + " empty encodeToArray");
        assertEquals("", tokenizer.decode(tokens), name + " empty decode");
        assertEquals("", tokenizer.decode(new int[0]), name + " empty decode(int[])");
    }

    @ParameterizedTest(name = "array convenience parity {0}")
    @MethodSource("tokenizers")
    void arrayConvenienceMethodsMatchCore(String name, Tokenizer tokenizer) {
        String text = "Array parity ✅ with unicode こんにちは and symbols <>[]{}";
        IntSequence tokens = tokenizer.encode(text);
        int[] tokenArray = tokenizer.encodeToArray(text);

        assertArrayEquals(tokens.toArray(), tokenArray, name + " encodeToArray parity");
        assertEquals(
                tokenizer.decode(tokens), tokenizer.decode(tokenArray), name + " decode parity");
        assertArrayEquals(
                tokenizer.decodeBytes(tokens),
                tokenizer.decodeBytes(tokenArray),
                name + " decodeBytes parity");
    }

    @ParameterizedTest(name = "count consistency corpus {0}")
    @MethodSource("tokenizers")
    void countMatchesEncodeLengthAcrossCorpus(String name, Tokenizer tokenizer) {
        String[] corpus = {
            "simple",
            "Whitespace\tand\nnewlines",
            "Combining: e\u0301 o\u0308 n\u0303",
            "Emoji: 😀🎉✨🔥",
            "Mixed: Hello世界مرحباשלום"
        };

        for (String text : corpus) {
            assertEquals(
                    tokenizer.encode(text).length(),
                    tokenizer.countTokens(text),
                    name + " count consistency for: " + text);
        }
    }

    @ParameterizedTest(name = "null handling {0}")
    @MethodSource("tokenizers")
    void nullHandlingIsExplicit(String name, Tokenizer tokenizer) {
        assertThrows(NullPointerException.class, () -> tokenizer.decode((IntSequence) null), name);
        assertThrows(NullPointerException.class, () -> tokenizer.decode((int[]) null), name);
        assertThrows(NullPointerException.class, () -> tokenizer.decodeBytes((int[]) null), name);
    }

    @ParameterizedTest(name = "encoded tokens exist in vocabulary {0}")
    @MethodSource("tokenizers")
    void encodedTokensExistInVocabulary(String name, Tokenizer tokenizer) {
        IntSequence tokens = tokenizer.encode("Vocabulary containment check");
        for (int i = 0; i < tokens.length(); i++) {
            assertTrue(
                    tokenizer.vocabulary().contains(tokens.intAt(i)),
                    name + " token in vocab @" + i);
        }
    }

    private static Stream<Arguments> tokenizers() {
        Stream<Arguments> bpe =
                Stream.of(Arguments.of("bpe-r50k", TiktokenFixtures.createBpeR50kTokenizer()));
        Stream<String> encodingNames =
                Stream.of("r50k_base", "p50k_base", "p50k_edit", "cl100k_base", "o200k_base");
        Stream<Arguments> jtokkit =
                encodingNames.map(
                        name ->
                                Arguments.of(
                                        "jtokkit-" + name,
                                        TiktokenFixtures.createJtokkitTokenizer(name)));
        return Stream.concat(bpe, jtokkit);
    }
}
