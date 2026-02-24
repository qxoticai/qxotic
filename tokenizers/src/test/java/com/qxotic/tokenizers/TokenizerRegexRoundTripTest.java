package com.qxotic.tokenizers;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.tokenizers.testkit.TiktokenFixtures;
import com.qxotic.tokenizers.testkit.TokenizerAssertions;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerRegexRoundTripTest {

    @ParameterizedTest(name = "contractions {0}")
    @MethodSource("tokenizers")
    void contractionsRoundTrip(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {
                    "don't", "won't", "can't", "it's", "I'm", "you'll", "I'd", "you're", "I've"
                },
                namedTokenizer.name());
    }

    @ParameterizedTest(name = "whitespace {0}")
    @MethodSource("tokenizers")
    void whitespaceRoundTrip(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {
                    "hello world",
                    "hello  world",
                    "hello\tworld",
                    "hello\nworld",
                    "hello\r\nworld",
                    ""
                },
                namedTokenizer.name());
    }

    @ParameterizedTest(name = "numbers {0}")
    @MethodSource("tokenizers")
    void numberRoundTrip(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {"1", "123", "1234567890", "3.14", "123.456"},
                namedTokenizer.name());
    }

    @ParameterizedTest(name = "punctuation {0}")
    @MethodSource("tokenizers")
    void punctuationRoundTrip(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {
                    "Hello.",
                    "(hello)",
                    "[hello]",
                    "{hello}",
                    "\"hello\"",
                    "test@example.com",
                    "a+b=c"
                },
                namedTokenizer.name());
    }

    @ParameterizedTest(name = "unicode {0}")
    @MethodSource("tokenizers")
    void unicodeRoundTrip(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {
                    "café", "你好世界", "مرحبا", "שלום", "Привет", "Γειά", "👋", "Hello世界", "e\u0301"
                },
                namedTokenizer.name());
    }

    @ParameterizedTest(name = "long and repeated {0}")
    @MethodSource("tokenizers")
    void longAndRepeatedRoundTrip(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Tokenizer tokenizer = namedTokenizer.tokenizer();
        String longWord = "a".repeat(1000);
        IntSequence longTokens = tokenizer.encode(longWord);
        assertTrue(longTokens.length() > 0, namedTokenizer.name());
        assertEquals(longWord, tokenizer.decode(longTokens), namedTokenizer.name());

        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {"abababab", "xyzxyzxyz", "123123123"},
                namedTokenizer.name());
    }

    private static Stream<TiktokenFixtures.NamedTokenizer> tokenizers() {
        return TiktokenFixtures.createAllJtokkitTokenizers().stream();
    }
}
