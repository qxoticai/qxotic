package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.testkit.TiktokenFixtures;
import com.qxotic.toknroll.testkit.TokenizerAssertions;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerUnicodeCornerCaseTest {

    private static final String[] ROUND_TRIP_TEXTS = {
        "",
        " ",
        "Hello world",
        "こんにちは世界",
        "مرحبا بالعالم",
        "שלום עולם",
        "Привет мир",
        "Γειά σου Κόσμε",
        "नमस्ते दुनिया",
        "你好世界",
        "😀🎉✨❤️🔥",
        "👨‍👩‍👧‍👦",
        "👍🏽",
        "e\u0301",
        "a\u200Bb",
        "Hello世界مرحباשלום"
    };

    @ParameterizedTest(name = "unicode round-trip {0}")
    @MethodSource("tokenizers")
    void unicodeAndEdgeTextRoundTrips(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(), ROUND_TRIP_TEXTS, namedTokenizer.name());
    }

    @ParameterizedTest(name = "unicode bytes {0}")
    @MethodSource("tokenizers")
    void decodeBytesMatchesUtf8(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        String text = "Byte-level unicode: 你好مرحبا 👨‍👩‍👧‍👦";
        TokenizerAssertions.assertDecodeBytesUtf8(
                namedTokenizer.tokenizer(), text, namedTokenizer.name());
    }

    @ParameterizedTest(name = "long text {0}")
    @MethodSource("tokenizers")
    void longInputRoundTrips(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        Tokenizer tokenizer = namedTokenizer.tokenizer();
        String text = "ab".repeat(5000);
        IntSequence tokens = tokenizer.encode(text);
        assertTrue(tokens.length() > 0, namedTokenizer.name() + " should produce tokens");
        assertEquals(text, tokenizer.decode(tokens), namedTokenizer.name() + " long round-trip");
    }

    private static Stream<TiktokenFixtures.NamedTokenizer> tokenizers() {
        return TiktokenFixtures.createAllJtokkitTokenizers().stream();
    }
}
