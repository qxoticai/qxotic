package com.qxotic.toknroll;

import com.qxotic.toknroll.testkit.TiktokenFixtures;
import com.qxotic.toknroll.testkit.TokenizerAssertions;
import java.util.stream.Stream;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class TokenizerRegexRealWorldTest {

    @ParameterizedTest(name = "code/json/url/email {0}")
    @MethodSource("tokenizers")
    void commonStructuredTextRoundTrips(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {
                    "def hello():\n    return 'world'",
                    "{\"key\": \"value\", \"num\": 42}",
                    "https://example.com/path?query=value&foo=bar",
                    "user.name+tag@example.co.uk"
                },
                namedTokenizer.name());
    }

    @ParameterizedTest(name = "math currency date time phone {0}")
    @MethodSource("tokenizers")
    void numericFormatsRoundTrip(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {
                    "E = mc^2", "Price: $1,234.56", "2024-01-15", "14:30:00", "+1-234-567-8900"
                },
                namedTokenizer.name());
    }

    @ParameterizedTest(name = "cross-variant consistency {0}")
    @MethodSource("tokenizers")
    void representativeSamplesRoundTrip(TiktokenFixtures.NamedTokenizer namedTokenizer) {
        TokenizerAssertions.assertRoundTripSamples(
                namedTokenizer.tokenizer(),
                new String[] {"Hello world 123", "Hello 世界 👋", "I'm don't you're"},
                namedTokenizer.name());
    }

    private static Stream<TiktokenFixtures.NamedTokenizer> tokenizers() {
        return TiktokenFixtures.createAllJtokkitTokenizers().stream();
    }
}
