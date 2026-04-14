package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class FastTikTokenLargeMergeRegressionTest {

    @ParameterizedTest(name = "large merge parity {0}")
    @ValueSource(strings = {"r50k_base", "cl100k_base", "o200k_base"})
    void fastLargePathMatchesClassicTokenizer(String encoding) {
        Tokenizer fast =
                Tokenizers.fastBpe(
                        TiktokenFixtures.mergeableRanks(encoding),
                        TiktokenFixtures.specialTokens(encoding),
                        Splitter.regex(TiktokenFixtures.splitPattern(encoding)));

        Tokenizer classic =
                Tokenizers.classicBpe(
                        TiktokenFixtures.mergeableRanks(encoding),
                        TiktokenFixtures.specialTokens(encoding),
                        Splitter.regex(TiktokenFixtures.splitPattern(encoding)));

        for (String sample : largeSamples()) {
            IntSequence expected = classic.encode(sample);
            IntSequence actual = fast.encode(sample);
            assertEquals(
                    expected,
                    actual,
                    () -> "encoding=" + encoding + " sampleHash=" + sample.hashCode());
            assertEquals(sample, fast.decode(actual), () -> "round-trip encoding=" + encoding);
            assertEquals(
                    expected.length(),
                    fast.countTokens(sample),
                    () -> "count encoding=" + encoding);
        }
    }

    private static List<String> largeSamples() {
        return List.of(
                "a".repeat(1000),
                "ab".repeat(700),
                "abc".repeat(600),
                "the quick brown fox ".repeat(120),
                randomAscii(3000, 1337L),
                randomAscii(4096, 424242L));
    }

    private static String randomAscii(int length, long seed) {
        Random random = new Random(seed);
        char[] chars = new char[length];
        String alphabet =
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-_()[]{}\\n"
                        + "\\t";
        for (int i = 0; i < length; i++) {
            chars[i] = alphabet.charAt(random.nextInt(alphabet.length()));
        }
        return new String(chars);
    }
}
