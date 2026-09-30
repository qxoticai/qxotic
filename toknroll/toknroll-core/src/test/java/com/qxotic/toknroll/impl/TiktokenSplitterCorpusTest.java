package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import com.qxotic.toknroll.testkit.corpus.Enwik8Corpus;
import java.nio.charset.StandardCharsets;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/** Every known pattern's scanner against its regex over all of enwik8, span for span. */
@Tag("corpus")
class TiktokenSplitterCorpusTest {

    private static final String TEXT =
            new String(Enwik8Corpus.load().data(), StandardCharsets.UTF_8);

    @ParameterizedTest(name = "{0}")
    @MethodSource("knownPatterns")
    void scannerSplitsEnwik8LikeItsRegex(String regex) {
        Pattern pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        long[] expected = spans(RegexSplitter.create(pattern));
        long[] actual = spans(Splitter.regex(pattern));
        assertEquals(expected[0], actual[0], "span count for " + regex);
        assertEquals(expected[1], actual[1], "span checksum for " + regex);
    }

    @Test
    void theTiktokenFixturesTakeTheScanner() {
        for (String encoding : new String[] {"r50k_base", "cl100k_base", "o200k_base"}) {
            Splitter splitter = Splitter.regex(TiktokenFixtures.splitPattern(encoding));
            assertTrue(splitter instanceof TiktokenSplitter, encoding);
        }
    }

    private static long[] spans(Splitter splitter) {
        long[] countAndChecksum = new long[2];
        splitter.splitAll(
                TEXT,
                (text, start, end) -> {
                    countAndChecksum[0]++;
                    countAndChecksum[1] = countAndChecksum[1] * 31 + start * 7L + end;
                });
        return countAndChecksum;
    }

    static Stream<String> knownPatterns() {
        return TiktokenSplitterTest.knownPatterns();
    }
}
