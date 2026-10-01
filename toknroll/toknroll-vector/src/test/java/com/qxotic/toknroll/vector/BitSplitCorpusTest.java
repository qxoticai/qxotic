package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.testkit.corpus.Enwik8Corpus;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/** The bitstream splitter against its regex over all of enwik8, piece for piece. */
@Tag("corpus")
class BitSplitCorpusTest {

    @ParameterizedTest(name = "{1}")
    @MethodSource("rules")
    void splitsEnwik8LikeTheRegex(int rules, String regex) {
        byte[] data = Enwik8Corpus.load().data();
        String text = new String(data, StandardCharsets.UTF_8);
        byte[] utf8 = Utf8Reference.encode(text, 0, text.length(), 16);
        int end = utf8.length - 16;
        // byte offset of every char index
        int[] byteAt = new int[text.length() + 1];
        for (int i = 0, b = 0; i < text.length(); i++) {
            byteAt[i] = b;
            char c = text.charAt(i);
            b += c < 0x80 ? 1 : c < 0x800 ? 2 : Character.isSurrogate(c) ? 2 : 3;
        }
        byteAt[text.length()] = end;
        Matcher m = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS).matcher(text);
        BitSplit split = BitSplit.forRules(rules);
        int[] ends = new int[1024];
        int cursor = 0;
        long pieces = 0;
        while (cursor < end) {
            int n = split.split(utf8, cursor, end, ends);
            for (int k = 0; k < n; k++) {
                assertEquals(true, m.find(), "regex ran out at piece " + pieces);
                assertEquals(byteAt[m.end()], ends[k], "piece " + pieces + " at " + cursor);
                pieces++;
            }
            cursor = ends[n - 1];
        }
        assertEquals(false, m.find(), "the regex has more pieces");
    }

    static Stream<Object[]> rules() {
        return BitSplitTest.rules();
    }
}
