package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.impl.FastTekkenSplitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class FastTekkenSplitterTest {

    private static final String TEKKEN_PATTERN =
            "[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r"
                + "\\n"
                + "\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}|"
                + " ?[^\\s\\p{L}\\p{N}]+[\\r"
                + "\\n"
                + "/]*|\\s*[\\r"
                + "\\n"
                + "]+|\\s+(?!\\S)|\\s+";

    @Test
    void matchesRegexSplitterForRepresentativeInputs() {
        Splitter fast = FastTekkenSplitter.INSTANCE;
        Splitter regex = Splitter.regex(TEKKEN_PATTERN);

        List<String> samples =
                List.of(
                        "Hello world",
                        "HELLO WORLD",
                        "camelCaseHTTPParser",
                        "kebab-case / slash\\",
                        "symbols <>[]{}()/*+-=_",
                        "123456789",
                        "line1\r\nline2\nline3",
                        "  leading and trailing  ",
                        "tabs\tand\fvertical\u000B",
                        "emoji 😀 fallback",
                        "Accents: cafe manana resume");

        for (String sample : samples) {
            assertEquals(tokens(regex, sample), tokens(fast, sample), sample);
        }
    }

    @Test
    void matchesRegexSplitterForRandomAsciiInputs() {
        Splitter fast = FastTekkenSplitter.INSTANCE;
        Splitter regex = Splitter.regex(TEKKEN_PATTERN);
        Random rnd = new Random(20260405L);

        for (int n = 0; n < 2_000; n++) {
            int len = 1 + rnd.nextInt(128);
            StringBuilder sb = new StringBuilder(len);
            for (int i = 0; i < len; i++) {
                char c = (char) (9 + rnd.nextInt(118));
                sb.append(c);
            }
            String sample = sb.toString();
            assertEquals(tokens(regex, sample), tokens(fast, sample), sample);
        }
    }

    private static List<String> tokens(Splitter splitter, String text) {
        List<String> out = new ArrayList<>();
        splitter.splitAll(
                text, (source, start, end) -> out.add(source.subSequence(start, end).toString()));
        return out;
    }
}
