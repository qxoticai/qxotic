package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.FastSplitters;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class FastR50kSplitterTest {

    private static final String R50K_PATTERN =
            "'(?:[sdmt]|ll|ve|re)| ?\\p{L}++| ?\\p{N}++|"
                    + " ?[^\\s\\p{L}\\p{N}]++|\\s++$|\\s+(?!\\S)|\\s";

    @Test
    void matchesRegexSplitterForRepresentativeInputs() {
        Splitter fast = FastSplitters.r50k();
        Splitter regex =
                Splitter.regex(Pattern.compile(R50K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));

        List<String> samples =
                List.of(
                        "Hello world",
                        "Hello, world!!!",
                        "it's we're they'll I'd",
                        "  leading and trailing  ",
                        "tabs\tand\nnewlines\r\n",
                        "numbers 123 456789",
                        "symbols <>[]{}()/*+-=_",
                        "mix: 's 't 're 've 'm 'll 'd",
                        "emoji 😀 fallback",
                        "Accents: café mañana résumé");

        for (String sample : samples) {
            assertEquals(tokens(regex, sample), tokens(fast, sample), sample);
        }
    }

    @Test
    void matchesRegexSplitterForRandomAsciiInputs() {
        Splitter fast = FastSplitters.r50k();
        Splitter regex =
                Splitter.regex(Pattern.compile(R50K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        Random rnd = new Random(12345L);

        for (int n = 0; n < 2_000; n++) {
            int len = 1 + rnd.nextInt(128);
            StringBuilder sb = new StringBuilder(len);
            for (int i = 0; i < len; i++) {
                char c = (char) (32 + rnd.nextInt(95));
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
