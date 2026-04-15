package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.FastCl100kSplitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class FastCl100kSplitterTest {

    private static final String CL100K_PATTERN =
            "'(?i:[sdmt]|ll|ve|re)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+| ?[^\\s\\p{L}\\p{N}]++[\\r"
                    + "\\n"
                    + "]*+|\\s++$|\\s*[\\r"
                    + "\\n"
                    + "]|\\s+(?!\\S)|\\s";

    @Test
    void matchesRegexSplitterForRepresentativeInputs() {
        Splitter fast = FastCl100kSplitter.INSTANCE;
        Splitter regex =
                Splitter.regex(Pattern.compile(CL100K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));

        List<String> samples =
                List.of(
                        "Hello world",
                        "It's we're THEY'LL I'd",
                        "  leading and trailing  ",
                        "tabs\tand\nnewlines\r\n",
                        "numbers 7 42 123 456789",
                        "symbols <>[]{}()/*+-=_",
                        "emoji 😀 fallback",
                        "Accents: café mañana résumé",
                        "CJK 你好 世界",
                        "Arabic مرحبا بالعالم");

        for (String sample : samples) {
            assertEquals(tokens(regex, sample), tokens(fast, sample), sample);
        }
    }

    @Test
    void matchesRegexSplitterForRandomMixedInputs() {
        Splitter fast = FastCl100kSplitter.INSTANCE;
        Splitter regex =
                Splitter.regex(Pattern.compile(CL100K_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        Random rnd = new Random(20260403L);

        for (int n = 0; n < 1_000; n++) {
            int len = 1 + rnd.nextInt(128);
            StringBuilder sb = new StringBuilder(len);
            for (int i = 0; i < len; i++) {
                int bucket = rnd.nextInt(7);
                char c;
                if (bucket <= 3) {
                    c = (char) (32 + rnd.nextInt(95));
                } else if (bucket == 4) {
                    c = (char) (0x00C0 + rnd.nextInt(0x017F - 0x00C0));
                } else if (bucket == 5) {
                    c = (char) (0x0400 + rnd.nextInt(0x04FF - 0x0400));
                } else {
                    c = (char) (0x4E00 + rnd.nextInt(256));
                }
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
