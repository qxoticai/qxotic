package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.FastQwen35Splitter;
import com.qxotic.toknroll.testkit.TestCorpora;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class FastQwen35SplitterTest {

    private static final String QWEN35_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    @Test
    void matchesRegexSplitterForRepresentativeInputs() {
        Splitter fast = FastQwen35Splitter.INSTANCE;
        Splitter regex =
                Splitter.regex(Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));

        for (String sample : TestCorpora.QWEN35_SPLITTER_REPRESENTATIVE_SAMPLES) {
            assertEquals(tokens(regex, sample), tokens(fast, sample), sample);
        }
    }

    @Test
    void matchesRegexSplitterForRandomMixedInputs() {
        Splitter fast = FastQwen35Splitter.INSTANCE;
        Splitter regex =
                Splitter.regex(Pattern.compile(QWEN35_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        Random rnd = new Random(20260403L);

        for (int n = 0; n < 2_000; n++) {
            int len = 1 + rnd.nextInt(128);
            StringBuilder sb = new StringBuilder(len);
            for (int i = 0; i < len; i++) {
                int bucket = rnd.nextInt(8);
                char c;
                if (bucket <= 3) {
                    c = (char) (32 + rnd.nextInt(95));
                } else if (bucket == 4) {
                    c = (char) (0x00C0 + rnd.nextInt(0x017F - 0x00C0));
                } else if (bucket == 5) {
                    c = (char) (0x0400 + rnd.nextInt(0x04FF - 0x0400));
                } else if (bucket == 6) {
                    c = (char) (0x0600 + rnd.nextInt(0x06FF - 0x0600));
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
