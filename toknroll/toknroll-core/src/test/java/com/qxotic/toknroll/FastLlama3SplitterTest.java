package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.impl.FastLlama3Splitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

class FastLlama3SplitterTest {

    private static final String LLAMA3_PATTERN =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    @Test
    void matchesRegexSplitterForRepresentativeInputs() {
        Splitter fast = FastLlama3Splitter.INSTANCE;
        Splitter regex =
                Splitter.regex(Pattern.compile(LLAMA3_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));

        List<String> samples =
                List.of(
                        "Hello world",
                        "Hello, world!!!",
                        "It's we're THEY'LL I'd",
                        "  leading and trailing  ",
                        "tabs\tand\nnewlines\r\n",
                        "numbers 123 456789",
                        "symbols <>[]{}()/*+-=_",
                        "mix: 'S 'T 'Re 'vE 'M 'Ll 'D",
                        "emoji 😀 fallback",
                        "Accents: café mañana résumé");

        for (String sample : samples) {
            assertEquals(tokens(regex, sample), tokens(fast, sample), sample);
        }
    }

    @Test
    void matchesRegexSplitterForRandomAsciiInputs() {
        Splitter fast = FastLlama3Splitter.INSTANCE;
        Splitter regex =
                Splitter.regex(Pattern.compile(LLAMA3_PATTERN, Pattern.UNICODE_CHARACTER_CLASS));
        Random rnd = new Random(424242L);

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
