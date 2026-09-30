package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.testkit.SplitterParityHarness;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class TiktokenSplitterTest {

    /** Every kind of character the patterns tell apart, and the ones that fooled a scanner once. */
    private static final String[] PIECES = {
        "a", "b", "Z", "Q", "hello", "World", " ", "  ", "\t", "\n", "\r", "\r\n", "\n\n", "'",
        "'s", "'S", "'ſ", "'re", "'LL", "'d", "'x", "1", "23", "456", "7890", "٣", "Ⅻ", "²", "é",
        "É", "ß", "ǅ", "ʰ", "中", "文", "ア", "́", "ः", "⃝", "!", ".", ",", "/", "\\", "-", "#", " ",
        " ", "\u0085", " ", " ", "\u001C", "​", "😀", "𝐀", "𝐚", "𐐀", "\uD800", "\uDC00", "ı",
        "İ", "K", "ẞ", "ſ", "ـ", "〇", "ᾈ"
    };

    private static final String[] CASES = {
        "",
        " ",
        "\n",
        "a",
        "'s",
        "'ſ",
        "'S'sx",
        "it's",
        "IT'S",
        "it'ſ",
        "they'RE",
        "we'll",
        "hello world",
        "  hello  world  ",
        "hello\n\nworld",
        "\n \n\n x",
        "abc.\n\n/n/ before",
        ".\r\n/\n/x",
        "1234567",
        " 1234",
        "a1b22c333",
        "ABC",
        "ABCdef",
        "abcDEF",
        "中文",
        "中文abc",
        "́abc",
        "́ABC",
        "́",
        "ABĆ",
        "ǅǅa",
        "𝐀𝐚",
        "x\uD800y",
        "\uDC00",
        "  a",
        "  ",
        "!abc",
        "\tabc",
        "\n\tabc",
        "a'",
        "a''",
        "tab\there",
        "\u0085x",
        " \n",
        "trailing   ",
        "trailing\n\n",
        "\r\r\n",
        " /",
        "//\n//",
        "٣٣٣٣",
        "é́",
    };

    @ParameterizedTest(name = "{0}")
    @MethodSource("knownPatterns")
    void knownPatternsAreRoutedToTheScanner(String regex) {
        Pattern pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        assertTrue(Splitter.regex(pattern) instanceof TiktokenSplitter, regex);
        assertFalse(Splitter.regex(Pattern.compile(regex)) instanceof TiktokenSplitter, regex);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("knownPatterns")
    void scannerSplitsLikeItsRegex(String regex) {
        Pattern pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        Splitter scanner = Splitter.regex(pattern);
        Splitter truth = RegexSplitter.create(pattern);
        SplitterParityHarness.runFullBattery(regex, scanner, truth);
        for (String text : CASES) {
            assertEquals(spans(truth, text), spans(scanner, text), () -> regex + " on " + text);
        }
        Random random = new Random(0x7EC70BE5L);
        for (int i = 0; i < 20_000; i++) {
            StringBuilder text = new StringBuilder();
            for (int n = random.nextInt(12); n > 0; n--) {
                text.append(PIECES[random.nextInt(PIECES.length)]);
            }
            String s = text.toString();
            assertEquals(spans(truth, s), spans(scanner, s), () -> regex + " on " + s);
        }
    }

    @Test
    void characterClassesAreTheRegexEngines() {
        String[] regexes = {
            "\\s",
            "\\p{L}",
            "\\p{N}",
            "\\p{M}",
            "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]",
            "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]",
            "[\\r\\n]"
        };
        Pattern[] patterns = new Pattern[regexes.length];
        for (int i = 0; i < regexes.length; i++) {
            patterns[i] = Pattern.compile(regexes[i], Pattern.UNICODE_CHARACTER_CLASS);
        }
        for (int cp = 0; cp <= Character.MAX_CODE_POINT; cp++) {
            String s = new String(Character.toChars(cp));
            int expected = 0;
            for (int i = 0; i < patterns.length; i++) {
                if (patterns[i].matcher(s).matches()) {
                    expected |= 1 << i;
                }
            }
            assertEquals(
                    expected, TiktokenSplitter.CharClasses.of(cp), "U+" + Integer.toHexString(cp));
        }
    }

    static Stream<String> knownPatterns() {
        return StreamSupport.stream(TiktokenSplitter.knownPatterns().spliterator(), false).sorted();
    }

    static List<String> spans(Splitter splitter, String text) {
        List<String> out = new ArrayList<>();
        splitter.splitAll(text, (t, s, e) -> out.add(t.subSequence(s, e).toString()));
        return out;
    }
}
