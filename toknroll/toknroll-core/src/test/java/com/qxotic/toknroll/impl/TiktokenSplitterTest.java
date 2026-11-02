package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.spi.CharClasses;
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
    static final String[] PIECES = {
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
        // where a first stage cuts: around numbers, before a newline run
        "a  1",
        "a  \n1",
        "1 2  3   4",
        "x1y22z333w4444",
        "12345678",
        " 1",
        "  1  ",
        "a.\n\nb",
        "a  \n\nb",
        "a\r\n\r\nb",
        "a\rb\r\nc",
        "\r\n",
        "\n\r\n\n x",
        "end.\r",
        "٣ ٣  ⅫⅫ²",
    };

    @ParameterizedTest(name = "{0}")
    @MethodSource("knownPatterns")
    void knownPatternsAreRoutedToTheScanner(String regex) {
        Pattern pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        assertTrue(Splitter.regex(pattern) instanceof TiktokenSplitter, regex);
        assertFalse(Splitter.regex(Pattern.compile(regex)) instanceof TiktokenSplitter, regex);
        int unicode = Pattern.UNICODE_CHARACTER_CLASS;
        assertTrue(
                Splitter.regex(Pattern.compile(regex, unicode | Pattern.UNICODE_CASE))
                        instanceof TiktokenSplitter,
                regex);
        // with another flag the regex is not the one the scanner stands for
        for (int flag : new int[] {Pattern.CASE_INSENSITIVE, Pattern.MULTILINE, Pattern.DOTALL}) {
            assertFalse(
                    Splitter.regex(Pattern.compile(regex, unicode | flag))
                            instanceof TiktokenSplitter,
                    regex);
        }
    }

    /** gpt2's contraction is lowercase: under CASE_INSENSITIVE "'S" is one too. */
    @Test
    void aCaseInsensitiveKnownRegexSplitsAsTheRegex() {
        String gpt2 =
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+|"
                        + " ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
        Pattern pattern =
                Pattern.compile(gpt2, Pattern.UNICODE_CHARACTER_CLASS | Pattern.CASE_INSENSITIVE);
        assertEquals(List.of("IT", "'S"), spans(Splitter.regex(pattern), "IT'S"));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("knownPatterns")
    void scannerSplitsLikeItsRegex(String regex) {
        Pattern pattern = unicode(regex);
        assertSplitsAlike(regex, Splitter.regex(pattern), RegexSplitter.create(pattern));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("knownStages")
    void knownStagesAreRoutedToOneScanner(String stages) {
        Pattern first = unicode(stages.split("\n")[0]);
        Pattern second = unicode(stages.split("\n")[1]);
        Splitter one = Splitter.regex(first);
        Splitter two = Splitter.regex(second);
        assertTrue(Splitter.sequence(one, two) instanceof TiktokenSplitter, stages);
        // not these stages: swapped, a third, a first stage that drops its gaps, no Unicode classes
        assertFalse(Splitter.sequence(two, one) instanceof TiktokenSplitter);
        assertFalse(Splitter.sequence(one, two, one) instanceof TiktokenSplitter);
        assertFalse(
                Splitter.sequence(RegexSplitter.create(first, false), two)
                        instanceof TiktokenSplitter);
        assertFalse(
                Splitter.sequence(Splitter.regex(Pattern.compile(first.pattern())), two)
                        instanceof TiktokenSplitter);
        assertFalse(
                Splitter.sequence(one, Splitter.regex(Pattern.compile(second.pattern())))
                        instanceof TiktokenSplitter);
        int caseless = Pattern.UNICODE_CHARACTER_CLASS | Pattern.CASE_INSENSITIVE;
        assertFalse(
                Splitter.sequence(one, Splitter.regex(Pattern.compile(second.pattern(), caseless)))
                        instanceof TiktokenSplitter);
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("knownStages")
    void scannerSplitsLikeItsStages(String stages) {
        RegexSplitter first = RegexSplitter.create(unicode(stages.split("\n")[0]));
        RegexSplitter second = RegexSplitter.create(unicode(stages.split("\n")[1]));
        Splitter scanner =
                Splitter.sequence(
                        Splitter.regex(first.pattern()), Splitter.regex(second.pattern()));
        // the stages as regexes: each piece of the first split by the second
        Splitter truth =
                (text, start, end, consumer) ->
                        first.splitAll(
                                text, start, end, (t, s, e) -> second.splitAll(t, s, e, consumer));
        assertSplitsAlike(stages, scanner, truth);
    }

    /**
     * Laguna's first stage, unrolled and possessive so that a long line cannot overflow the stack,
     * splits as the regex it replaced, which recursed once per char.
     */
    @Test
    void lagunasFirstStageSplitsAsItsRecursiveOriginal() {
        String original = "(?:\\r?\\n)+(?:[^\\r\\n]|\\r(?!\\n))*|(?:[^\\r\\n]|\\r(?!\\n))+";
        RegexSplitter truth = RegexSplitter.create(unicode(original));
        RegexSplitter fixed = RegexSplitter.create(unicode(TiktokenSplitter.LAGUNA_NEWLINES));
        String[] parts = {
            "a", " ", "\n", "\r", "\r\n", "\n\n", "\r\r", "x\r", "\u00e9", "\uD83D\uDE00"
        };
        Random random = new Random(1);
        for (int i = 0; i < 100_000; i++) {
            StringBuilder text = new StringBuilder();
            for (int k = random.nextInt(14); k > 0; k--)
                text.append(parts[random.nextInt(parts.length)]);
            String t = text.toString();
            assertEquals(spans(truth, t), spans(fixed, t), t);
        }
        for (String huge :
                new String[] {
                    "x".repeat(2_000_000), "\r\n".repeat(200_000), "a\rb".repeat(100_000)
                }) {
            assertEquals(1, spans(fixed, huge).size());
        }
    }

    /** Cases, the parity battery, short random texts, and long ones past a batch of pieces. */
    private static void assertSplitsAlike(String name, Splitter scanner, Splitter truth) {
        SplitterParityHarness.runFullBattery(name, scanner, truth);
        for (String text : CASES) {
            assertEquals(spans(truth, text), spans(scanner, text), () -> name + " on " + text);
        }
        Random random = new Random(0x7EC70BE5L);
        for (int i = 0; i < 20_000; i++) {
            String s = random(random, random.nextInt(12));
            assertEquals(spans(truth, s), spans(scanner, s), () -> name + " on " + s);
        }
        for (int i = 0; i < 20; i++) {
            String s = random(random, 3000 + random.nextInt(3000));
            assertEquals(spans(truth, s), spans(scanner, s), () -> name + " on a long text");
        }
        // a first-stage piece longer than a batch of pieces: digits, then text
        for (String s : new String[] {"7".repeat(5000), "word ".repeat(3000) + "1"}) {
            assertEquals(spans(truth, s), spans(scanner, s), () -> name + " on a long run");
        }
    }

    private static String random(Random random, int pieces) {
        StringBuilder text = new StringBuilder();
        for (int n = pieces; n > 0; n--) {
            text.append(PIECES[random.nextInt(PIECES.length)]);
        }
        return text.toString();
    }

    private static Pattern unicode(String regex) {
        return Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
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
            assertEquals(expected, CharClasses.of(cp), "U+" + Integer.toHexString(cp));
        }
    }

    static Stream<String> knownStages() {
        return StreamSupport.stream(TiktokenSplitter.knownStages().spliterator(), false).sorted();
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
