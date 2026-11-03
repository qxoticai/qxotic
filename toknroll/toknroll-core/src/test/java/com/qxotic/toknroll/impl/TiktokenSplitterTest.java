package com.qxotic.toknroll.impl;

import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.spi.CharClasses;
import com.qxotic.toknroll.testkit.SplitterParityHarness;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class TiktokenSplitterTest {

    /**
     * Every kind of character the patterns tell apart, and the ones that fooled a scanner once;
     * last, DeepSeek's: ASCII and other punctuation and symbols, kana, controls, format chars.
     */
    static final String[] PIECES = {
        "a", "b", "Z", "Q", "hello", "World", " ", "  ", "\t", "\n", "\r", "\r\n", "\n\n", "'",
        "'s", "'S", "'ſ", "'re", "'LL", "'d", "'x", "1", "23", "456", "7890", "٣", "Ⅻ", "²", "é",
        "É", "ß", "ǅ", "ʰ", "中", "文", "ア", "́", "ः", "⃝", "!", ".", ",", "/", "\\", "-", "#", " ",
        " ", "\u0085", " ", " ", "\u001C", "​", "😀", "𝐀", "𝐚", "𐐀", "\uD800", "\uDC00", "ı",
        "İ", "K", "ẞ", "ſ", "ـ", "〇", "ᾈ", "'abc", "$x", "+", "^", "~", "€", "©", "«", "・", "゛",
        "゠", "あ", "カナ", "ー", "\u0001", "\u00ad", "龥", "\u9fa6"
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
        String[] regexes = stages.split("\0");
        int last = regexes.length - 1;
        Splitter[] each = new Splitter[regexes.length];
        for (int k = 0; k < each.length; k++) {
            each[k] = Splitter.regex(unicode(regexes[k]));
        }
        assertTrue(Splitter.sequence(each) instanceof TiktokenSplitter, stages);
        // not these stages: the last first, one more, a stage before the last that drops its
        // gaps, a stage without Unicode classes or with a flag that changes what it matches
        Splitter[] rotated = each.clone();
        rotated[0] = each[last];
        rotated[last] = each[0];
        assertFalse(Splitter.sequence(rotated) instanceof TiktokenSplitter);
        Splitter[] more = Arrays.copyOf(each, each.length + 1);
        more[each.length] = each[0];
        assertFalse(Splitter.sequence(more) instanceof TiktokenSplitter);
        int caseless = Pattern.UNICODE_CHARACTER_CLASS | Pattern.CASE_INSENSITIVE;
        for (int k = 0; k < each.length; k++) {
            Splitter[] changed = each.clone();
            if (k < last) {
                changed[k] = RegexSplitter.create(unicode(regexes[k]), false);
                assertFalse(Splitter.sequence(changed) instanceof TiktokenSplitter);
            }
            changed[k] = Splitter.regex(Pattern.compile(regexes[k]));
            assertFalse(Splitter.sequence(changed) instanceof TiktokenSplitter);
            changed[k] = Splitter.regex(Pattern.compile(regexes[k], caseless));
            assertFalse(Splitter.sequence(changed) instanceof TiktokenSplitter);
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("knownStages")
    void scannerSplitsLikeItsStages(String stages) {
        String[] regexes = stages.split("\0");
        Splitter[] scanned = new Splitter[regexes.length];
        Splitter[] regexed = new Splitter[regexes.length];
        for (int k = 0; k < regexes.length; k++) {
            scanned[k] = Splitter.regex(unicode(regexes[k]));
            // the regex itself, as a splitter no scanner recognizes: chained stage by stage
            regexed[k] = RegexSplitter.create(unicode(regexes[k]))::splitAll;
        }
        Splitter truth = surrogatesAsReplacement(Splitter.sequence(regexed));
        assertSplitsAlike(stages, Splitter.sequence(scanned), truth);
    }

    /**
     * The regexes reading a lone surrogate as the scanner does, as U+FFFD, which DeepSeek's rule
     * tells apart (a symbol, a surrogate no class); the spans stay on the text.
     */
    private static Splitter surrogatesAsReplacement(Splitter truth) {
        return (text, start, end, consumer) -> {
            // lone surrogate: U+FFFD
            String read = new String(TiktokenSplitter.utf8(text, 0, text.length()), UTF_8);
            truth.splitAll(read, start, end, (t, s, e) -> consumer.accept(text, s, e));
        };
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
        Pattern punctOrSymbol = Pattern.compile("[\\p{P}\\p{S}]", Pattern.UNICODE_CHARACTER_CLASS);
        for (int cp = 0; cp <= Character.MAX_CODE_POINT; cp++) {
            String s = new String(Character.toChars(cp));
            int expected = 0;
            for (int i = 0; i < patterns.length; i++) {
                if (patterns[i].matcher(s).matches()) {
                    expected |= 1 << i;
                }
            }
            assertEquals(expected, CharClasses.of(cp), "U+" + Integer.toHexString(cp));
            assertEquals(
                    punctOrSymbol.matcher(s).matches(),
                    TiktokenSplitter.punctOrSymbol(cp),
                    "U+" + Integer.toHexString(cp));
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
