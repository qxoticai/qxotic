package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.spi.TiktokenEngine;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/** The bitstream splitter against the regex it implements. */
class BitSplitTest {

    private static final String CONTRACTION = "'s|'t|'re|'ve|'m|'ll|'d";
    private static final String ASCII_BOTH =
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])";
    private static final String REST =
            "[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?\\p{L}+|%s| ?[^\\s\\p{L}\\p{N}]+[\\r"
                    + "\\n"
                    + "]*|\\s*[\\r"
                    + "\\n"
                    + "]+|\\s+(?!\\S)|\\s+";

    /** Every kind of character the patterns tell apart, and the ones that fooled a scanner once. */
    static final String[] PIECES = {
        "a",
        "b",
        "Z",
        "Q",
        "hello",
        "World",
        " ",
        "  ",
        "\t",
        "\n",
        "\r",
        "\r\n",
        "\n\n",
        "'",
        "'s",
        "'S",
        "'ſ",
        "'re",
        "'LL",
        "'d",
        "'x",
        "1",
        "23",
        "456",
        "7890",
        "٣",
        "Ⅻ",
        "²",
        "é",
        "É",
        "ß",
        "ǅ",
        "ʰ",
        "中",
        "文",
        "ア",
        // words in the scripts the merge seeds, and beside the bytes that forbid a seed
        "的",
        "一",
        "日本語の文章",
        "中华人民共和国",
        "한국어",
        "привет",
        "мир",
        "Ωμέγα",
        "العربية",
        "עברית",
        "नमस्ते",
        "ภาษาไทย",
        "ქართული",
        "𠀀",
        "́",
        "ः",
        "⃝",
        "!",
        ".",
        ",",
        "/",
        "\\",
        "-",
        "#",
        " ",
        " ",
        "\u0085",
        " ",
        " ",
        "\u001C",
        "​",
        "😀",
        "𝐀",
        "𝐚",
        "𐐀",
        "\uD800",
        "\uDC00",
        "ı",
        "İ",
        "K",
        "ẞ",
        "ſ",
        "ـ",
        "〇",
        "ᾈ",
        "\n\n\n",
        "    ",
        "\n   \n",
        "   \n   ",
        "12345678",
        // what DeepSeek's stages tell apart: ASCII punctuation before letters, symbols, kana
        "'abc",
        "$x",
        "€",
        "・",
        "゛",
        "あ",
        "\u0001",
        "\u00ad"
    };

    private static final String UPPER = "[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]";
    private static final String LOWER = "[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]";
    private static final String PREFIX = "[^\\r\\n\\p{L}\\p{N}]?";
    private static final String CASED =
            PREFIX + UPPER + "*" + LOWER + "+%s|" + PREFIX + UPPER + "+" + LOWER + "*%s";
    private static final String O200K_REST =
            "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";
    private static final String R50K =
            CONTRACTION + "| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

    /**
     * R50K_NUMBERS's stages, {@code \p{N}} then r50k, as one regex: a number alone, and a blank run
     * before one whole, the number ending the text before it.
     */
    private static final String R50K_NUMBERS =
            CONTRACTION + "| ?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+|\\s+(?=\\p{N})|\\s+(?!\\S)|\\s+";

    static Stream<Object[]> rules() {
        String suffix = "(?i:" + CONTRACTION + ")?";
        return Stream.of(
                new Object[] {TiktokenEngine.R50K, R50K},
                new Object[] {TiktokenEngine.R50K_NUMBERS, R50K_NUMBERS},
                new Object[] {
                    TiktokenEngine.O200K,
                    String.format(CASED, suffix, suffix) + "|\\p{N}{1,3}" + O200K_REST
                },
                new Object[] {
                    TiktokenEngine.TEKKEN, String.format(CASED, "", "") + "|\\p{N}" + O200K_REST
                },
                new Object[] {
                    TiktokenEngine.CL100K,
                    "(?i:" + CONTRACTION + ")|" + String.format(REST, "\\p{N}{1,3}")
                },
                new Object[] {
                    TiktokenEngine.CL100K_ASCII,
                    ASCII_BOTH + "|" + String.format(REST, "\\p{N}{1,3}")
                },
                new Object[] {
                    TiktokenEngine.QWEN2,
                    "(?i:" + CONTRACTION + ")|" + String.format(REST, "\\p{N}")
                },
                new Object[] {
                    TiktokenEngine.QWEN2_ASCII, ASCII_BOTH + "|" + String.format(REST, "\\p{N}")
                });
    }

    @ParameterizedTest(name = "{1}")
    @MethodSource("rules")
    void splitsLikeTheRegex(int rules, String regex) {
        Pattern pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        Random random = new Random(0xB17CA110L);
        for (int i = 0; i < 40_000; i++) {
            StringBuilder text = new StringBuilder();
            int pieces = i % 10 == 0 ? random.nextInt(200) : random.nextInt(12);
            for (int n = pieces; n > 0; n--) {
                text.append(PIECES[random.nextInt(PIECES.length)]);
            }
            check(rules, pattern, text.toString(), i % 3 == 0 ? 1 + random.nextInt(5) : 1024);
        }
    }

    @ParameterizedTest(name = "{1}")
    @MethodSource("rules")
    void longRunsAcrossBlocks(int rules, String regex) {
        Pattern pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        for (int n : new int[] {1, 62, 63, 64, 65, 127, 128, 129, 200, 4095, 4096, 4097, 5000}) {
            check(rules, pattern, "a" + "\n".repeat(n) + "b", 1024);
            check(rules, pattern, "a" + " ".repeat(n) + "b", 1024);
            check(rules, pattern, "a\n" + " ".repeat(n) + "\nb", 1024);
            check(rules, pattern, "a\n" + " ".repeat(n) + "\n" + " ".repeat(n) + "b", 1024);
            check(rules, pattern, " ".repeat(n) + "\n" + " ".repeat(n), 1024);
            check(rules, pattern, "1".repeat(n) + " " + "٣".repeat(n), 1024);
            check(rules, pattern, "x".repeat(n) + "'s" + "'ll".repeat(3), 1024);
            check(rules, pattern, "!".repeat(n) + "\n\n" + "!".repeat(n) + "abc", 1024);
            check(rules, pattern, "中".repeat(n) + "😀".repeat(n) + "é".repeat(n), 1024);
            check(rules, pattern, "a" + " ".repeat(n) + "b", 2);
            check(rules, pattern, "A".repeat(n) + "a" + "A".repeat(n) + " " + "A".repeat(n), 1024);
            check(rules, pattern, "A".repeat(n) + "ʰ" + "A".repeat(n) + "bA" + "A".repeat(n), 1024);
        }
    }

    /** The hand look-aheads past the next block start on a character, not inside one. */
    @ParameterizedTest(name = "{1}")
    @MethodSource("rules")
    void handLookAheadsStartOnACharacter(int rules, String regex) {
        Pattern pattern = Pattern.compile(regex, Pattern.UNICODE_CHARACTER_CLASS);
        // a two-byte letter across the edge of the block after next
        check(rules, pattern, "ʰ" + "A".repeat(125) + "Ä" + "A".repeat(10) + "a", 1024);
        check(rules, pattern, "ʰ" + "A".repeat(125) + "Ä" + "A".repeat(10) + "ʰ", 1024);
        // a multi-byte blank across the edge of the block where the ring overflows
        for (String blank : new String[] {"\u00a0", "\u2028"}) {
            for (int at = 4090; at < 4098; at++) {
                String run = " ".repeat(at - 2) + blank + " ".repeat(200);
                check(rules, pattern, "a\n" + run + "\nb", 1024);
                check(rules, pattern, "a\n" + run + "b", 1024);
            }
        }
    }

    /**
     * A long cased run is looked at by hand once, not once per block: at a million capitals the
     * per-block look took seconds.
     */
    @Test
    void longCasedRunsSplitInLinearTime() {
        String capitals = "A".repeat(1 << 20);
        String[] texts = {capitals, capitals + "a", capitals + "ʰ", "ʰ" + capitals + "a"};
        for (int rules : new int[] {TiktokenEngine.O200K, TiktokenEngine.TEKKEN}) {
            BitSplit split = BitSplit.forRules(rules);
            int[] ends = new int[1024];
            for (String text : texts) {
                byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
                long t0 = System.nanoTime();
                int n = split.split(utf8, 0, utf8.length, ends, new BitSplit.Registers());
                long millis = (System.nanoTime() - t0) / 1_000_000;
                assertTrue(n > 0 && ends[n - 1] == utf8.length);
                assertTrue(millis < 2_000, () -> millis + " ms for " + text.length() + " chars");
            }
        }
    }

    @Test
    void illFormedBytesSplitAsTheScannerDecodesThem() {
        Pattern pattern =
                Pattern.compile(
                        "(?i:" + CONTRACTION + ")|" + String.format(REST, "\\p{N}{1,3}"),
                        Pattern.UNICODE_CHARACTER_CLASS);
        Random random = new Random(7);
        for (int i = 0; i < 20_000; i++) {
            byte[] bytes = new byte[random.nextInt(150) + 16];
            for (int k = 0; k < bytes.length - 16; k++) {
                int r = random.nextInt(10);
                bytes[k] =
                        (byte)
                                (r < 5
                                        ? 'a' + random.nextInt(3)
                                        : r < 7
                                                ? random.nextInt(256)
                                                : r < 8 ? ' ' : r < 9 ? '\n' : '1');
            }
            check(TiktokenEngine.CL100K, pattern, bytes, bytes.length - 16, 1024);
        }
    }

    /** The one regex above splits as its two stages chained, the reference it stands in for. */
    @Test
    void r50kNumbersIsItsStages() {
        Pattern one = Pattern.compile(R50K_NUMBERS, Pattern.UNICODE_CHARACTER_CLASS);
        Pattern digits = Pattern.compile("\\p{N}", Pattern.UNICODE_CHARACTER_CLASS);
        Pattern r50k = Pattern.compile(R50K, Pattern.UNICODE_CHARACTER_CLASS);
        Random random = new Random(0x50);
        for (int i = 0; i < 50_000; i++) {
            StringBuilder text = new StringBuilder();
            for (int k = random.nextInt(16); k > 0; k--) {
                text.append(PIECES[random.nextInt(PIECES.length)]);
            }
            String t = text.toString();
            List<String> staged = new ArrayList<>();
            for (String piece : pieces(digits, t, true)) staged.addAll(pieces(r50k, piece, false));
            assertEquals(staged, pieces(one, t, false), t);
        }
    }

    /** A regex's matches, or with {@code gaps} the text between them too, in order. */
    private static List<String> pieces(Pattern pattern, String text, boolean gaps) {
        List<String> out = new ArrayList<>();
        Matcher m = pattern.matcher(text);
        int last = 0;
        while (m.find()) {
            if (gaps && m.start() > last) out.add(text.substring(last, m.start()));
            out.add(m.group());
            last = m.end();
        }
        if (gaps && last < text.length()) out.add(text.substring(last));
        return out;
    }

    private static void check(int rules, Pattern pattern, String text, int batch) {
        byte[] utf8 = Utf8Reference.encode(text, 0, text.length(), 16);
        check(rules, pattern, utf8, utf8.length - 16, batch);
    }

    /** The splitter's piece ends against the regex over the text as the scanner decodes it. */
    private static void check(int rules, Pattern pattern, byte[] utf8, int end, int batch) {
        // decode as the scanner does: a lead claims its length, anything else is U+FFFD
        StringBuilder chars = new StringBuilder();
        List<Integer> byteAt = new ArrayList<>(); // byte offset of each char index
        for (int i = 0; i < end; ) {
            int at = at(utf8, i, end);
            byteAt.add(i);
            int cp = at & 0x1FFFFF;
            if (cp > Character.MAX_CODE_POINT) {
                cp = 0xFFFD;
            }
            chars.appendCodePoint(cp);
            if (Character.charCount(cp) == 2) {
                byteAt.add(-1); // the low surrogate is never a boundary
            }
            i += at >>> 24;
        }
        byteAt.add(end);
        List<Integer> expected = new ArrayList<>();
        Matcher m = pattern.matcher(chars);
        while (m.find()) {
            assertTrue(m.end() > m.start());
            expected.add(byteAt.get(m.end()));
        }
        List<Integer> actual = new ArrayList<>();
        int[] ends = new int[batch];
        int cursor = 0;
        BitSplit split = BitSplit.forRules(rules);
        while (cursor < end) {
            int n = split.split(utf8, cursor, end, ends, new BitSplit.Registers());
            assertTrue(n > 0, "no progress at " + cursor + " in " + chars);
            for (int k = 0; k < n; k++) {
                actual.add(ends[k]);
            }
            cursor = ends[n - 1];
        }
        assertEquals(
                expected,
                actual,
                () -> pattern + " on " + chars + " " + Arrays.toString(Arrays.copyOf(utf8, end)));
    }

    private static int at(byte[] text, int i, int end) {
        int b = text[i];
        if (b >= 0) {
            return (1 << 24) | b;
        }
        if ((b & 0xE0) == 0xC0 && i + 1 < end) {
            return (2 << 24) | ((b & 0x1F) << 6) | (text[i + 1] & 0x3F);
        }
        if ((b & 0xF0) == 0xE0 && i + 2 < end) {
            return (3 << 24)
                    | ((b & 0x0F) << 12)
                    | ((text[i + 1] & 0x3F) << 6)
                    | (text[i + 2] & 0x3F);
        }
        if ((b & 0xF8) == 0xF0 && i + 3 < end) {
            return (4 << 24)
                    | ((b & 0x07) << 18)
                    | ((text[i + 1] & 0x3F) << 12)
                    | ((text[i + 2] & 0x3F) << 6)
                    | (text[i + 3] & 0x3F);
        }
        return (1 << 24) | 0xFFFD;
    }
}
