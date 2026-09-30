package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.spi.Kernels;
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
        "12345678"
    };

    static Stream<Object[]> rules() {
        return Stream.of(
                new Object[] {
                    Kernels.CL100K, "(?i:" + CONTRACTION + ")|" + String.format(REST, "\\p{N}{1,3}")
                },
                new Object[] {
                    Kernels.CL100K_ASCII, ASCII_BOTH + "|" + String.format(REST, "\\p{N}{1,3}")
                },
                new Object[] {
                    Kernels.QWEN2, "(?i:" + CONTRACTION + ")|" + String.format(REST, "\\p{N}")
                },
                new Object[] {
                    Kernels.QWEN2_ASCII, ASCII_BOTH + "|" + String.format(REST, "\\p{N}")
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
            check(Kernels.CL100K, pattern, bytes, bytes.length - 16, 1024);
        }
    }

    private static void check(int rules, Pattern pattern, String text, int batch) {
        byte[] utf8 = Kernels.scalar().utf8(text, 0, text.length(), 16);
        check(rules, pattern, utf8, utf8.length - 16, batch);
    }

    /** The kernel's piece ends against the regex over the text as the scanner decodes it. */
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
            int n = split.split(utf8, cursor, end, ends);
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
