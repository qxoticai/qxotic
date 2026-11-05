package com.qxotic.toknroll;

import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.impl.ImplAccessor;
import com.qxotic.toknroll.testkit.TestTokenizers;
import com.qxotic.toknroll.testkit.TiktokenFixtures;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.FutureTask;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * {@link Specials}. Every behavior is checked through every way in ({@link Input}), so the chars
 * and the UTF-8 paths cannot drift apart. Ids are positions: a tokenizer of {@code ("x", "<|s|>")}
 * has x = 0 and &lt;|s|&gt; = 1.
 */
class SpecialsTest {

    // ---- every way in: chars, UTF-8, and the default that decodes ----

    interface Way {
        void encodeInto(Specials s, Tokenizer t, String text, IntSequence.Builder out);
    }

    /** The ways text reaches a {@link Specials}. UTF-8 is read from a range of a larger array. */
    enum Input {
        ENCODE((s, t, text, out) -> Arrays.stream(s.encode(t, text).toArray()).forEach(out::add)),
        CHARS(Specials::encodeInto),
        UTF8((s, t, text, out) -> utf8InRange(s, t, text, out)),
        /** {@link Specials#encodeUtf8Into}'s default, which decodes. */
        UTF8_DEFAULT((s, t, text, out) -> utf8InRange(delegating(s), t, text, out));

        final Way way;

        Input(Way way) {
            this.way = way;
        }

        int[] encode(Specials s, Tokenizer t, String text) {
            IntSequence.Builder out = IntSequence.newBuilder();
            way.encodeInto(s, t, text, out);
            return out.build().toArray();
        }
    }

    /** Padded with what completes a special if the range is not respected. */
    private static void utf8InRange(Specials s, Tokenizer t, String text, IntSequence.Builder out) {
        byte[] padded = ("<|" + text + "|>").getBytes(UTF_8);
        s.encodeUtf8Into(t, padded, 2, padded.length - 2, out);
    }

    /** What {@link Specials#of} finds, as the ways in see it. */
    @Nested
    class Of {

        @Test
        void aTokenizerWithoutSpecialTokensHasNone() {
            assertSame(Specials.none(), Specials.of(Mini.of("x")));
            assertTrue(Specials.none().isEmpty());
            assertTrue(Specials.none().tokens().isEmpty());
        }

        @Test
        void specialsMayShareAPrefixAndBeAPrefixOfAnother() {
            Set<String> names = Set.of("<|im_start|>", "<|im_end|>", "<|i|>", "<|im_end|>x");
            Specials specials = specials(Mini.of(names.toArray(new String[0])), names);

            assertEquals(names, specials.tokens().keySet());
            assertFalse(specials.isEmpty());
            assertThrows(UnsupportedOperationException.class, () -> specials.tokens().put("x", 0));
        }

        @Test
        void aSpecialTokenTensOfThousandsLongBuildsOnASmallStack() throws Exception {
            // a token is as deep in the trie as it is long: frozen by recursion, it overflowed
            String token = "a".repeat(50_000);
            FutureTask<Specials> build =
                    new FutureTask<>(() -> specials(Mini.of(token), Set.of(token)));
            new Thread(null, build, "small stack", 256 * 1024).start();
            assertEquals(Set.of(token), build.get().tokens().keySet()); // get rethrows its error
        }

        @Test
        void idsComeFromAVocabularyThatHoldsItsSpecialsApart() {
            Vocabulary vocabulary =
                    Toknroll.vocabulary(Map.of("<|s1|>", 100, "<|s2|>", 200), "a", "b");
            Mini tokenizer = new Mini(vocabulary, false);
            Specials specials = Specials.of(tokenizer);

            assertEquals(Map.of("<|s1|>", 100, "<|s2|>", 200), specials.tokens());
            for (Input input : Input.values()) {
                assertArrayEquals(
                        new int[] {0, 100, 1, 200},
                        input.encode(specials, tokenizer, "a<|s1|>b<|s2|>"),
                        input::name);
            }
        }
    }

    @Nested
    class Encoding {

        @ParameterizedTest
        @EnumSource(Input.class)
        void specialsBecomeTheirIdsAndTheTextBetweenIsEncoded(Input input) {
            Tokenizer t = Mini.of("<|a|>", "<|b|>", "x", "<", "|", ">", "a", "mid");
            Specials specials = specials(t, Set.of("<|a|>", "<|b|>"));
            Map<String, int[]> cases =
                    Map.of(
                            "", new int[0],
                            "x", new int[] {2},
                            "<|a|>", new int[] {0},
                            "x<|a|>x", new int[] {2, 0, 2},
                            "<|a|>mid<|b|>", new int[] {0, 7, 1},
                            "<|a|><|b|><|a|>x", new int[] {0, 1, 0, 2},
                            "<|a|", new int[] {3, 4, 6, 4}, // cut short: text
                            "|a|>", new int[] {4, 6, 4, 5});
            cases.forEach(
                    (text, ids) -> assertArrayEquals(ids, input.encode(specials, t, text), text));
        }

        @ParameterizedTest
        @EnumSource(Input.class)
        void noneEncodesAsTheTokenizerDoes(Input input) {
            Tokenizer t = Mini.of("a", "b", "<|s|>", "a<|s|>b");

            assertArrayEquals(new int[] {3}, input.encode(Specials.none(), t, "a<|s|>b"));
        }

        @ParameterizedTest
        @EnumSource(Input.class)
        void specialsAreMatchedBeforeTheTokenizerPreprocesses(Input input) {
            Tokenizer t = Mini.lowercasing("a", "b", "a<|up|>b", "<|UP|>", "a<|other|>b");
            Specials specials = specials(t, Set.of("<|UP|>"));

            assertArrayEquals(new int[] {2}, t.encode("A<|UP|>B").toArray());
            assertArrayEquals(new int[] {0, 3, 1}, input.encode(specials, t, "A<|UP|>B"));
            assertArrayEquals(new int[] {4}, input.encode(specials, t, "A<|other|>B"));
        }

        @ParameterizedTest
        @EnumSource(Input.class)
        void regexMetacharactersAreLiteral(Input input) {
            String meta = "(.*)?+[]{}|^$\\";
            Tokenizer t = Mini.of("x", "y", meta);
            Specials specials = specials(t, Set.of(meta));

            assertArrayEquals(new int[] {0, 2, 1}, input.encode(specials, t, "x" + meta + "y"));
        }

        @ParameterizedTest
        @EnumSource(Input.class)
        void multiByteSpecialsAmongMultiByteText(Input input) {
            String deepseek = "<｜end▁of▁sentence｜>"; // full-width bars
            String emoji = "😀";
            Tokenizer t = Mini.of("你", "好", deepseek, emoji);
            Specials specials = specials(t, Set.of(deepseek, emoji));
            String text = "你" + deepseek + "好" + emoji + "你";

            assertArrayEquals(new int[] {0, 2, 1, 3, 0}, input.encode(specials, t, text));
        }

        @ParameterizedTest
        @EnumSource(Input.class)
        void aOneCharSpecialSplitsTheTextAroundIt(Input input) {
            Tokenizer t = Mini.of("a", "<", "b", "a<"); // a missed "<" would be the word "a<"
            Specials specials = specials(t, Set.of("<"));

            assertArrayEquals(new int[] {0, 1}, input.encode(specials, t, "a<"));
            assertArrayEquals(new int[] {0, 1, 2, 1}, input.encode(specials, t, "a<b<"));
        }

        @ParameterizedTest
        @EnumSource(Input.class)
        void idsAreAppendedAndDecodeBackToTheText(Input input) {
            Tokenizer t = Mini.of("hello ", " world", "<|special|>");
            Specials specials = specials(t, Set.of("<|special|>"));
            IntSequence.Builder out = IntSequence.newBuilder();
            out.add(1);

            input.way.encodeInto(specials, t, "hello <|special|> world", out);

            assertArrayEquals(new int[] {1, 0, 2, 1}, out.build().toArray());
            assertEquals(" worldhello <|special|> world", t.decode(out.build()));
        }

        @Test
        void encodeAndEncodeIntoRejectNulls() {
            Tokenizer t = Mini.of("x", "<|s|>");
            IntSequence.Builder out = IntSequence.newBuilder();
            Specials compiled = specials(t, Set.of("<|s|>"));
            for (Specials s : List.of(compiled, Specials.none())) {
                assertThrows(NullPointerException.class, () -> s.encode(null, "x"));
                assertThrows(NullPointerException.class, () -> s.encode(t, null));
                assertThrows(NullPointerException.class, () -> s.encodeInto(null, "x", out));
                assertThrows(NullPointerException.class, () -> s.encodeInto(t, null, out));
                assertThrows(NullPointerException.class, () -> s.encodeInto(t, "x", null));
            }
        }
    }

    /** What {@link Specials#encodeUtf8Into} adds: a range, and bytes that are never decoded. */
    @Nested
    class EncodeUtf8Into {

        private final Tokenizer t = Mini.of("a", "b", "<", "|", "s", ">", "<|s|>");
        private final Specials compiled = specials(t, Set.of("<|s|>"));
        private final byte[] utf8 = "a<|s|>b".getBytes(UTF_8);
        private final List<Specials> all = List.of(compiled, Specials.none(), delegating(compiled));

        @Test
        void rejectsNullsAndARangeOutsideTheArray() {
            IntSequence.Builder out = IntSequence.newBuilder();
            int n = utf8.length;
            for (Specials s : all) {
                assertThrows(
                        NullPointerException.class, () -> s.encodeUtf8Into(null, utf8, 0, n, out));
                assertThrows(
                        NullPointerException.class, () -> s.encodeUtf8Into(t, null, 0, 0, out));
                assertThrows(
                        NullPointerException.class, () -> s.encodeUtf8Into(t, utf8, 0, n, null));
                for (int[] r : new int[][] {{-1, 1}, {2, 1}, {0, n + 1}, {n + 1, n + 1}}) {
                    assertThrows(
                            IndexOutOfBoundsException.class,
                            () -> s.encodeUtf8Into(t, utf8, r[0], r[1], out),
                            () -> Arrays.toString(r));
                }
            }
            assertEquals(0, out.size(), "a refused call writes nothing");
        }

        @Test
        void onlyTheRangeIsRead() {
            for (Specials s : all) {
                for (int at = 0; at <= utf8.length; at++) {
                    assertArrayEquals(new int[0], encode(s, t, utf8, at, at));
                }
            }
            for (Specials s : List.of(compiled, delegating(compiled))) {
                // a range that cuts the special holds neither half of it as the special
                assertArrayEquals(new int[] {0, 2, 3}, encode(s, t, utf8, 0, 3));
                assertArrayEquals(new int[] {4, 3, 5, 1}, encode(s, t, utf8, 3, 7));
                assertArrayEquals(new int[] {6, 1}, encode(s, t, utf8, 1, 7));
            }
        }

        @Test
        void wellFormedTextIsNeverDecodedAndNoneHandsOverTheRange() {
            Recording recording = new Recording(t.vocabulary());
            encode(compiled, recording, utf8, 0, utf8.length);
            encode(Specials.none(), recording, utf8, 1, 7);

            assertEquals(List.of("utf8:a", "utf8:b", "utf8:<|s|>b"), recording.calls);
        }

        @Test
        void illFormedBytesReachTheTokenizerAsTheyAre() {
            Recording recording = new Recording(t.vocabulary());
            // Latin-1, one byte per char: a stray continuation byte, a lead cut short by the
            // special, a lead cut by the end
            byte[] text = "\u0080<|s|>â<|s|>að\u009F".getBytes(ISO_8859_1);

            assertArrayEquals(new int[] {6, 6}, encode(compiled, recording, text, 0, text.length));
            assertEquals(List.of("utf8:\\x80", "utf8:\\xE2", "utf8:a\\xF0\\x9F"), recording.calls);
        }

        @Test
        void theDefaultDecodesIllFormedBytesToReplacementChars() {
            Recording recording = new Recording(t.vocabulary());
            byte[] text = "\u0080<|s|>a".getBytes(ISO_8859_1);

            encode(delegating(compiled), recording, text, 0, text.length);

            assertEquals(List.of("chars:�", "chars:a"), recording.calls);
        }

        private int[] encode(Specials s, Tokenizer tokenizer, byte[] text, int from, int to) {
            IntSequence.Builder out = IntSequence.newBuilder();
            s.encodeUtf8Into(tokenizer, text, from, to, out);
            return out.build().toArray();
        }
    }

    /**
     * The matcher's structure, each part against the alternation regex it replaced: specials past
     * 64 units, more opening pairs than the pair table has slots, pairs that share their first
     * bytes, a special at every offset of the 8-byte scan with few leads (scanned eight bytes at a
     * time) and with many (one at a time).
     */
    @Nested
    class Structure {

        private final String[] alphabet = {"a", "b", "<", "|", ">", "😀"};

        /** Specials drawn from a tiny alphabet, so that they overlap, share prefixes and cross. */
        @Test
        void randomSpecials() {
            Random random = new Random(42);
            for (int round = 0; round < 300; round++) {
                Set<String> drawn = new TreeSet<>();
                int longest = round % 8 == 0 ? 80 : 6; // some past 64 chars and bytes
                for (int k = 1 + random.nextInt(12); k > 0; k--) {
                    StringBuilder special = new StringBuilder();
                    for (int len = 1 + random.nextInt(longest); len > 0; len--) {
                        special.append(alphabet[random.nextInt(alphabet.length)]);
                    }
                    drawn.add(special.toString());
                }
                Set<String> specials = new LinkedHashSet<>(drawn); // prefixes of one another too
                String[] pool = specials.toArray(new String[0]);
                List<String> texts = new ArrayList<>();
                for (int t = 0; t < 20; t++) {
                    StringBuilder text = new StringBuilder();
                    for (int len = random.nextInt(80); len > 0; len--) {
                        String[] from = random.nextInt(4) == 0 ? pool : alphabet;
                        text.append(from[random.nextInt(from.length)]);
                    }
                    texts.add(text.toString());
                }
                assertMatchesTheRegex(chars(alphabet), specials, texts);
            }
        }

        @Test
        void oneCharSpecials() {
            // words across the specials: a text that misses one becomes a word
            List<String> words = List.of("a<", "<b", "b<", "|a", "a|", "a😀a", "😀a");
            List<String> vocabulary = chars(alphabet);
            vocabulary.addAll(words);
            List<String> texts = new ArrayList<>(words);
            texts.addAll(List.of("", "<", "a<b<<", "<b>b>", "||a|"));
            for (Set<String> specials :
                    List.of(Set.of("<"), Set.of("<", "b>"), Set.of("a", "😀"), Set.of("|", "<b"))) {
                assertMatchesTheRegex(vocabulary, specials, texts);
            }
        }

        @Test
        void specialsPast64Units() {
            String ascii = "<|" + "x".repeat(70) + "|>"; // 74 chars and bytes
            String emoji = "<|" + "😀".repeat(40) + "|>"; // 84 chars, 164 bytes
            assertMatchesTheRegex(
                    chars("<|>x😀"),
                    Set.of(ascii, emoji),
                    List.of(
                            "x" + ascii + "x" + emoji + "x",
                            ascii.substring(0, ascii.length() - 1),
                            emoji.substring(0, emoji.length() - 1) + "x",
                            ascii + ascii + emoji + emoji));
        }

        @Test
        void moreOpeningPairsThanThePairTableHasSlots() {
            // 5000 specials, each with its own first two chars (and bytes): slots are shared
            Set<String> specials = new LinkedHashSet<>();
            StringBuilder every = new StringBuilder();
            StringBuilder misses = new StringBuilder(); // each one's pair, then not its rest
            for (int i = 0; i < 5000; i++) {
                String s = "" + (char) ('!' + i / 90) + (char) ('!' + i % 90) + "~";
                specials.add(s);
                every.append(s).append(' ');
                misses.append(s, 0, 2).append(' ');
            }
            List<String> printable = new ArrayList<>();
            for (char c = ' '; c <= '~'; c++) printable.add(String.valueOf(c));
            assertMatchesTheRegex(
                    printable, specials, List.of(every.toString(), misses.toString()));
        }

        @Test
        void specialsSharingTheirFirstBytesDifferLater() {
            assertMatchesTheRegex(
                    chars("<|>_/abcdefghijklmnopqrstuvwxyz"),
                    Set.of("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>"),
                    List.of(
                            "<|im_sep|>",
                            "<|im_",
                            "<|im_end|><|im_start|>",
                            "<|endoftext",
                            "<thin><think></think>",
                            "<t<|<|im_end|>",
                            "<td><tr><title>x</title></p></div><think>"));
        }

        @Test
        void aSpecialAtEveryOffsetOfTheScanWithFewAndManyLeads() {
            for (Set<String> specials :
                    List.of(
                            Set.of("<|a|>"), // one lead byte: eight bytes at a time
                            Set.of("<|a|>", "[b]", "{c}", "(d)", "!e!", "#f#"))) { // six: one
                List<String> texts = new ArrayList<>();
                for (String special : specials) {
                    for (int offset = 0; offset < 25; offset++) {
                        String decoys = "y<".repeat(offset); // a lead not followed by its pair
                        String before = decoys.substring(0, offset);
                        texts.add(before + special + decoys.substring(0, offset % 9));
                        texts.add(before + special.substring(0, special.length() - 1));
                        texts.add(before + "<"); // a lead as the last byte
                    }
                }
                assertMatchesTheRegex(chars("y<|a>[b]{c}(d)!e#f"), specials, texts);
            }
        }
    }

    static Stream<String> encodings() {
        return TiktokenFixtures.encodings().stream().map(TiktokenFixtures.EncodingFixture::name);
    }

    /**
     * Real vocabularies with their specials, against a reference that owes toknroll nothing: the
     * specials split out by a regex, the text between encoded by JTokkit. These tokenizers encode
     * UTF-8 from the bytes.
     */
    @Nested
    class RealTokenizers {

        @ParameterizedTest(name = "{0}")
        @MethodSource("com.qxotic.toknroll.SpecialsTest#encodings")
        void theIdsOfARegexSplitAndJTokkit(String encoding) {
            Tokenizer tokenizer = TestTokenizers.tiktoken(encoding);
            Tokenizer jtokkit = TestTokenizers.tiktokenReference(encoding);
            Set<String> names = TiktokenFixtures.encoding(encoding).specialTokens().keySet();
            Specials specials = Specials.of(tokenizer);
            assertEquals(names, specials.tokens().keySet(), "the specials are the fixture's");
            List<String> texts = new ArrayList<>(List.of("only <| and |> and <|endof"));
            for (String special : names) {
                texts.add(special);
                texts.add("Hello " + special + " world");
                texts.add(special + special + "x" + special);
                texts.add("日本語" + special + "😀🎸 café");
                texts.add("  \n\t" + special + "\r\n  ");
                texts.add("don't" + special + "WE'LL 1234567 3.14159");
                texts.add(special.substring(0, special.length() - 1) + " cut short");
                texts.add("<|x" + special.substring(2)); // starts like one, is text
            }
            for (String text : texts) {
                int[] expected = regexReference(jtokkit, names, text);
                for (Input input : Input.values()) {
                    int[] ids = input.encode(specials, tokenizer, text);
                    assertArrayEquals(expected, ids, () -> encoding + " " + input + ": " + text);
                    assertEquals(text, tokenizer.decode(IntSequence.wrap(ids)), input::name);
                }
            }
        }

        @ParameterizedTest(name = "{0}")
        @MethodSource("com.qxotic.toknroll.SpecialsTest#encodings")
        void illFormedUtf8IsEncodedAsTheTokenizerEncodesItsBytes(String encoding) {
            Tokenizer tokenizer = TestTokenizers.tiktoken(encoding);
            Set<String> names = TiktokenFixtures.encoding(encoding).specialTokens().keySet();
            Specials specials = Specials.of(tokenizer);
            byte[] bad = "â\u0082".getBytes(ISO_8859_1); // a three-byte sequence cut short
            byte[] text = "â\u0082<|endoftext|>â\u0082".getBytes(ISO_8859_1);
            IntSequence.Builder out = IntSequence.newBuilder();

            specials.encodeUtf8Into(tokenizer, text, 0, text.length, out);

            IntSequence.Builder expected = IntSequence.newBuilder();
            tokenizer.encodeUtf8Into(bad, 0, bad.length, expected);
            expected.add(tokenizer.vocabulary().id("<|endoftext|>"));
            tokenizer.encodeUtf8Into(bad, 0, bad.length, expected);
            assertArrayEquals(expected.build().toArray(), out.build().toArray());
        }
    }

    @Test
    void oneInstanceServesManyThreads() {
        Tokenizer tokenizer = TestTokenizers.tiktoken("cl100k_base");
        Set<String> names = TiktokenFixtures.encoding("cl100k_base").specialTokens().keySet();
        Specials specials = Specials.of(tokenizer);
        String text = "Hi <|endoftext|> world <|fim_prefix|>日本<|fim_suffix|> ".repeat(200);
        Input[] inputs = Input.values();
        int[][] expected =
                Arrays.stream(inputs)
                        .map(i -> i.encode(specials, tokenizer, text))
                        .toArray(int[][]::new);

        IntStream.range(0, 400)
                .parallel()
                .forEach(
                        k ->
                                assertArrayEquals(
                                        expected[k % inputs.length],
                                        inputs[k % inputs.length].encode(
                                                specials, tokenizer, text)));
    }

    // ---- helpers ----

    /** A {@link Specials} of its own, so that {@link Specials#encodeUtf8Into}'s default runs. */
    private static Specials delegating(Specials compiled) {
        return new Specials() {
            @Override
            public Map<String, Integer> tokens() {
                return compiled.tokens();
            }

            @Override
            public void encodeInto(
                    Tokenizer tokenizer, CharSequence text, IntSequence.Builder out) {
                compiled.encodeInto(tokenizer, text, out);
            }
        };
    }

    /**
     * {@link Specials#of} a tokenizer like {@code t} in which the {@code names} are typed CONTROL:
     * the matcher {@code Specials.compile(vocabulary, names)} built before special tokens were
     * found by type.
     */
    private static Specials specials(Tokenizer t, Set<String> names) {
        Vocabulary vocabulary = t.vocabulary();
        String[] tokens = new String[vocabulary.size()];
        int[] types = new int[tokens.length];
        for (int id = 0; id < tokens.length; id++) {
            tokens[id] = vocabulary.token(id);
            types[id] =
                    (names.contains(tokens[id])
                                    ? StandardTokenType.CONTROL
                                    : StandardTokenType.NORMAL)
                            .getId();
        }
        assertTrue(Arrays.asList(tokens).containsAll(names), "every name is a token");
        return Specials.of(new Mini(ImplAccessor.createVocabulary(tokens, types), false));
    }

    /** Each char of the units, once, in order. */
    private static List<String> chars(String... units) {
        Set<String> chars = new LinkedHashSet<>();
        for (String unit : units) {
            for (char c : unit.toCharArray()) chars.add(String.valueOf(c));
        }
        return new ArrayList<>(chars);
    }

    /** Every {@link Input} gives the ids an alternation regex of the specials finds. */
    private static void assertMatchesTheRegex(
            List<String> tokens, Set<String> specials, List<String> texts) {
        Set<String> vocabulary = new LinkedHashSet<>(tokens);
        vocabulary.addAll(specials);
        Tokenizer tokenizer = Mini.of(vocabulary.toArray(new String[0]));
        Specials compiled = specials(tokenizer, specials);
        for (String text : texts) {
            int[] expected = regexReference(tokenizer, specials, text);
            for (Input input : Input.values()) {
                assertArrayEquals(
                        expected,
                        input.encode(compiled, tokenizer, text),
                        () -> input + ": " + specials + " in " + text);
            }
        }
    }

    /**
     * The specials split out by an alternation regex, leftmost-longest, the text between encoded by
     * the tokenizer.
     */
    private static int[] regexReference(Tokenizer tokenizer, Set<String> specials, String text) {
        // longest first: a regex alternation takes the first alternative that matches, so this
        // finds the leftmost special and, there, the longest
        List<String> longestFirst = new ArrayList<>(specials);
        longestFirst.sort(Comparator.comparingInt(String::length).reversed());
        StringBuilder alternation = new StringBuilder();
        for (String special : longestFirst) {
            alternation.append(alternation.length() == 0 ? "" : "|").append(Pattern.quote(special));
        }
        Matcher matcher = Pattern.compile(alternation.toString()).matcher(text);
        IntSequence.Builder out = IntSequence.newBuilder();
        int cursor = 0;
        while (matcher.find()) {
            tokenizer.encodeInto(text, cursor, matcher.start(), out);
            out.add(tokenizer.vocabulary().id(matcher.group()));
            cursor = matcher.end();
        }
        tokenizer.encodeInto(text, cursor, text.length(), out);
        return out.build().toArray();
    }

    /**
     * A tokenizer whose ids are easy to read: a span that is a whole token is that token, any other
     * span one char at a time. It can lowercase first, a stand-in for a normalizer.
     */
    private static final class Mini implements Tokenizer {
        private final Vocabulary vocabulary;
        private final boolean lowercase;

        Mini(Vocabulary vocabulary, boolean lowercase) {
            this.vocabulary = vocabulary;
            this.lowercase = lowercase;
        }

        static Mini of(String... tokens) {
            return new Mini(Toknroll.vocabulary(tokens), false);
        }

        static Mini lowercasing(String... tokens) {
            return new Mini(Toknroll.vocabulary(tokens), true);
        }

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            String span = text.subSequence(start, end).toString();
            if (lowercase) span = span.toLowerCase(Locale.ROOT);
            if (vocabulary.contains(span)) {
                out.add(vocabulary.id(span));
                return;
            }
            for (char c : span.toCharArray()) out.add(vocabulary.id(String.valueOf(c)));
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int at, ByteBuffer out) {
            if (at == tokens.length()) return 0;
            byte[] bytes = vocabulary.token(tokens.intAt(at)).getBytes(UTF_8);
            if (bytes.length > out.remaining()) // the contract's way: the caller grows and retries
            throw new IllegalArgumentException("token " + tokens.intAt(at) + " does not fit");
            out.put(bytes);
            return 1;
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            IntSequence.Builder out = IntSequence.newBuilder();
            encodeInto(text, start, end, out);
            return out.size();
        }
    }

    /** Records each span it is asked to encode, as chars or as UTF-8 (past ASCII as \xHH). */
    private static final class Recording implements Tokenizer {
        final List<String> calls = new ArrayList<>();
        private final Vocabulary vocabulary;

        Recording(Vocabulary vocabulary) {
            this.vocabulary = vocabulary;
        }

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            calls.add("chars:" + text.subSequence(start, end));
        }

        @Override
        public void encodeUtf8Into(byte[] utf8, int start, int end, IntSequence.Builder out) {
            StringBuilder seen = new StringBuilder("utf8:");
            for (int i = start; i < end; i++) {
                int b = utf8[i] & 0xFF;
                seen.append(b < 0x80 ? String.valueOf((char) b) : String.format("\\x%02X", b));
            }
            calls.add(seen.toString());
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int at, ByteBuffer out) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            throw new UnsupportedOperationException();
        }
    }

    // ---- Specials.of, leftmost-longest, gaps as ranges ----

    @Test
    void noneDelegatesToTokenizerEncode() {
        Tokenizer tokenizer = tokenizer(Map.of("hello", 1), Map.of());

        IntSequence expected = tokenizer.encode("hello");
        IntSequence actual = Specials.none().encode(tokenizer, "hello");

        assertArrayEquals(expected.toArray(), actual.toArray());
        assertTrue(Specials.none().tokens().isEmpty());
    }

    @Test
    void ofReturnsNoneWhenNoTokenIsSpecial() {
        Specials specials = Specials.of(tokenizer(Map.of("x", 1), Map.of()));

        assertSame(Specials.none(), specials);
        assertTrue(specials.tokens().isEmpty());
    }

    @Test
    void ofRejectsNullTokenizer() {
        assertThrows(NullPointerException.class, () -> Specials.of(null));
    }

    @Test
    void ofTakesOnlyControlUserDefinedAndUnknownTokens() {
        Map<String, Integer> ids = new LinkedHashMap<>();
        Map<Integer, StandardTokenType> types = new HashMap<>();
        StandardTokenType[] all = StandardTokenType.values();
        for (int i = 0; i < all.length; i++) {
            ids.put("<" + all[i] + ">", i);
            types.put(i, all[i]);
        }
        Specials specials = Specials.of(new MiniTokenizer(ids, types, Map.of(), false));

        assertEquals(
                Map.of(
                        "<CONTROL>", ids.get("<CONTROL>"),
                        "<USER_DEFINED>", ids.get("<USER_DEFINED>"),
                        "<UNKNOWN>", ids.get("<UNKNOWN>")),
                specials.tokens());
    }

    @Test
    void ofSkipsTokensThatDecodeToNoText() {
        Tokenizer tokenizer =
                new MiniTokenizer(
                        Map.of("<|empty|>", 1, "<|s|>", 2),
                        Map.of(1, StandardTokenType.CONTROL, 2, StandardTokenType.CONTROL),
                        Map.of(1, ""),
                        false);

        assertEquals(Map.of("<|s|>", 2), Specials.of(tokenizer).tokens());
    }

    /** DeepSeek's case: the vocabulary spells a special token differently from its text. */
    @Test
    void ofKnowsSpecialTokensByTheirDecodedTextNotTheirSpelling() {
        String text = "<｜User｜>", spelling = "<ï½ľUserï½ľ>"; // the byte-level spelling of text
        Tokenizer tokenizer =
                new MiniTokenizer(
                        Map.of(spelling, 128803, "hi", 1),
                        Map.of(128803, StandardTokenType.CONTROL),
                        Map.of(128803, text),
                        false);
        Specials specials = Specials.of(tokenizer);

        assertEquals(Map.of(text, 128803), specials.tokens());
        assertArrayEquals(new int[] {128803, 1}, specials.encode(tokenizer, text + "hi").toArray());
    }

    @Test
    void ofGivesTheLowerIdWhenTwoTokensDecodeToTheSameText() {
        Tokenizer tokenizer =
                new MiniTokenizer(
                        Map.of("<a>", 7, "<b>", 3),
                        Map.of(7, StandardTokenType.CONTROL, 3, StandardTokenType.CONTROL),
                        Map.of(7, "<s>", 3, "<s>"),
                        false);

        assertEquals(Map.of("<s>", 3), Specials.of(tokenizer).tokens());
    }

    /** Laguna's case: {@code 〈|} begins 60 others; every one of them stays reachable. */
    @Test
    void longestSpecialWinsWhereOneBeginsAnother() {
        Tokenizer tokenizer =
                tokenizer(Map.of("x", 1), Map.of("〈|", 14, "〈|EOS|〉", 2, "〈|UNK|〉", 0));
        Specials specials = Specials.of(tokenizer);

        assertArrayEquals(new int[] {2}, specials.encode(tokenizer, "〈|EOS|〉").toArray());
        assertArrayEquals(new int[] {0, 14}, specials.encode(tokenizer, "〈|UNK|〉〈|").toArray());
        assertArrayEquals(new int[] {14, 1}, specials.encode(tokenizer, "〈|x").toArray());
    }

    @Test
    void longestSpecialWinsRegardlessOfVocabularyOrder() {
        Map<String, Integer> specials = new LinkedHashMap<>();
        specials.put("<|z|>", 12);
        specials.put("<|a|>x", 11);
        specials.put("<|a|>", 10);
        Tokenizer tokenizer = tokenizer(Map.of("y", 1), specials);

        assertArrayEquals(
                new int[] {11, 10, 1, 12},
                Specials.of(tokenizer).encode(tokenizer, "<|a|>x<|a|>y<|z|>").toArray());
    }

    /** Overlap at different positions: the leftmost match wins, as in Hugging Face. */
    @Test
    void leftmostSpecialWinsOverALongerOneStartingLater() {
        Tokenizer tokenizer = tokenizer(Map.of("c", 3, "d", 4), Map.of("ab", 10, "bcd", 11));

        assertArrayEquals(
                new int[] {10, 3, 4}, Specials.of(tokenizer).encode(tokenizer, "abcd").toArray());
    }

    @Test
    void tokensMapsTextToIdAndIsImmutable() {
        Specials specials = Specials.of(tokenizer(Map.of(), Map.of("<|b|>", 2, "<|a|>", 1)));

        assertEquals(Map.of("<|a|>", 1, "<|b|>", 2), specials.tokens());
        assertThrows(UnsupportedOperationException.class, () -> specials.tokens().put("<|c|>", 3));
    }

    @Test
    void encodeInjectsSpecialIdsAndEncodesGaps() {
        Tokenizer tokenizer = tokenizer(Map.of("hi ", 10, " there", 11), Map.of("<|special|>", 42));

        IntSequence encoded = Specials.of(tokenizer).encode(tokenizer, "hi <|special|> there");

        assertArrayEquals(new int[] {10, 42, 11}, encoded.toArray());
    }

    @Test
    void encodeEmptyTextProducesEmptySequence() {
        Tokenizer tokenizer = tokenizer(Map.of(), Map.of("<|s|>", 9));

        assertTrue(Specials.of(tokenizer).encode(tokenizer, "").isEmpty());
    }

    @Test
    void encodeSingleSpecialProducesSingleSpecialId() {
        Tokenizer tokenizer = tokenizer(Map.of(), Map.of("<|s|>", 9));

        assertArrayEquals(
                new int[] {9}, Specials.of(tokenizer).encode(tokenizer, "<|s|>").toArray());
    }

    @Test
    void encodeHandlesSpecialsAtBothTextBoundaries() {
        Tokenizer tokenizer = tokenizer(Map.of("mid", 2), Map.of("<|a|>", 1, "<|b|>", 3));

        IntSequence encoded = Specials.of(tokenizer).encode(tokenizer, "<|a|>mid<|b|>");

        assertArrayEquals(new int[] {1, 2, 3}, encoded.toArray());
    }

    @Test
    void encodeHandlesAdjacentAndRepeatedSpecials() {
        Tokenizer tokenizer = tokenizer(Map.of("x", 3), Map.of("<|a|>", 1, "<|b|>", 2));

        IntSequence encoded = Specials.of(tokenizer).encode(tokenizer, "<|a|><|b|><|a|>x");

        assertArrayEquals(new int[] {1, 2, 1, 3}, encoded.toArray());
    }

    @Test
    void encodeWithSpecialsRoundTripsToOriginalText() {
        Tokenizer tokenizer =
                tokenizer(Map.of("hello ", 1, " world", 2), Map.of("<|special|>", 42));

        String text = "hello <|special|> world";
        IntSequence encoded = Specials.of(tokenizer).encode(tokenizer, text);

        assertEquals(text, tokenizer.decode(encoded));
    }

    @Test
    void encodeTreatsANormalTokenSpellingLikeText() {
        Tokenizer tokenizer =
                new MiniTokenizer(
                        Map.of("a<|other|>b", 55, "<|special|>", 99, "a", 1, "b", 2),
                        Map.of(99, StandardTokenType.CONTROL),
                        Map.of(),
                        true);

        assertArrayEquals(
                new int[] {55}, Specials.of(tokenizer).encode(tokenizer, "A<|other|>B").toArray());
    }

    @Test
    void encodeSupportsRegexMetaCharactersInSpecials() {
        String meta = "(.*)?+[]{}|^$\\";
        Tokenizer tokenizer = tokenizer(Map.of("x", 1, "y", 2), Map.of(meta, 77));

        IntSequence encoded = Specials.of(tokenizer).encode(tokenizer, "x" + meta + "y");

        assertArrayEquals(new int[] {1, 77, 2}, encoded.toArray());
    }

    @Test
    void encodeMatchesWhitespaceOnlyAndSupplementarySpecials() {
        Tokenizer tokenizer =
                tokenizer(Map.of("a", 1, " ", 2), Map.of("\n\n", 30, "🙂!", 31, "🙂🙂", 32));
        Specials specials = Specials.of(tokenizer);

        assertArrayEquals(
                new int[] {1, 30, 31, 32, 2},
                specials.encode(tokenizer, "a\n\n🙂!🙂🙂 ").toArray());
    }

    @Test
    void specialsAreSplitBeforeTokenizerPreprocessing() {
        Tokenizer tokenizer =
                new MiniTokenizer(
                        Map.of("a", 1, "b", 2, "a<|up|>b", 7, "<|UP|>", 99),
                        Map.of(99, StandardTokenType.CONTROL),
                        Map.of(),
                        true);

        assertArrayEquals(new int[] {7}, tokenizer.encode("A<|UP|>B").toArray());
        assertArrayEquals(
                new int[] {1, 99, 2},
                Specials.of(tokenizer).encode(tokenizer, "A<|UP|>B").toArray());
    }

    /** SentencePiece tokenizers need the gaps as ranges of the one input, empty ones skipped. */
    @Test
    void gapsAreEncodedAsRangesOfTheInputAndEmptyOnesSkipped() {
        MiniTokenizer tokenizer =
                (MiniTokenizer) tokenizer(Map.of("x", 1, "y", 2), Map.of("<a>", 10, "<b>", 11));
        Specials specials = Specials.of(tokenizer);

        specials.encode(tokenizer, "<a>x<a><b>yy<b>");

        assertEquals(List.of("3-4", "10-12"), tokenizer.ranges);
    }

    @Test
    void encodeIntoAppendsToProvidedBuilder() {
        Tokenizer tokenizer = tokenizer(Map.of("x", 1), Map.of("<|s|>", 9));

        IntSequence.Builder out = IntSequence.newBuilder();
        out.add(1000);
        Specials.of(tokenizer).encodeInto(tokenizer, "x<|s|>x", out);

        assertArrayEquals(new int[] {1000, 1, 9, 1}, out.build().toArray());
    }

    @Test
    void noneEncodeIntoAppendsLikeTokenizerEncodeInto() {
        Tokenizer tokenizer = tokenizer(Map.of("x", 1), Map.of());
        IntSequence.Builder out = IntSequence.newBuilder();
        out.add(7);

        Specials.none().encodeInto(tokenizer, "x", out);

        assertArrayEquals(new int[] {7, 1}, out.build().toArray());
    }

    @Test
    void encodeAndEncodeIntoValidateNulls() {
        Tokenizer tokenizer = tokenizer(Map.of("x", 1), Map.of("<|s|>", 9));
        Specials specials = Specials.of(tokenizer);
        IntSequence.Builder out = IntSequence.newBuilder();

        assertThrows(NullPointerException.class, () -> specials.encode(null, "x"));
        assertThrows(NullPointerException.class, () -> specials.encode(tokenizer, null));
        assertThrows(NullPointerException.class, () -> specials.encodeInto(null, "x", out));
        assertThrows(NullPointerException.class, () -> specials.encodeInto(tokenizer, null, out));
        assertThrows(NullPointerException.class, () -> specials.encodeInto(tokenizer, "x", null));
    }

    /**
     * The matching rule against its definition, on random texts and heavily overlapping special
     * tokens: at each position the longest special token starting there wins.
     */
    @Test
    void matchesTheBruteForceDefinitionOnRandomInput() {
        Random random = new Random(42);
        String alphabet = "ab<|>";
        for (int round = 0; round < 10_000; round++) {
            Map<String, Integer> normal = new HashMap<>();
            for (int i = 0; i < alphabet.length(); i++) normal.put(alphabet.substring(i, i + 1), i);
            Map<String, Integer> specials = new HashMap<>();
            for (int i = 0, n = 1 + random.nextInt(6); i < n; i++) {
                String special = randomText(random, alphabet, 1 + random.nextInt(4));
                if (!normal.containsKey(special)) specials.putIfAbsent(special, 100 + i);
            }
            Tokenizer tokenizer = tokenizer(normal, specials);
            String text = randomText(random, alphabet, random.nextInt(30));

            assertEquals(
                    bruteForce(tokenizer, specials, text),
                    toList(Specials.of(tokenizer).encode(tokenizer, text)),
                    () -> "specials " + specials + ", text '" + text + "'");
        }
    }

    private static List<Integer> bruteForce(
            Tokenizer tokenizer, Map<String, Integer> specials, String text) {
        IntSequence.Builder out = IntSequence.newBuilder();
        int gap = 0, at = 0;
        while (at < text.length()) {
            String best = null;
            for (String special : specials.keySet()) {
                if (text.startsWith(special, at)
                        && (best == null || special.length() > best.length())) best = special;
            }
            if (best == null) {
                at++;
                continue;
            }
            if (at > gap) tokenizer.encodeInto(text, gap, at, out);
            out.add(specials.get(best));
            at += best.length();
            gap = at;
        }
        if (gap < text.length()) tokenizer.encodeInto(text, gap, text.length(), out);
        return toList(out.build());
    }

    private static String randomText(Random random, String alphabet, int length) {
        StringBuilder text = new StringBuilder(length);
        for (int i = 0; i < length; i++)
            text.append(alphabet.charAt(random.nextInt(alphabet.length())));
        return text.toString();
    }

    private static List<Integer> toList(IntSequence ids) {
        List<Integer> list = new ArrayList<>(ids.length());
        for (int i = 0; i < ids.length(); i++) list.add(ids.intAt(i));
        return list;
    }

    /** A tokenizer whose {@code specials} are CONTROL tokens, each decoding to its spelling. */
    private static Tokenizer tokenizer(Map<String, Integer> normal, Map<String, Integer> specials) {
        Map<String, Integer> all = new LinkedHashMap<>(normal);
        all.putAll(specials);
        Map<Integer, StandardTokenType> types = new HashMap<>();
        for (int id : specials.values()) types.put(id, StandardTokenType.CONTROL);
        return new MiniTokenizer(all, types, Map.of(), false);
    }

    /**
     * Encodes a range whole when the vocabulary has it, else character by character; decodes an id
     * to its {@code decoded} text, else its spelling. Records the ranges it is asked to encode.
     */
    private static final class MiniTokenizer implements Tokenizer {
        private final MapVocabulary vocabulary;
        private final Map<Integer, String> decoded;
        private final boolean lowercaseBeforeEncode;
        final List<String> ranges = new ArrayList<>();

        MiniTokenizer(
                Map<String, Integer> tokenToId,
                Map<Integer, StandardTokenType> types,
                Map<Integer, String> decoded,
                boolean lowercaseBeforeEncode) {
            this.vocabulary = new MapVocabulary(tokenToId, types);
            this.decoded = decoded;
            this.lowercaseBeforeEncode = lowercaseBeforeEncode;
        }

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(
                CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
            ranges.add(startInclusive + "-" + endExclusive);
            String slice = text.subSequence(startInclusive, endExclusive).toString();
            if (lowercaseBeforeEncode) {
                slice = slice.toLowerCase(Locale.ROOT);
            }
            Integer whole = vocabulary.tokenToId.get(slice);
            if (whole != null) {
                out.add(whole);
                return;
            }
            for (int i = 0; i < slice.length(); ) {
                int width = Character.charCount(slice.codePointAt(i));
                String unit = slice.substring(i, i + width);
                Integer id = vocabulary.tokenToId.get(unit);
                if (id == null) {
                    throw new NoSuchElementException(unit);
                }
                out.add(id);
                i += width;
            }
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            int length = tokens.length();
            if (tokenStartIndex < 0 || tokenStartIndex > length) {
                throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
            }
            if (tokenStartIndex == length) {
                return 0;
            }
            int id = tokens.intAt(tokenStartIndex);
            String token = decoded.getOrDefault(id, vocabulary.token(id));
            byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
            if (bytes.length > out.remaining()) {
                throw new IllegalArgumentException("Not enough output space");
            }
            out.put(bytes);
            return 1;
        }

        @Override
        public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
            IntSequence.Builder out = IntSequence.newBuilder();
            encodeInto(text, startInclusive, endExclusive, out);
            return out.size();
        }
    }

    private static final class MapVocabulary implements Vocabulary {
        private final Map<String, Integer> tokenToId;
        private final Map<Integer, String> idToToken;
        private final Map<Integer, StandardTokenType> types;

        MapVocabulary(Map<String, Integer> tokenToId, Map<Integer, StandardTokenType> types) {
            this.tokenToId = new LinkedHashMap<>(tokenToId);
            this.idToToken = new LinkedHashMap<>();
            this.types = types;
            for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
                String previous = idToToken.put(entry.getValue(), entry.getKey());
                if (previous != null && !previous.equals(entry.getKey())) {
                    throw new IllegalArgumentException(
                            "Duplicate id "
                                    + entry.getValue()
                                    + " for "
                                    + previous
                                    + " and "
                                    + entry.getKey());
                }
            }
        }

        @Override
        public int size() {
            return tokenToId.size();
        }

        @Override
        public String token(int id) {
            String token = idToToken.get(id);
            if (token == null) {
                throw new NoSuchElementException(String.valueOf(id));
            }
            return token;
        }

        @Override
        public int id(String text) {
            Integer id = tokenToId.get(text);
            if (id == null) {
                throw new NoSuchElementException(text);
            }
            return id;
        }

        @Override
        public boolean contains(int id) {
            return idToToken.containsKey(id);
        }

        @Override
        public boolean contains(String text) {
            return tokenToId.containsKey(text);
        }

        @Override
        public boolean isTokenOfType(int id, TokenType type) {
            token(id);
            return types.getOrDefault(id, StandardTokenType.NORMAL) == type;
        }

        @Override
        public Iterator<Map.Entry<String, Integer>> iterator() {
            return tokenToId.entrySet().iterator();
        }
    }
}
