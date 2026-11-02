package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Toknroll;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The SentencePiece word cut: it goes only where no token can span it, so a merge model encodes the
 * words as it would the whole text.
 */
class SentencePieceSplitterTest {

    private static final Pattern LINES = Pattern.compile("[^\\n]+|[\\n]+");

    static List<String> spans(Splitter splitter, String text) {
        List<String> out = new ArrayList<>();
        splitter.splitAll(text, (t, s, e) -> out.add(t.subSequence(s, e).toString()));
        return out;
    }

    private static Splitter words(boolean isolateNewlines, String... tokens) {
        return new SentencePieceSplitter(tokens, isolateNewlines);
    }

    /** Every example in the class's javadoc, as written there. */
    @Test
    void theJavadocExamples() {
        assertEquals(List.of("hello", " world"), spans(words(false, "a", " b"), "hello world"));
        assertEquals(List.of("x>", " </"), spans(words(false, "a", " b"), "x> </"));
        assertEquals(List.of("x> </"), spans(words(false, "> </"), "x> </"));
        assertEquals(List.of("a", " ", " ", " b"), spans(words(false, "a"), "a   b"));
        assertEquals(List.of("a", "   b"), spans(words(false, "  "), "a   b"));
        assertEquals(
                List.of("ab", "\n\n", "cd", " ef"), spans(words(true, "a", " b"), "ab\n\ncd ef"));
        assertEquals(List.of("ab\n\ncd", " ef"), spans(words(false, "a", " b"), "ab\n\ncd ef"));
    }

    @Test
    void onlyASpaceStartsAWord() {
        Splitter words = words(false, "a");
        assertEquals(List.of("a\t", " b"), spans(words, "a\t b"));
        assertEquals(List.of("a\n", " b"), spans(words, "a\n b"));
        assertEquals(List.of("a\tb\nc"), spans(words, "a\tb\nc"));
        assertEquals(List.of(" a"), spans(words, " a"));
        assertEquals(List.of(), spans(words, ""));
    }

    /** A space spelled U+2581 is a space: in the text, the tokens, or one and not the other. */
    @Test
    void aSpaceSpelledU2581() {
        // Hugging Face: both spelled U+2581
        assertEquals(List.of("a", "\u2581b"), spans(words(false, "x"), "a\u2581b"));
        assertEquals(
                List.of("a\u2581\u2581b"),
                spans(words(false, "a\u2581", "\u2581\u2581"), "a\u2581\u2581b"));
        assertEquals(List.of("a", "\u2581", "\u2581b"), spans(words(false, "x"), "a\u2581\u2581b"));
        // GGUF: the vocabulary spelled U+2581, the text ' ', as the model reads its tokens
        assertEquals(List.of("a", "  b"), spans(words(false, "\u2581\u2581"), "a  b"));
        assertEquals(List.of("x> </"), spans(words(false, ">\u2581</"), "x> </"));
    }

    @Test
    void isolatedNewlineRunsArePiecesOfTheirOwn() {
        assertEquals(List.of("a", "\n\n", " b"), spans(words(true, "x"), "a\n\n b"));
        assertEquals(List.of("a", " ", "\n", "b"), spans(words(true, "x"), "a \nb"));
        assertEquals(List.of("a\n\nb"), spans(words(false, "x"), "a\n\nb"));
    }

    /**
     * Isolating newlines, every piece lies inside one of the regex's, and together they make it up.
     */
    @Test
    void isolatedNewlinesOnlyRefineTheRegexs() {
        Splitter regex = Splitter.regex(LINES);
        Splitter words = words(true, "a", " b", "> </", "  ");
        Random random = new Random(4);
        for (int i = 0; i < 20_000; i++) {
            String text = text(random, random.nextInt(30));
            List<String> ours = spans(words, text);
            int k = 0;
            for (String piece : spans(regex, text)) {
                StringBuilder joined = new StringBuilder();
                while (joined.length() < piece.length()) joined.append(ours.get(k++));
                assertEquals(piece, joined.toString(), text);
            }
            assertEquals(ours.size(), k, text);
        }
    }

    /** The point of it: random vocabularies, many spanning spaces, encode words as the whole. */
    @ParameterizedTest
    @ValueSource(chars = {' ', '\u2581'})
    void wordsEncodeAsTheWhole(char space) {
        Random random = new Random(0x5EED + space);
        for (int v = 0; v < 40; v++) {
            String[] tokens = vocabulary(random, space);
            TokenizationModel model = model(random, tokens);
            Splitter words = new SentencePieceSplitter(tokens, false);
            Splitter isolated = new SentencePieceSplitter(tokens, true);
            for (int i = 0; i < 300; i++) {
                String text = text(random, random.nextInt(60)).replace(' ', space);
                assertArrayEquals(
                        model.encode(text).toArray(),
                        Toknroll.pipeline(words, model).encode(text).toArray(),
                        text);
                assertArrayEquals(
                        Toknroll.pipeline(Splitter.regex(LINES), model).encode(text).toArray(),
                        Toknroll.pipeline(isolated, model).encode(text).toArray(),
                        text);
            }
        }
    }

    private static final String[] PIECES = {
        "a", "b", "ab", " ", "  ", "\n", "\n\n", "\t", ">", "<", "/", "</", "> </", "é", "日", "😀"
    };

    private static String text(Random random, int pieces) {
        StringBuilder text = new StringBuilder();
        for (int n = pieces; n > 0; n--) text.append(PIECES[random.nextInt(PIECES.length)]);
        return text.toString();
    }

    /** Byte fallback, single chars, and random substrings of random text: many cross a space. */
    private static String[] vocabulary(Random random, char space) {
        Set<String> tokens = new LinkedHashSet<>();
        for (int b = 0; b < 256; b++) tokens.add(String.format("<0x%02X>", b));
        for (String piece : PIECES)
            piece.codePoints().forEach(cp -> tokens.add(Character.toString(cp)));
        String source = text(random, 400).replace(' ', space);
        for (int n = 0; n < 150; n++) {
            int start = random.nextInt(source.length() - 6);
            int end = start + 2 + random.nextInt(4);
            if (!Character.isLowSurrogate(source.charAt(start))
                    && !Character.isHighSurrogate(source.charAt(end - 1))) {
                tokens.add(source.substring(start, end));
            }
        }
        return tokens.toArray(new String[0]);
    }

    private static TokenizationModel model(Random random, String[] tokens) {
        int[] types = new int[tokens.length];
        float[] scores = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            boolean isByte = i < 256;
            types[i] = (isByte ? StandardTokenType.BYTE : StandardTokenType.NORMAL).getId();
            scores[i] = isByte ? 0f : random.nextFloat();
        }
        return SentencePieceBpeModel.fromVocabulary(new VocabularyImpl(tokens, types), scores);
    }
}
