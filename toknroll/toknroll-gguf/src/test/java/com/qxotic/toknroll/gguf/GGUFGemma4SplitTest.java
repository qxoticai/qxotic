package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.testkit.corpus.Enwik8Corpus;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Gemma 4's word split on its real vocabulary, against its regex {@code [^\n]+|[\n]+}: it only cuts
 * the regex's pieces further, where no token spans the cut, so the ids are the regex's. The rule
 * itself is pinned in core's SentencePieceSplitterTest.
 */
class GGUFGemma4SplitTest {

    private static final Pattern LINES =
            Pattern.compile("[^\\n]+|[\\n]+", Pattern.UNICODE_CHARACTER_CLASS);

    private static final String[] PIECES = {
        "a", "the", " ", "  ", "\n", "\n\n", "\r", "\r\n", "\t", ">", "<", "/", "</", "> </", "=",
        "->", "▁", "café", "日本", "😀", "1", "don't", " \n", "x "
    };

    static String random(Random random, int pieces) {
        StringBuilder text = new StringBuilder();
        for (int n = pieces; n > 0; n--) text.append(PIECES[random.nextInt(PIECES.length)]);
        return text.toString();
    }

    /** The real vocabulary: the ids of the regex's lines, merged whole, on text and on UTF-8. */
    @Nested
    @Tag("network")
    @Tag("local-external")
    class Gemma4Vocabulary {

        private final Tokenizer fast = load(GGUFTokenizerLoader.createBuilderWithBuiltins());
        private final Tokenizer regex =
                load(
                        GGUFTokenizerLoader.createBuilderWithBuiltins()
                                .registerPreTokenizer("gemma4", gguf -> Splitter.regex(LINES)));

        private Tokenizer load(GGUFTokenizerLoader.Builder builder) {
            return builder.build()
                    .fromHuggingFace("unsloth", "gemma-4-E2B-it-GGUF", "gemma-4-E2B-it-Q8_0.gguf");
        }

        private void same(String text) {
            int[] expected = regex.encode(text).toArray();
            assertArrayEquals(expected, fast.encode(text).toArray(), text);
            assertArrayEquals(
                    expected,
                    fast.encodeUtf8(text.getBytes(StandardCharsets.UTF_8)).toArray(),
                    text);
        }

        @Test
        void theSplitIsTheWordSplit() {
            assertEquals(
                    "SentencePieceSplitter",
                    ((TokenizationPipeline) fast)
                            .splitter()
                            .orElseThrow()
                            .getClass()
                            .getSimpleName());
        }

        @Test
        void fuzz() {
            Random random = new Random(0x6E3A4);
            for (int i = 0; i < 20_000; i++) same(random(random, random.nextInt(40)));
            for (int i = 0; i < 20; i++) same(random(random, 5_000));
        }

        @Test
        void wordsAfterEveryCharOfTheVocabulary() {
            // every char a token holds, before and after a space: no cut may separate a token
            StringBuilder text = new StringBuilder();
            for (int id = 0; id < fast.vocabulary().size(); id += 7) {
                text.append(fast.vocabulary().token(id)).append(id % 3 == 0 ? "\n" : "");
            }
            same(text.toString());
        }

        @Tag("corpus")
        @Test
        void enwik8() {
            same(new String(Enwik8Corpus.load().data(), StandardCharsets.UTF_8));
        }
    }
}
