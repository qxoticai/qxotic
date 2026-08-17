package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Test;

/** Thinking-budget behavior over a tiny tokenizer, without a model. */
class ThinkingTest {

    private static final String[] TOKENS = {"<think>", "</think>", "\n", "a", "b"};
    private static final int OPEN = 0, CLOSE = 1, NEWLINE = 2, A = 3, B = 4;
    private static final Tokenizer TOKENIZER = new FakeTokenizer();

    @Test
    void spentBudgetClosesAtAParagraphBoundaryThenBansBothMarkers() {
        Sampler capped = Thinking.capBudget(Sampler.ARGMAX, TOKENIZER, 2, true);
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(OPEN)));
        assertEquals(A, capped.sampleToken(logits(CLOSE)));
    }

    @Test
    void reopeningWithinTheBudgetRemainsLegal() {
        Sampler capped = Thinking.capBudget(Sampler.ARGMAX, TOKENIZER, 5, true);
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(CLOSE)));
        assertEquals(B, capped.sampleToken(logits(B)));
        assertEquals(OPEN, capped.sampleToken(logits(OPEN)));
        assertEquals(A, capped.sampleToken(logits(A)));
    }

    @Test
    void aNullMessageIsExactlyTheBareBreak() {
        Sampler capped = Thinking.capBudget(Sampler.ARGMAX, TOKENIZER, 2, true, null);
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
    }

    @Test
    void aBlankMessageIsAlsoTheBareBreak() {
        Sampler capped = Thinking.capBudget(Sampler.ARGMAX, TOKENIZER, 2, true, "  ");
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
    }

    @Test
    void aMessageIsForcedBetweenParagraphBreaksThenBothMarkersStayBanned() {
        Sampler capped = Thinking.capBudget(Sampler.ARGMAX, TOKENIZER, 2, true, "ab");
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(A)));
        // "\n\n" + "ab" + "\n\n", then the close - the ONLY close id in the sequence:
        // encoding is the non-special-aware path, so message text can never inject a marker
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(B, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
        // the ping-pong guard holds with a message spent: both markers stay banned
        assertEquals(A, capped.sampleToken(logits(OPEN)));
        assertEquals(A, capped.sampleToken(logits(CLOSE)));
    }

    @Test
    void startInThinkArmsTheMessagePath() {
        Sampler capped = Thinking.capBudget(Sampler.ARGMAX, TOKENIZER, 1, false, "ab");
        assertEquals(OPEN, capped.sampleToken(logits(OPEN))); // the model opens the span itself
        assertEquals(A, capped.sampleToken(logits(A))); // thought = 1 = the budget
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(B, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
    }

    @Test
    void aMessageTheTokenizerCannotEncodeClosesHard() {
        // '<' is not in the fake's alphabet: the encode fails, and the close still lands
        Sampler capped = Thinking.capBudget(Sampler.ARGMAX, TOKENIZER, 1, true, "<nope>");
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
    }

    private static MemoryView<MemorySegment> logits(int favorite) {
        float[] values = new float[TOKENS.length];
        values[favorite] = 2;
        values[A] = Math.max(values[A], 1);
        return Views.fromFloatArray(new PanamaMemoryArena(Arena.ofAuto()), values);
    }

    private static final class FakeTokenizer implements Tokenizer {
        private final Vocabulary vocabulary = new FakeVocabulary();

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            for (int i = start; i < end; i++) {
                out.add(
                        switch (text.charAt(i)) {
                            case '\n' -> NEWLINE;
                            case 'a' -> A;
                            case 'b' -> B;
                            default ->
                                    throw new IllegalArgumentException(
                                            "character " + text.charAt(i));
                        });
            }
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return end - start;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex == tokens.length()) return 0;
            out.put(TOKENS[tokens.intAt(tokenStartIndex)].getBytes(StandardCharsets.UTF_8));
            return 1;
        }
    }

    private static final class FakeVocabulary implements Vocabulary {
        @Override
        public int size() {
            return TOKENS.length;
        }

        @Override
        public String token(int id) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            return TOKENS[id];
        }

        @Override
        public int id(String text) {
            for (int i = 0; i < TOKENS.length; i++) if (TOKENS[i].equals(text)) return i;
            throw new NoSuchElementException(text);
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < TOKENS.length;
        }

        @Override
        public boolean contains(String text) {
            for (String token : TOKENS) if (token.equals(text)) return true;
            return false;
        }

        @Override
        public boolean isTokenOfType(int id, TokenType type) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            if (type == StandardTokenType.CONTROL) return id == OPEN || id == CLOSE;
            if (type == StandardTokenType.NORMAL) return id != OPEN && id != CLOSE;
            return false;
        }

        @Override
        public Iterator<Map.Entry<String, Integer>> iterator() {
            return java.util.stream.IntStream.range(0, TOKENS.length)
                    .mapToObj(i -> Map.entry(TOKENS[i], i))
                    .iterator();
        }
    }
}
