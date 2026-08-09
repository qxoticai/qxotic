package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Test;

/**
 * {@link Thinking#capBudget}: the budget forces the close AND bans re-opening once spent - a model
 * force-closed mid-thought re-opens on its very next token (greedy Qwen3.5 does,
 * deterministically), and without the ban the cap ping-pongs forced closes against re-opens until
 * LENGTH with a blank visible answer.
 */
final class ThinkingCapBudgetTest {

    static final String[] SPECIALS = {"<think>", "</think>"};
    static final String CHARS = "ab";
    static final int OPEN = 0, CLOSE = 1, A = 2, B = 3;

    static final Tokenizer TOK = new FakeTokenizer();

    static F32FloatTensor logits(int favorite) {
        F32FloatTensor l =
                F32FloatTensor.allocate(Arena.ofAuto(), SPECIALS.length + CHARS.length());
        for (int i = 0; i < SPECIALS.length + CHARS.length(); i++) l.setFloat(i, 0f);
        l.setFloat(favorite, 2f);
        l.setFloat(A, 1f); // the runner-up a ban falls back to (unless A is the favorite)
        return l;
    }

    @Test
    void aSpentBudgetForcesTheCloseThenBansReopening() {
        Sampler capped = Thinking.capBudget(FloatTensor::argmax, TOK, 2, true);
        assertEquals(A, capped.sampleToken(logits(A)), "thinking, 1 of 2");
        assertEquals(A, capped.sampleToken(logits(A)), "thinking, 2 of 2");
        assertEquals(CLOSE, capped.sampleToken(logits(A)), "the budget forces the close");
        // the model's own preference is now to REOPEN - the spent budget must ban it, so the
        // sampler falls to the runner-up: visible answer text, not another think span
        assertEquals(A, capped.sampleToken(logits(OPEN)), "reopening is banned once spent");
        // with only the open banned, a greedy model's next-best is the PAIRED CLOSE - both
        // markers are scaffold noise once the budget is spent
        assertEquals(A, capped.sampleToken(logits(CLOSE)), "the close is banned too");
        assertEquals(A, capped.sampleToken(logits(OPEN)), "and stays banned");
    }

    @Test
    void reopeningWithinTheBudgetStaysLegal() {
        Sampler capped = Thinking.capBudget(FloatTensor::argmax, TOK, 5, true);
        assertEquals(A, capped.sampleToken(logits(A)), "thinking, 1 of 5");
        assertEquals(CLOSE, capped.sampleToken(logits(CLOSE)), "the model closes on its own");
        assertEquals(B, capped.sampleToken(logits(B)), "content");
        assertEquals(
                OPEN,
                capped.sampleToken(logits(OPEN)),
                "budget remains: reopen is the model's right");
        assertEquals(A, capped.sampleToken(logits(A)), "thinking again, 2 of 5");
    }

    // ---- fixture -----------------------------------------------------------

    private static final class FakeTokenizer implements Tokenizer {
        private final Vocabulary vocab = new FakeVocabulary();

        @Override
        public Vocabulary vocabulary() {
            return vocab;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            for (int i = start; i < end; i++)
                out.add(SPECIALS.length + CHARS.indexOf(text.charAt(i)));
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return end - start;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex == tokens.length()) return 0;
            int id = tokens.intAt(tokenStartIndex);
            String piece =
                    id < SPECIALS.length
                            ? SPECIALS[id]
                            : String.valueOf(CHARS.charAt(id - SPECIALS.length));
            out.put(piece.getBytes(StandardCharsets.UTF_8));
            return 1;
        }
    }

    private static final class FakeVocabulary implements Vocabulary {
        @Override
        public int size() {
            return SPECIALS.length + CHARS.length();
        }

        @Override
        public String token(int id) {
            return id < SPECIALS.length
                    ? SPECIALS[id]
                    : String.valueOf(CHARS.charAt(id - SPECIALS.length));
        }

        @Override
        public int id(String text) {
            for (int i = 0; i < SPECIALS.length; i++) {
                if (SPECIALS[i].equals(text)) return i;
            }
            int at = text.length() == 1 ? CHARS.indexOf(text.charAt(0)) : -1;
            if (at < 0) throw new NoSuchElementException(text);
            return SPECIALS.length + at;
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < size();
        }

        @Override
        public boolean contains(String text) {
            try {
                id(text);
                return true;
            } catch (NoSuchElementException e) {
                return false;
            }
        }

        @Override
        public boolean isTokenOfType(int id, TokenType type) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            boolean special = id < SPECIALS.length;
            if (type == StandardTokenType.NORMAL) return !special;
            if (type == StandardTokenType.CONTROL) return special;
            return false;
        }

        @Override
        public Iterator<Map.Entry<String, Integer>> iterator() {
            return Collections.emptyIterator();
        }
    }
}
