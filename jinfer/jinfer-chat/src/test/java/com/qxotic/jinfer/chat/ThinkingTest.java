package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jota.memory.MemoryAllocators;
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
import java.util.stream.IntStream;

/** Thinking-budget behavior over a tiny tokenizer, without a model. */
class ThinkingTest {

    private static final String[] TOKENS = {"<think>", "</think>", "\n", "a", "b"};
    private static final int OPEN = 0, CLOSE = 1, NEWLINE = 2, A = 3, B = 4;
    private static final Tokenizer TOKENIZER = new FakeTokenizer(TOKENS);

    // Gemma 4's channel span: <|channel> ... <channel|> (the generic <think> pair is absent)
    private static final String[] GEMMA_TOKENS = {"<|channel>", "<channel|>", "\n", "a", "b"};
    private static final int G_OPEN = 0, G_CLOSE = 1, G_NEWLINE = 2, G_A = 3, G_B = 4;
    private static final Tokenizer GEMMA_TOKENIZER = new FakeTokenizer(GEMMA_TOKENS);

    // a vocabulary with no think markers at all: every policy must be a no-op over it
    private static final String[] PLAIN_TOKENS = {"\n", "a", "b"};
    private static final int P_A = 1;
    private static final Tokenizer PLAIN_TOKENIZER = new FakeTokenizer(PLAIN_TOKENS);

    @Test
    void bannedMarkersNeverWinTheArgmax() {
        Sampler banned =
                Thinking.banMarkers(Sampler.ARGMAX, TOKENIZER, Thinking.OPEN, Thinking.CLOSE);
        assertEquals(A, banned.sampleToken(logits(OPEN))); // the open cannot start a span
        assertEquals(A, banned.sampleToken(logits(CLOSE))); // the close cannot leak either
        assertEquals(A, banned.sampleToken(logits(A))); // content is untouched
    }

    @Test
    void gemmaChannelMarkersAreBanned() {
        Sampler banned =
                Thinking.banMarkers(Sampler.ARGMAX, GEMMA_TOKENIZER, "<|channel>", "<channel|>");
        assertEquals(G_A, banned.sampleToken(logits(GEMMA_TOKENS, G_OPEN)));
        assertEquals(G_A, banned.sampleToken(logits(GEMMA_TOKENS, G_CLOSE)));
        assertEquals(G_A, banned.sampleToken(logits(GEMMA_TOKENS, G_A)));
    }

    @Test
    void banningOnATokenizerWithoutMarkersIsANoOp() {
        Sampler banned =
                Thinking.banMarkers(Sampler.ARGMAX, PLAIN_TOKENIZER, Thinking.OPEN, Thinking.CLOSE);
        assertEquals(P_A, banned.sampleToken(logits(PLAIN_TOKENS, P_A)));
    }

    @Test
    void aNegativeBudgetNeverCapsNorBans() {
        Sampler uncapped = cap(-1, true, null);
        for (int i = 0; i < 10; i++) {
            assertEquals(A, uncapped.sampleToken(logits(A)));
        }
        // the model closes and re-opens on its own: markers stay legal
        assertEquals(CLOSE, uncapped.sampleToken(logits(CLOSE)));
        assertEquals(OPEN, uncapped.sampleToken(logits(OPEN)));
    }

    @Test
    void cappingOnATokenizerWithoutMarkersIsANoOp() {
        Sampler capped =
                Thinking.capBudget(
                        Sampler.ARGMAX,
                        PLAIN_TOKENIZER,
                        1,
                        true,
                        null,
                        Thinking.OPEN,
                        Thinking.CLOSE);
        assertEquals(P_A, capped.sampleToken(logits(PLAIN_TOKENS, P_A)));
        assertEquals(P_A, capped.sampleToken(logits(PLAIN_TOKENS, P_A)));
    }

    @Test
    void aZeroBudgetBansBothMarkersFromTheFirstDraw() {
        Sampler capped = cap(0, false, null);
        // thought(0) >= budget(0) before anything is sampled: a zero budget is a ban
        assertEquals(A, capped.sampleToken(logits(OPEN)));
        assertEquals(A, capped.sampleToken(logits(CLOSE)));
        assertEquals(A, capped.sampleToken(logits(A)));
    }

    @Test
    void theBudgetIsCumulativeAcrossSpans() {
        Sampler capped = cap(2, false, null);
        assertEquals(OPEN, capped.sampleToken(logits(OPEN)));
        assertEquals(A, capped.sampleToken(logits(A))); // span 1: thought = 1
        assertEquals(CLOSE, capped.sampleToken(logits(CLOSE)));
        assertEquals(A, capped.sampleToken(logits(A))); // visible content is free
        assertEquals(OPEN, capped.sampleToken(logits(OPEN))); // reopen within budget is legal
        assertEquals(A, capped.sampleToken(logits(A))); // span 2: thought = 2 = budget, cap fires
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(OPEN))); // spent: reopen banned
    }

    @Test
    void forcedTokensConsumeNoInnerDraws() {
        int[] draws = {0};
        Sampler counting =
                logits -> {
                    draws[0]++;
                    return Sampler.ARGMAX.sampleToken(logits);
                };
        Sampler capped =
                Thinking.capBudget(
                        counting, TOKENIZER, 1, true, "ab", Thinking.OPEN, Thinking.CLOSE);
        assertEquals(A, capped.sampleToken(logits(A))); // thought = 1 = budget
        assertEquals(1, draws[0]);
        // the cap forces "\n\nab\n\n" + close = 7 tokens without consulting the model
        int[] forced = {NEWLINE, NEWLINE, A, B, NEWLINE, NEWLINE, CLOSE};
        for (int expected : forced) {
            assertEquals(expected, capped.sampleToken(logits(B)));
        }
        assertEquals(1, draws[0]);
        assertEquals(A, capped.sampleToken(logits(A))); // content resumes through the inner sampler
        assertEquals(2, draws[0]);
    }

    @Test
    void gemmaChannelSpanIsCappedLikeTheGenericThinkSpan() {
        Sampler capped =
                Thinking.capBudget(
                        Sampler.ARGMAX, GEMMA_TOKENIZER, 2, true, null, "<|channel>", "<channel|>");
        assertEquals(G_A, capped.sampleToken(logits(GEMMA_TOKENS, G_A)));
        assertEquals(G_A, capped.sampleToken(logits(GEMMA_TOKENS, G_A)));
        assertEquals(G_NEWLINE, capped.sampleToken(logits(GEMMA_TOKENS, G_A)));
        assertEquals(G_NEWLINE, capped.sampleToken(logits(GEMMA_TOKENS, G_A)));
        assertEquals(G_CLOSE, capped.sampleToken(logits(GEMMA_TOKENS, G_A)));
        // spent budget bans both channel markers: no reopen ping-pong
        assertEquals(G_A, capped.sampleToken(logits(GEMMA_TOKENS, G_OPEN)));
        assertEquals(G_A, capped.sampleToken(logits(GEMMA_TOKENS, G_CLOSE)));
    }

    @Test
    void gemmaChannelOpenedByTheModelIsCappedFromItsFirstToken() {
        Sampler capped =
                Thinking.capBudget(
                        Sampler.ARGMAX,
                        GEMMA_TOKENIZER,
                        1,
                        false,
                        null,
                        "<|channel>",
                        "<channel|>");
        assertEquals(
                G_OPEN, capped.sampleToken(logits(GEMMA_TOKENS, G_OPEN))); // model opens the span
        assertEquals(G_A, capped.sampleToken(logits(GEMMA_TOKENS, G_A))); // thought = 1 = budget
        assertEquals(G_NEWLINE, capped.sampleToken(logits(GEMMA_TOKENS, G_A))); // break, then close
        assertEquals(G_NEWLINE, capped.sampleToken(logits(GEMMA_TOKENS, G_A)));
        assertEquals(G_CLOSE, capped.sampleToken(logits(GEMMA_TOKENS, G_A)));
        assertEquals(G_A, capped.sampleToken(logits(GEMMA_TOKENS, G_OPEN))); // both markers banned
    }

    @Test
    void spentBudgetClosesAtAParagraphBoundaryThenBansBothMarkers() {
        Sampler capped = cap(2, true, null);
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
        Sampler capped = cap(5, true, null);
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(CLOSE)));
        assertEquals(B, capped.sampleToken(logits(B)));
        assertEquals(OPEN, capped.sampleToken(logits(OPEN)));
        assertEquals(A, capped.sampleToken(logits(A)));
    }

    @Test
    void aNullMessageIsExactlyTheBareBreak() {
        Sampler capped = cap(2, true, null);
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
    }

    @Test
    void aBlankMessageIsAlsoTheBareBreak() {
        Sampler capped = cap(2, true, "  ");
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(NEWLINE, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
    }

    @Test
    void aMessageIsForcedBetweenParagraphBreaksThenBothMarkersStayBanned() {
        Sampler capped = cap(2, true, "ab");
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
        Sampler capped = cap(1, false, "ab");
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
        Sampler capped = cap(1, true, "<nope>");
        assertEquals(A, capped.sampleToken(logits(A)));
        assertEquals(CLOSE, capped.sampleToken(logits(A)));
    }

    private static Sampler cap(int budget, boolean startInThink, String message) {
        return Thinking.capBudget(
                Sampler.ARGMAX,
                TOKENIZER,
                budget,
                startInThink,
                message,
                Thinking.OPEN,
                Thinking.CLOSE);
    }

    private static MemoryView<MemorySegment> logits(int favorite) {
        return logits(TOKENS, favorite, A);
    }

    private static MemoryView<MemorySegment> logits(String[] tokens, int favorite) {
        return logits(tokens, favorite, tokens.length - 2); // the "a" slot in every fixture
    }

    /** Logits favoring {@code favorite}, with a second-best {@code fallback} bans can drop to. */
    private static MemoryView<MemorySegment> logits(String[] tokens, int favorite, int fallback) {
        float[] values = new float[tokens.length];
        values[favorite] = 2;
        values[fallback] = Math.max(values[fallback], 1);
        return Views.fromFloatArray(MemoryAllocators.ofArena(Arena.ofAuto()), values);
    }

    private static final class FakeTokenizer implements Tokenizer {
        private final String[] tokens;
        private final Vocabulary vocabulary;

        FakeTokenizer(String[] tokens) {
            this.tokens = tokens;
            this.vocabulary = new FakeVocabulary(tokens);
        }

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
            out.put(this.tokens[tokens.intAt(tokenStartIndex)].getBytes(StandardCharsets.UTF_8));
            return 1;
        }
    }

    private static final class FakeVocabulary implements Vocabulary {
        private final String[] tokens;

        FakeVocabulary(String[] tokens) {
            this.tokens = tokens;
        }

        @Override
        public int size() {
            return tokens.length;
        }

        @Override
        public String token(int id) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            return tokens[id];
        }

        @Override
        public int id(String text) {
            for (int i = 0; i < tokens.length; i++) if (tokens[i].equals(text)) return i;
            throw new NoSuchElementException(text);
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < tokens.length;
        }

        @Override
        public boolean contains(String text) {
            for (String token : tokens) if (token.equals(text)) return true;
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
            return IntStream.range(0, tokens.length)
                    .mapToObj(i -> Map.entry(tokens[i], i))
                    .iterator();
        }
    }
}
