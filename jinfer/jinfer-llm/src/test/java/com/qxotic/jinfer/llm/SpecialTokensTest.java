package com.qxotic.jinfer.llm;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * The trust semantics of {@link SpecialTokens} over a minimal fake vocabulary: lookups resolve
 * SPECIALS ONLY (a plain vocab string can never alias a scaffold id), require() fails naming the
 * missing token, and the prefix-conflict pruning keeps the specials encoder compilable.
 */
public final class SpecialTokensTest {

    /** tokens 0-3: specials; token 4: "&lt;think&gt;" as a PLAIN vocab string (the trust trap). */
    static final List<String> TOKENS =
            List.of("<|im_start|>", "<|im_end|>", "<param", "<parameters>", "<think>", "hello");

    static final int PLAIN_THINK = 4;
    static final Tokenizer TOK = new FakeTokenizer();

    static void check(boolean ok, String what) {
        Assertions.assertTrue(ok, what);
    }

    @Test
    void specialsTrustSemantics() {
        // find: specials only
        check(SpecialTokens.find(TOK, "<|im_start|>").orElse(-1) == 0, "find <|im_start|> -> 0");
        check(SpecialTokens.find(TOK, "<|im_end|>").orElse(-1) == 1, "find <|im_end|> -> 1");
        check(
                SpecialTokens.find(TOK, "<think>").isEmpty(),
                "plain-vocab <think> must NOT resolve (trust property)");
        check(SpecialTokens.find(TOK, "absent").isEmpty(), "absent name -> empty");

        // require: throws naming the token; present-but-plain is as absent as absent
        check(SpecialTokens.require(TOK, "<|im_start|>") == 0, "require <|im_start|> -> 0");
        try {
            SpecialTokens.require(TOK, "<eot>");
            check(false, "require missing must throw");
        } catch (IllegalArgumentException e) {
            check(e.getMessage().contains("<eot>"), "require names the missing token");
        }
        try {
            SpecialTokens.require(TOK, "<think>");
            check(false, "require plain-vocab string must throw");
        } catch (IllegalArgumentException expected) {
        }

        // isSpecial: vocabulary typing, no throw out of range
        check(SpecialTokens.isSpecial(TOK, 0), "id 0 is special");
        check(!SpecialTokens.isSpecial(TOK, PLAIN_THINK), "plain <think> id is not special");
        check(!SpecialTokens.isSpecial(TOK, 999), "out-of-vocab id is not special (no throw)");

        // encoder: "<param" strictly prefixes "<parameters>" - compile must survive (pruning),
        // and specials strings map to ids around plainly-encoded text
        IntSequence ids = SpecialTokens.encode(TOK, "<|im_start|>hello<|im_end|>");
        check(ids.toList().equals(List.of(0, 5, 1)), "specials-aware encode maps markers: " + ids);
    }

    private static final class FakeTokenizer implements Tokenizer {
        private final Vocabulary vocab = new FakeVocabulary();

        @Override
        public Vocabulary vocabulary() {
            return vocab;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            // whole-word lookup of the plain tokens only - enough for the encoder test
            String s = text.subSequence(start, end).toString();
            if (s.isEmpty()) return;
            int id = TOKENS.indexOf(s);
            if (id < 0) throw new IllegalArgumentException("fake vocab lacks: " + s);
            out.add(id);
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return end > start ? 1 : 0;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex == tokens.length()) return 0;
            out.put(TOKENS.get(tokens.intAt(tokenStartIndex)).getBytes(StandardCharsets.UTF_8));
            return 1;
        }
    }

    private static final class FakeVocabulary implements Vocabulary {
        @Override
        public int size() {
            return TOKENS.size();
        }

        @Override
        public String token(int id) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            return TOKENS.get(id);
        }

        @Override
        public int id(String text) {
            int id = TOKENS.indexOf(text);
            if (id < 0) throw new NoSuchElementException(text);
            return id;
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < TOKENS.size();
        }

        @Override
        public boolean contains(String text) {
            return TOKENS.contains(text);
        }

        @Override
        public boolean isTokenOfType(int id, TokenType type) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            boolean special = id < PLAIN_THINK;
            if (type == StandardTokenType.NORMAL) return !special;
            if (type == StandardTokenType.CONTROL) return special;
            return false;
        }

        @Override
        public java.util.Iterator<Map.Entry<String, Integer>> iterator() {
            return java.util.stream.IntStream.range(0, TOKENS.size())
                    .<Map.Entry<String, Integer>>mapToObj(i -> Map.entry(TOKENS.get(i), i))
                    .iterator();
        }
    }
}
