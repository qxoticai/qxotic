package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import org.junit.jupiter.api.Test;

class ToknrollApiComponentsTest {

    @Test
    void tokenizerLoadExceptionMessage() {
        TokenizerLoadException e = new TokenizerLoadException("fail");
        assertEquals("fail", e.getMessage());
        assertThrows(
                TokenizerLoadException.class,
                () -> {
                    throw new TokenizerLoadException("fail");
                });
    }

    @Test
    void tokenizerLoadExceptionMessageWithCause() {
        RuntimeException cause = new RuntimeException("root");
        TokenizerLoadException e = new TokenizerLoadException("fail", cause);
        assertEquals("fail", e.getMessage());
        assertEquals(cause, e.getCause());
        assertThrows(
                TokenizerLoadException.class,
                () -> {
                    throw new TokenizerLoadException("fail", cause);
                });
    }

    @Test
    void vocabularyDefaultFindTokenFound() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        assertEquals(Optional.of("hello"), vocab.findToken(0));
    }

    @Test
    void vocabularyDefaultFindTokenNotFound() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        assertEquals(Optional.empty(), vocab.findToken(99));
    }

    @Test
    void vocabularyDefaultFindIdFound() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        assertEquals(OptionalInt.of(0), vocab.findId("hello"));
    }

    @Test
    void vocabularyDefaultFindIdNotFound() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        assertEquals(OptionalInt.empty(), vocab.findId("unknown"));
    }

    @Test
    void vocabularyIsTokenOfTypePresent() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        assertFalse(vocab.isTokenOfType(0, StandardTokenType.NORMAL));
    }

    @Test
    void vocabularyIsTokenOfTypeAbsent() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        assertThrows(
                NoSuchElementException.class,
                () -> vocab.isTokenOfType(99, StandardTokenType.NORMAL));
    }

    @Test
    void vocabularyDefaultContainsInt() {
        Vocabulary vocab = Toknroll.vocabulary("a", "b", "c");
        assertTrue(vocab.contains(0));
        assertTrue(vocab.contains(2));
        assertFalse(vocab.contains(99));
    }

    @Test
    void vocabularyDefaultContainsString() {
        Vocabulary vocab = Toknroll.vocabulary("a", "b", "c");
        assertTrue(vocab.contains("a"));
        assertTrue(vocab.contains("c"));
        assertFalse(vocab.contains("unknown"));
    }

    @Test
    void vocabularyWithSpecialTokens() {
        Map<String, Integer> specials = new LinkedHashMap<>();
        specials.put("<|endoftext|>", 50256);
        specials.put("<|im_start|>", 100000);
        String[] tokens = {"hello", "world"};
        Vocabulary vocab = Toknroll.vocabulary(specials, tokens);
        assertEquals(4, vocab.size());
        assertEquals("hello", vocab.token(0));
        assertEquals("world", vocab.token(1));
        assertEquals("<|endoftext|>", vocab.token(50256));
        assertEquals("<|im_start|>", vocab.token(100000));
        assertEquals(0, vocab.id("hello"));
        assertEquals(50256, vocab.id("<|endoftext|>"));
    }

    @Test
    void vocabularySpecialTokenOverlapsBaseVocab() {
        Map<String, Integer> specials = Map.of("hello", 100);
        assertThrows(
                IllegalArgumentException.class,
                () -> Toknroll.vocabulary(specials, "hello", "world"));
    }

    @Test
    void vocabularySpecialTokenIdOverlapsBaseVocab() {
        String[] tokens = new String[3];
        tokens[0] = "a";
        tokens[1] = "b";
        tokens[2] = "c";
        Map<String, Integer> specials = Map.of("<|special|>", 1);
        assertThrows(IllegalArgumentException.class, () -> Toknroll.vocabulary(specials, tokens));
    }

    @Test
    void vocabularyDuplicateTokenInArray() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Toknroll.vocabulary("hello", "world", "hello"));
    }

    @Test
    void vocabularyFindTokenOfTypeNormal() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        assertTrue(vocab.contains(0));
        assertFalse(vocab.isTokenOfType(0, StandardTokenType.CONTROL));
    }

    @Test
    void specialsCompileWithOverlappingSpecials() {
        Map<String, Integer> specials = new LinkedHashMap<>();
        specials.put("<|s1|>", 100);
        specials.put("<|s2|>", 200);
        String[] tokens = new String[2];
        tokens[0] = "a";
        tokens[1] = "b";
        Vocabulary vocab = Toknroll.vocabulary(specials, tokens);
        Set<String> specialNames = new LinkedHashSet<>();
        specialNames.add("<|s1|>");
        specialNames.add("<|s2|>");
        Specials compiled = Specials.compile(vocab, specialNames);
        assertNotNull(compiled);
        assertTrue(compiled.tokens().contains("<|s1|>"));
        assertTrue(compiled.tokens().contains("<|s2|>"));
    }

    @Test
    void specialsContainsCompiledTokenStrings() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        Specials specials = Specials.compile(vocab, Set.of("hello"));
        assertTrue(specials.tokens().contains("hello"));
        assertEquals(1, specials.tokens().size());
    }

    @Test
    void specialsIsEmpty() {
        assertTrue(Specials.none().isEmpty());
        Vocabulary vocab = Toknroll.vocabulary("hello");
        Specials specials = Specials.compile(vocab, Set.of("hello"));
        assertFalse(specials.isEmpty());
    }

    @Test
    void specialsEncodeInto() {
        Vocabulary vocab = Toknroll.vocabulary("hello");
        Tokenizer tokenizer =
                new Tokenizer() {
                    @Override
                    public Vocabulary vocabulary() {
                        return vocab;
                    }

                    @Override
                    public void encodeInto(
                            CharSequence text, int start, int end, IntSequence.Builder out) {
                        out.add(0);
                    }

                    @Override
                    public int countTokens(CharSequence text, int start, int end) {
                        return 1;
                    }

                    @Override
                    public int decodeBytesInto(
                            IntSequence tokens, int idx, java.nio.ByteBuffer out) {
                        return 0;
                    }
                };
        Specials specials = Specials.compile(vocab, Set.of("hello"));
        IntSequence.Builder out = IntSequence.newBuilder();
        specials.encodeInto(tokenizer, "hello", out);
        assertEquals(1, out.size());
    }

    @Test
    void specialsEncodeWithSpecials() {
        Vocabulary vocab = Toknroll.vocabulary("hello", "world", "!");
        Tokenizer dummy =
                new Tokenizer() {
                    @Override
                    public Vocabulary vocabulary() {
                        return vocab;
                    }

                    @Override
                    public void encodeInto(
                            CharSequence text, int start, int end, IntSequence.Builder out) {
                        for (int i = start; i < end; i++) {
                            String c = text.subSequence(i, i + 1).toString();
                            if (vocab.contains(c)) out.add(vocab.id(c));
                            else out.add(0);
                        }
                    }

                    @Override
                    public int countTokens(CharSequence text, int start, int end) {
                        return end - start;
                    }

                    @Override
                    public int decodeBytesInto(
                            IntSequence tokens, int idx, java.nio.ByteBuffer out) {
                        return 0;
                    }
                };
        Specials specials = Specials.compile(vocab, Set.of("hello"));
        IntSequence result = specials.encode(dummy, "hello world !");
        assertTrue(result.length() >= 1);
    }

    @Test
    void mergeRuleValidationNegativeLeftId() {
        assertThrows(IllegalArgumentException.class, () -> Toknroll.MergeRule.of(-1, 5, 0));
    }

    @Test
    void mergeRuleValidationNegativeRightId() {
        assertThrows(IllegalArgumentException.class, () -> Toknroll.MergeRule.of(5, -1, 0));
    }

    @Test
    void mergeRuleFactoryAndAccessors() {
        Toknroll.MergeRule rule = Toknroll.MergeRule.of(10, 20, 5);
        assertEquals(10, rule.leftId());
        assertEquals(20, rule.rightId());
        assertEquals(5, rule.rank());
    }

    @Test
    void mergeRuleFactoryCreatesValidRule() {
        Toknroll.MergeRule rule = Toknroll.MergeRule.of(1, 2, 3);
        assertEquals(1, rule.leftId());
        assertEquals(2, rule.rightId());
        assertEquals(3, rule.rank());
    }

    @Test
    void vocabularyIterator() {
        Vocabulary vocab = Toknroll.vocabulary("a", "b", "c");
        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, Integer> e : vocab) {
            sb.append(e.getKey()).append("=").append(e.getValue()).append(";");
        }
        assertEquals("a=0;b=1;c=2;", sb.toString());
    }
}
