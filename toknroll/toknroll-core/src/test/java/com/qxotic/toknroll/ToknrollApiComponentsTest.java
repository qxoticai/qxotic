package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.OptionalInt;
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
    void mergeRuleValidationNegativeLeftId() {
        assertThrows(IllegalArgumentException.class, () -> MergeRule.of(-1, 5, 0));
    }

    @Test
    void mergeRuleValidationNegativeRightId() {
        assertThrows(IllegalArgumentException.class, () -> MergeRule.of(5, -1, 0));
    }

    @Test
    void mergeRuleFactoryAndAccessors() {
        MergeRule rule = MergeRule.of(10, 20, 5);
        assertEquals(10, rule.leftId());
        assertEquals(20, rule.rightId());
        assertEquals(5, rule.rank());
    }

    @Test
    void mergeRuleFactoryCreatesValidRule() {
        MergeRule rule = MergeRule.of(1, 2, 3);
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
