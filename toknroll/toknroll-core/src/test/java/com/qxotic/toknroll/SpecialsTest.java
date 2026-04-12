package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import org.junit.jupiter.api.Test;

class SpecialsTest {

    @Test
    void noneDelegatesToTokenizerEncode() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("hello", 1));

        IntSequence expected = tokenizer.encode("hello");
        IntSequence actual = Specials.none().encode(tokenizer, "hello");

        assertArrayEquals(expected.toArray(), actual.toArray());
        assertTrue(Specials.none().tokens().isEmpty());
    }

    @Test
    void compileWithEmptySetReturnsNone() {
        Vocabulary vocabulary = vocabulary(Map.of("x", 1));

        Specials specials = Specials.compile(vocabulary, Set.of());

        assertSame(Specials.none(), specials);
        assertTrue(specials.tokens().isEmpty());
    }

    @Test
    void compileRejectsNullArguments() {
        Vocabulary vocabulary = vocabulary(Map.of("x", 1));

        assertThrows(NullPointerException.class, () -> Specials.compile(null, Set.of("x")));
        assertThrows(NullPointerException.class, () -> Specials.compile(vocabulary, null));
    }

    @Test
    void compileRejectsUnknownSpecials() {
        Vocabulary vocabulary = vocabulary(Map.of("known", 1));

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Specials.compile(vocabulary, Set.of("<|missing|>")));

        assertTrue(error.getMessage().contains("not present in vocabulary"));
    }

    @Test
    void compileRejectsEmptySpecials() {
        Vocabulary vocabulary = vocabulary(Map.of("a", 1));
        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Specials.compile(vocabulary, Set.of("")));
        assertTrue(error.getMessage().contains("cannot be empty"));
    }

    @Test
    void compileRejectsNullSpecials() {
        Vocabulary vocabulary = vocabulary(Map.of("a", 1));
        LinkedHashSet<String> withNull = new LinkedHashSet<>();
        withNull.add(null);

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Specials.compile(vocabulary, withNull));

        assertTrue(error.getMessage().contains("cannot be null"));
    }

    @Test
    void compileRejectsPrefixConflicts() {
        Vocabulary vocabulary = vocabulary(Map.of("<|a|>", 10, "<|a|>x", 11));

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Specials.compile(vocabulary, Set.of("<|a|>", "<|a|>x")));

        assertTrue(error.getMessage().contains("prefix conflict"));
    }

    @Test
    void compileRejectsPrefixConflictsRegardlessOfInputOrder() {
        Vocabulary vocabulary = vocabulary(Map.of("<|a|>", 10, "<|a|>x", 11, "<|z|>", 12));

        IllegalArgumentException error =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Specials.compile(
                                        vocabulary,
                                        new LinkedHashSet<>(List.of("<|z|>", "<|a|>x", "<|a|>"))));

        assertTrue(error.getMessage().contains("prefix conflict"));
    }

    @Test
    void tokensViewContainsConfiguredTokensAndIsImmutable() {
        Vocabulary vocabulary = vocabulary(Map.of("<|b|>", 2, "<|a|>", 1));
        Specials specials =
                Specials.compile(vocabulary, new LinkedHashSet<>(List.of("<|b|>", "<|a|>")));

        assertTrue(specials.tokens().contains("<|a|>"));
        assertTrue(specials.tokens().contains("<|b|>"));
        assertEquals(2, specials.tokens().size());
        assertThrows(UnsupportedOperationException.class, () -> specials.tokens().add("<|c|>"));
    }

    @Test
    void encodeInjectsSpecialIdsAndEncodesGaps() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("hi ", 10, " there", 11, "<|special|>", 42));
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|special|>"));

        IntSequence encoded = specials.encode(tokenizer, "hi <|special|> there");

        assertArrayEquals(new int[] {10, 42, 11}, encoded.toArray());
    }

    @Test
    void encodeEmptyTextProducesEmptySequence() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("<|s|>", 9));
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|s|>"));

        assertTrue(specials.encode(tokenizer, "").isEmpty());
    }

    @Test
    void encodeSingleSpecialProducesSingleSpecialId() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("<|s|>", 9));
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|s|>"));

        assertArrayEquals(new int[] {9}, specials.encode(tokenizer, "<|s|>").toArray());
    }

    @Test
    void encodeHandlesSpecialsAtBothTextBoundaries() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("<|a|>", 1, "mid", 2, "<|b|>", 3));
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|a|>", "<|b|>"));

        IntSequence encoded = specials.encode(tokenizer, "<|a|>mid<|b|>");

        assertArrayEquals(new int[] {1, 2, 3}, encoded.toArray());
    }

    @Test
    void encodeHandlesAdjacentAndRepeatedSpecials() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("<|a|>", 1, "<|b|>", 2, "x", 3));
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|a|>", "<|b|>"));

        IntSequence encoded = specials.encode(tokenizer, "<|a|><|b|><|a|>x");

        assertArrayEquals(new int[] {1, 2, 1, 3}, encoded.toArray());
    }

    @Test
    void encodeWithSpecialsRoundTripsToOriginalText() {
        Tokenizer tokenizer =
                new MiniTokenizer(Map.of("hello ", 1, " world", 2, "<|special|>", 42));
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|special|>"));

        String text = "hello <|special|> world";
        IntSequence encoded = specials.encode(tokenizer, text);

        assertEquals(text, tokenizer.decode(encoded));
    }

    @Test
    void encodeTreatsUnconfiguredSpecialLikeNormalText() {
        Tokenizer tokenizer =
                new MiniTokenizer(
                        Map.of("a<|other|>b", 55, "<|special|>", 99, "a", 1, "b", 2), true);
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|special|>"));

        assertArrayEquals(new int[] {55}, specials.encode(tokenizer, "A<|other|>B").toArray());
    }

    @Test
    void encodeSupportsRegexMetaCharactersInSpecials() {
        Tokenizer tokenizer =
                new MiniTokenizer(
                        Map.of(
                                "x", 1,
                                "y", 2,
                                "(.*)?+[]{}|^$\\", 77));
        String meta = "(.*)?+[]{}|^$\\";
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of(meta));

        IntSequence encoded = specials.encode(tokenizer, "x" + meta + "y");

        assertArrayEquals(new int[] {1, 77, 2}, encoded.toArray());
    }

    @Test
    void specialsAreSplitBeforeTokenizerPreprocessing() {
        Tokenizer tokenizer =
                new MiniTokenizer(Map.of("a", 1, "b", 2, "a<|up|>b", 7, "<|UP|>", 99), true);
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|UP|>"));

        assertArrayEquals(new int[] {7}, tokenizer.encode("A<|UP|>B").toArray());
        assertArrayEquals(new int[] {1, 99, 2}, specials.encode(tokenizer, "A<|UP|>B").toArray());
    }

    @Test
    void encodeIntoAppendsToProvidedBuilder() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("x", 1, "<|s|>", 9));
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|s|>"));

        IntSequence.Builder out = IntSequence.newBuilder();
        out.add(1000);
        specials.encodeInto(tokenizer, "x<|s|>x", out);

        assertArrayEquals(new int[] {1000, 1, 9, 1}, out.build().toArray());
    }

    @Test
    void noneEncodeIntoAppendsLikeTokenizerEncodeInto() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("x", 1));
        IntSequence.Builder out = IntSequence.newBuilder();
        out.add(7);

        Specials.none().encodeInto(tokenizer, "x", out);

        assertArrayEquals(new int[] {7, 1}, out.build().toArray());
    }

    @Test
    void encodeAndEncodeIntoValidateNulls() {
        Tokenizer tokenizer = new MiniTokenizer(Map.of("x", 1, "<|s|>", 9));
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|s|>"));
        IntSequence.Builder out = IntSequence.newBuilder();

        assertThrows(NullPointerException.class, () -> specials.encode(null, "x"));
        assertThrows(NullPointerException.class, () -> specials.encode(tokenizer, null));
        assertThrows(NullPointerException.class, () -> specials.encodeInto(null, "x", out));
        assertThrows(NullPointerException.class, () -> specials.encodeInto(tokenizer, null, out));
        assertThrows(NullPointerException.class, () -> specials.encodeInto(tokenizer, "x", null));
    }

    private static Vocabulary vocabulary(Map<String, Integer> tokenToId) {
        return new MapVocabulary(tokenToId);
    }

    private static final class MiniTokenizer implements Tokenizer {
        private final MapVocabulary vocabulary;
        private final boolean lowercaseBeforeEncode;

        MiniTokenizer(Map<String, Integer> tokenToId) {
            this(tokenToId, false);
        }

        MiniTokenizer(Map<String, Integer> tokenToId, boolean lowercaseBeforeEncode) {
            this.vocabulary = new MapVocabulary(tokenToId);
            this.lowercaseBeforeEncode = lowercaseBeforeEncode;
        }

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(
                CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
            String slice = text.subSequence(startInclusive, endExclusive).toString();
            if (lowercaseBeforeEncode) {
                slice = slice.toLowerCase(Locale.ROOT);
            }
            Integer whole = vocabulary.tokenToId.get(slice);
            if (whole != null) {
                out.add(whole);
                return;
            }
            for (int i = 0; i < slice.length(); i++) {
                String unit = String.valueOf(slice.charAt(i));
                Integer id = vocabulary.tokenToId.get(unit);
                if (id == null) {
                    throw new NoSuchElementException(unit);
                }
                out.add(id);
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
            String token = vocabulary.token(tokens.intAt(tokenStartIndex));
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

        MapVocabulary(Map<String, Integer> tokenToId) {
            this.tokenToId = new LinkedHashMap<>(tokenToId);
            this.idToToken = new LinkedHashMap<>();
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
        public Iterator<Map.Entry<String, Integer>> iterator() {
            return tokenToId.entrySet().iterator();
        }
    }
}
