package com.qxotic.toknroll.jtokkit;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.GptBytePairEncodingParams;
import com.knuddels.jtokkit.api.IntArrayList;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.StandardTokenType;
import com.qxotic.toknroll.advanced.SymbolCodec;
import com.qxotic.toknroll.impl.VocabularyImpl;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public final class JTokkitTokenizers {
    private JTokkitTokenizers() {}

    public static Tokenizer fromTiktoken(
            String name,
            Map<String, Integer> mergeableRanks,
            Pattern splitPattern,
            Map<String, Integer> specialTokens) {
        Objects.requireNonNull(name, "name");
        Objects.requireNonNull(mergeableRanks, "mergeableRanks");
        Objects.requireNonNull(splitPattern, "splitPattern");
        Objects.requireNonNull(specialTokens, "specialTokens");

        Map<byte[], Integer> rawMergeableRanks =
                mergeableRanks.entrySet().stream()
                        .collect(
                                Collectors.toMap(
                                        entry ->
                                                SymbolCodec.BYTE_LEVEL.decodeSymbols(
                                                        entry.getKey()),
                                        Map.Entry::getValue));

        GptBytePairEncodingParams params =
                new GptBytePairEncodingParams(name, splitPattern, rawMergeableRanks, specialTokens);

        EncodingRegistry encodingRegistry = Encodings.newLazyEncodingRegistry();
        encodingRegistry.registerGptBytePairEncoding(params);

        Encoding encoding = encodingRegistry.getEncoding(name).orElseThrow();
        Vocabulary vocabulary = createVocabulary(mergeableRanks, specialTokens);

        return new Adapter(vocabulary, encoding);
    }

    public static Tokenizer fromTiktoken(
            String name, Map<String, Integer> mergeableRanks, Pattern splitPattern) {
        return fromTiktoken(name, mergeableRanks, splitPattern, Collections.emptyMap());
    }

    public static Tokenizer fromTiktoken(
            String name,
            Map<String, Integer> mergeableRanks,
            String splitPattern,
            Map<String, Integer> specialTokens) {
        return fromTiktoken(
                name,
                mergeableRanks,
                Pattern.compile(Objects.requireNonNull(splitPattern, "splitPattern")),
                specialTokens);
    }

    public static Tokenizer fromTiktoken(
            String name, Map<String, Integer> mergeableRanks, String splitPattern) {
        return fromTiktoken(name, mergeableRanks, splitPattern, Collections.emptyMap());
    }

    private static Vocabulary createVocabulary(
            Map<String, Integer> mergeableRanks, Map<String, Integer> specialTokens) {
        int maxMergeableIndex =
                mergeableRanks.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        int maxSpecialIndex =
                specialTokens.values().stream().mapToInt(Integer::intValue).max().orElse(0);
        int maxIndex = Math.max(maxMergeableIndex, maxSpecialIndex);

        String[] tokens = new String[maxIndex + 1];
        for (Map.Entry<String, Integer> entry : mergeableRanks.entrySet()) {
            tokens[entry.getValue()] = entry.getKey();
        }
        return VocabularyWithSpecials.create(new VocabularyImpl(tokens), specialTokens);
    }

    private static final class Adapter implements Tokenizer {
        private final Vocabulary vocabulary;
        private final Encoding encoding;

        private Adapter(Vocabulary vocabulary, Encoding encoding) {
            this.vocabulary = vocabulary;
            this.encoding = encoding;
        }

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public int countTokens(CharSequence text) {
            return encoding.countTokens(Objects.requireNonNull(text, "text").toString());
        }

        @Override
        public void encodeInto(
                CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
            Objects.requireNonNull(text, "text");
            Objects.requireNonNull(out, "out");
            if (startInclusive < 0
                    || endExclusive < startInclusive
                    || endExclusive > text.length()) {
                throw new IndexOutOfBoundsException(
                        "Invalid range ["
                                + startInclusive
                                + ", "
                                + endExclusive
                                + ") for text length "
                                + text.length());
            }
            String slice = text.subSequence(startInclusive, endExclusive).toString();
            IntArrayList encoded = encoding.encode(slice);
            int size = encoded.size();
            out.ensureCapacity(out.size() + size);
            for (int i = 0; i < size; i++) {
                out.add(encoded.get(i));
            }
        }

        @Override
        public int countBytes(IntSequence tokens) {
            Objects.requireNonNull(tokens, "tokens");
            IntArrayList list = new IntArrayList(tokens.length());
            for (int i = 0; i < tokens.length(); i++) {
                list.add(tokens.intAt(i));
            }
            try {
                return encoding.decodeBytes(list).length;
            } catch (IllegalArgumentException e) {
                NoSuchElementException nsee = new NoSuchElementException(e.getMessage());
                nsee.initCause(e);
                throw nsee;
            }
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            Objects.requireNonNull(tokens, "tokens");
            Objects.requireNonNull(out, "out");
            int length = tokens.length();
            if (tokenStartIndex < 0 || tokenStartIndex > length) {
                throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
            }
            if (tokenStartIndex == length) {
                return 0;
            }

            int consumed = 0;
            IntArrayList oneToken = new IntArrayList(1);
            for (int i = tokenStartIndex; i < length; i++) {
                int tokenId = tokens.intAt(i);
                oneToken.clear();
                oneToken.add(tokenId);
                byte[] chunk;
                try {
                    chunk = encoding.decodeBytes(oneToken);
                } catch (IllegalArgumentException e) {
                    NoSuchElementException nsee = new NoSuchElementException(e.getMessage());
                    nsee.initCause(e);
                    throw nsee;
                }
                if (chunk.length > out.remaining()) {
                    if (consumed == 0) {
                        throw new IllegalArgumentException(
                                "Not enough output space for token at index "
                                        + i
                                        + ": need "
                                        + chunk.length
                                        + ", remaining "
                                        + out.remaining());
                    }
                    break;
                }
                out.put(chunk);
                consumed++;
            }
            return consumed;
        }

        @Override
        public String toString() {
            return "Tiktoken (JTokkit): " + encoding.getName();
        }
    }

    private static final class VocabularyWithSpecials implements Vocabulary {
        private final Vocabulary innerVocabulary;
        private final Map<String, Integer> specialToIndex;
        private final Map<Integer, String> indexToSpecial;

        private VocabularyWithSpecials(
                Vocabulary innerVocabulary, Map<String, Integer> specialToIndex) {
            this.innerVocabulary = innerVocabulary;
            this.specialToIndex = Map.copyOf(specialToIndex);
            this.indexToSpecial =
                    specialToIndex.entrySet().stream()
                            .collect(
                                    Collectors.toUnmodifiableMap(
                                            Map.Entry::getValue, Map.Entry::getKey));
        }

        private static Vocabulary create(
                Vocabulary innerVocabulary, Map<String, Integer> specialToIndex) {
            if (specialToIndex.isEmpty()) {
                return innerVocabulary;
            }
            return new VocabularyWithSpecials(innerVocabulary, specialToIndex);
        }

        @Override
        public int size() {
            return innerVocabulary.size() + specialToIndex.size();
        }

        @Override
        public String token(int tokenIndex) {
            String tokenString = indexToSpecial.get(tokenIndex);
            if (tokenString != null) {
                return tokenString;
            }
            return innerVocabulary.token(tokenIndex);
        }

        @Override
        public int id(String tokenString) {
            Integer tokenIndex = specialToIndex.get(tokenString);
            if (tokenIndex != null) {
                return tokenIndex;
            }
            return innerVocabulary.id(tokenString);
        }

        @Override
        public boolean contains(int tokenIndex) {
            return indexToSpecial.containsKey(tokenIndex) || innerVocabulary.contains(tokenIndex);
        }

        @Override
        public boolean contains(String tokenString) {
            return specialToIndex.containsKey(tokenString) || innerVocabulary.contains(tokenString);
        }

        @Override
        public boolean isTokenOfType(int tokenIndex, TokenType tokenType) {
            if (indexToSpecial.containsKey(tokenIndex)) {
                return tokenType == StandardTokenType.CONTROL;
            }
            return innerVocabulary.isTokenOfType(tokenIndex, tokenType);
        }

        @Override
        public Iterator<Map.Entry<String, Integer>> iterator() {
            return new Iterator<>() {
                private final Iterator<Map.Entry<String, Integer>> inner =
                        innerVocabulary.iterator();
                private final Iterator<Map.Entry<String, Integer>> special =
                        specialToIndex.entrySet().iterator();

                @Override
                public boolean hasNext() {
                    return inner.hasNext() || special.hasNext();
                }

                @Override
                public Map.Entry<String, Integer> next() {
                    if (inner.hasNext()) {
                        return inner.next();
                    }
                    return special.next();
                }
            };
        }
    }
}
