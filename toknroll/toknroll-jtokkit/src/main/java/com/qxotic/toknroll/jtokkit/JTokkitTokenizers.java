package com.qxotic.toknroll.jtokkit;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.GptBytePairEncodingParams;
import com.knuddels.jtokkit.api.IntArrayList;
import com.qxotic.toknroll.ByteLevel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.util.Collections;
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
                                        entry -> ByteLevel.decode(entry.getKey()),
                                        Map.Entry::getValue));

        GptBytePairEncodingParams params =
                new GptBytePairEncodingParams(name, splitPattern, rawMergeableRanks, specialTokens);

        EncodingRegistry encodingRegistry = Encodings.newLazyEncodingRegistry();
        encodingRegistry.registerGptBytePairEncoding(params);

        Encoding encoding = encodingRegistry.getEncoding(name).orElseThrow();
        Vocabulary vocabulary =
                Tokenizers.tikToken(mergeableRanks, specialTokens, Splitter.regex(splitPattern))
                        .vocabulary();

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
        public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
            Objects.requireNonNull(text, "text");
            return encoding.countTokens(text.subSequence(startInclusive, endExclusive).toString());
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
}
