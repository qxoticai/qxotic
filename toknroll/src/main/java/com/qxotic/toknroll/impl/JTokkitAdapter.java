package com.qxotic.toknroll.impl;

import com.knuddels.jtokkit.Encodings;
import com.knuddels.jtokkit.api.Encoding;
import com.knuddels.jtokkit.api.EncodingRegistry;
import com.knuddels.jtokkit.api.GptBytePairEncodingParams;
import com.knuddels.jtokkit.api.IntArrayList;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import com.qxotic.toknroll.advanced.SymbolCodec;
import java.nio.ByteBuffer;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

class JTokkitAdapter implements Tokenizer {
    private final Vocabulary vocabulary;
    private final Encoding encoding;

    JTokkitAdapter(Vocabulary vocabulary, Encoding encoding) {
        this.vocabulary = vocabulary;
        this.encoding = encoding;
    }

    static Tokenizer create(
            String name,
            Pattern splitPattern,
            Map<String, Integer> mergeableRanks,
            Map<String, Integer> specialTokens) {
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

        return new JTokkitAdapter(vocabulary, encoding);
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

    @Override
    public Vocabulary vocabulary() {
        return this.vocabulary;
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
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
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
