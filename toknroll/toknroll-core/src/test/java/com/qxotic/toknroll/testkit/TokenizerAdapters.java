package com.qxotic.toknroll.testkit;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.util.Objects;

public final class TokenizerAdapters {
    private TokenizerAdapters() {}

    public static Tokenizer withNormalizer(Tokenizer tokenizer, Normalizer normalizer) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(normalizer, "normalizer");
        return new Tokenizer() {
            @Override
            public Vocabulary vocabulary() {
                return tokenizer.vocabulary();
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                CharSequence slice = text.subSequence(startInclusive, endExclusive);
                CharSequence transformed = normalizer.apply(slice);
                tokenizer.encodeInto(transformed, 0, transformed.length(), out);
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                CharSequence slice = text.subSequence(startInclusive, endExclusive);
                CharSequence transformed = normalizer.apply(slice);
                return tokenizer.countTokens(transformed, 0, transformed.length());
            }

            @Override
            public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
                return tokenizer.decodeBytesInto(tokens, tokenStartIndex, out);
            }
        };
    }

    public static Tokenizer withSplitter(Tokenizer tokenizer, Splitter splitter) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(splitter, "splitter");
        return new Tokenizer() {
            @Override
            public Vocabulary vocabulary() {
                return tokenizer.vocabulary();
            }

            @Override
            public void encodeInto(
                    CharSequence text,
                    int startInclusive,
                    int endExclusive,
                    IntSequence.Builder out) {
                splitter.splitAll(
                        text,
                        startInclusive,
                        endExclusive,
                        (source, chunkStart, chunkEnd) ->
                                tokenizer.encodeInto(source, chunkStart, chunkEnd, out));
            }

            @Override
            public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
                int[] total = {0};
                splitter.splitAll(
                        text,
                        startInclusive,
                        endExclusive,
                        (source, chunkStart, chunkEnd) ->
                                total[0] += tokenizer.countTokens(source, chunkStart, chunkEnd));
                return total[0];
            }

            @Override
            public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
                return tokenizer.decodeBytesInto(tokens, tokenStartIndex, out);
            }
        };
    }
}
