package com.qxotic.toknroll.advanced;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.Optional;

/**
 * Composed tokenizer pipeline that applies optional normalization and splitting around a base
 * tokenizer.
 *
 * <p>By default, core tokenizer usage targets round-trip integrity for ordinary text. Adding a
 * {@link Normalizer} is an explicit opt-in that may be lossy and is generally discouraged unless
 * mutation is intentionally required.
 */
public final class TokenizationPipeline implements Tokenizer {

    private final Tokenizer baseTokenizer;
    private final Normalizer normalizer;
    private final Splitter splitter;
    private final boolean hasNormalizer;
    private final boolean hasSplitter;

    private TokenizationPipeline(Builder builder) {
        this.baseTokenizer =
                Objects.requireNonNull(builder.baseTokenizer, "baseTokenizer is required");
        this.normalizer = builder.normalizer;
        this.splitter = builder.splitter;
        this.hasNormalizer = normalizer != null;
        this.hasSplitter = splitter != null;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static Builder builder(Tokenizer baseTokenizer) {
        return new Builder().baseTokenizer(baseTokenizer);
    }

    public Tokenizer baseTokenizer() {
        return baseTokenizer;
    }

    public Optional<Normalizer> normalizer() {
        return Optional.ofNullable(normalizer);
    }

    public Optional<Splitter> splitter() {
        return Optional.ofNullable(splitter);
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

        if (!hasNormalizer && !hasSplitter) {
            baseTokenizer.encodeInto(text, startInclusive, endExclusive, out);
            return;
        }

        CharSequence current =
                (startInclusive == 0 && endExclusive == text.length())
                        ? text
                        : text.subSequence(startInclusive, endExclusive);
        if (hasNormalizer) {
            current = normalizer.apply(current);
        }
        encodeNormalizedInto(current, out);
    }

    private void encodeNormalizedInto(CharSequence normalizedText, IntSequence.Builder out) {
        if (!hasSplitter) {
            baseTokenizer.encodeInto(normalizedText, out);
            return;
        }
        splitter.splitAll(
                normalizedText,
                0,
                normalizedText.length(),
                (source, chunkStart, chunkEnd) ->
                        baseTokenizer.encodeInto(source, chunkStart, chunkEnd, out));
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        Objects.requireNonNull(text, "text");
        if (!hasNormalizer && !hasSplitter) {
            return baseTokenizer.countTokens(text, startInclusive, endExclusive);
        }

        CharSequence current =
                (startInclusive == 0 && endExclusive == text.length())
                        ? text
                        : text.subSequence(startInclusive, endExclusive);
        if (hasNormalizer) {
            current = normalizer.apply(current);
        }
        if (!hasSplitter) {
            return baseTokenizer.countTokens(current, 0, current.length());
        }
        int[] total = {0};
        final CharSequence normalized = current;
        splitter.splitAll(
                normalized,
                0,
                normalized.length(),
                (source, chunkStart, chunkEnd) ->
                        total[0] += baseTokenizer.countTokens(source, chunkStart, chunkEnd));
        return total[0];
    }

    @Override
    public String decode(IntSequence tokens) {
        return baseTokenizer.decode(tokens);
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        return baseTokenizer.decodeBytes(tokens);
    }

    @Override
    public int countBytes(IntSequence tokens) {
        return baseTokenizer.countBytes(tokens);
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        return baseTokenizer.decodeBytesInto(tokens, tokenStartIndex, out);
    }

    @Override
    public Vocabulary vocabulary() {
        return baseTokenizer.vocabulary();
    }

    @Override
    public String toString() {
        return "Pipeline[norm="
                + hasNormalizer
                + ", split="
                + hasSplitter
                + ", "
                + baseTokenizer
                + "]";
    }

    /** Builder for {@link TokenizationPipeline}. */
    public static final class Builder {
        private Tokenizer baseTokenizer;
        private Normalizer normalizer;
        private Splitter splitter;

        public Optional<Tokenizer> baseTokenizer() {
            return Optional.ofNullable(baseTokenizer);
        }

        public Builder baseTokenizer(Tokenizer tokenizer) {
            this.baseTokenizer = Objects.requireNonNull(tokenizer, "baseTokenizer");
            return this;
        }

        public Optional<Normalizer> normalizer() {
            return Optional.ofNullable(normalizer);
        }

        /** Sets or replaces the normalizer used by this builder. */
        public Builder normalizer(Normalizer normalizer) {
            this.normalizer = Objects.requireNonNull(normalizer, "normalizer");
            return this;
        }

        public Optional<Splitter> splitter() {
            return Optional.ofNullable(splitter);
        }

        /** Sets or replaces the splitter used by this builder. */
        public Builder splitter(Splitter splitter) {
            this.splitter = Objects.requireNonNull(splitter, "splitter");
            return this;
        }

        public TokenizationPipeline build() {
            if (baseTokenizer == null) {
                throw new IllegalStateException(
                        "TokenizationPipeline.Builder is missing required component:"
                                + " baseTokenizer");
            }
            return new TokenizationPipeline(this);
        }
    }
}
