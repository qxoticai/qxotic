package com.qxotic.toknroll;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.Optional;

/**
 * Composed tokenizer pipeline that applies optional normalization and splitting around a base
 * {@link TokenizationModel}.
 *
 * <p>Implements {@link Tokenizer} but not {@link TokenizationModel}, so it cannot be nested as a
 * model inside another pipeline.
 *
 * <p>Optional mutation phases (for example, Unicode normalization) can be configured in the
 * pipeline and are applied before chunk splitting and model encoding.
 */
public final class TokenizationPipeline implements Tokenizer {

    private final TokenizationModel model;
    private final Normalizer normalizer;
    private final Splitter splitter;
    private final boolean hasNormalizer;
    private final boolean hasSplitter;

    private TokenizationPipeline(Builder builder) {
        this.model = Objects.requireNonNull(builder.model, "model is required");
        this.normalizer = builder.normalizer;
        this.splitter = builder.splitter;
        this.hasNormalizer = normalizer != null;
        this.hasSplitter = splitter != null;
    }

    public static Builder builder(TokenizationModel model) {
        return new Builder(Objects.requireNonNull(model, "model"));
    }

    public TokenizationModel model() {
        return model;
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
            model.encodeInto(text, startInclusive, endExclusive, out);
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
            model.encodeInto(normalizedText, out);
            return;
        }
        splitter.splitAll(
                normalizedText,
                0,
                normalizedText.length(),
                (source, chunkStart, chunkEnd) ->
                        model.encodeInto(source, chunkStart, chunkEnd, out));
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        Objects.requireNonNull(text, "text");
        if (!hasNormalizer && !hasSplitter) {
            return model.countTokens(text, startInclusive, endExclusive);
        }

        CharSequence current =
                (startInclusive == 0 && endExclusive == text.length())
                        ? text
                        : text.subSequence(startInclusive, endExclusive);
        if (hasNormalizer) {
            current = normalizer.apply(current);
        }
        if (!hasSplitter) {
            return model.countTokens(current, 0, current.length());
        }
        int[] total = {0};
        splitter.splitAll(
                current,
                0,
                current.length(),
                (source, chunkStart, chunkEnd) ->
                        total[0] += model.countTokens(source, chunkStart, chunkEnd));
        return total[0];
    }

    @Override
    public float expectedTokensPerChar() {
        return model.expectedTokensPerChar();
    }

    @Override
    public String decode(IntSequence tokens) {
        return model.decode(tokens);
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        return model.decodeBytes(tokens);
    }

    @Override
    public int countBytes(IntSequence tokens) {
        return model.countBytes(tokens);
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        return model.decodeBytesInto(tokens, tokenStartIndex, out);
    }

    @Override
    public Vocabulary vocabulary() {
        return model.vocabulary();
    }

    @Override
    public String toString() {
        return "Pipeline[norm=" + hasNormalizer + ", split=" + hasSplitter + ", " + model + "]";
    }

    /** Builder for {@link TokenizationPipeline}. */
    public static final class Builder {
        private final TokenizationModel model;
        private Normalizer normalizer;
        private Splitter splitter;

        private Builder(TokenizationModel model) {
            this.model = model;
        }

        public TokenizationModel model() {
            return model;
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
            return new TokenizationPipeline(this);
        }
    }
}
