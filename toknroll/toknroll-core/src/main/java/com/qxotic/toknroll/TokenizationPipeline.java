package com.qxotic.toknroll;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.Optional;

/**
 * Composed tokenizer pipeline that applies optional normalization and splitting around a base
 * {@link TokenizationModel}.
 */
public final class TokenizationPipeline implements Tokenizer {

    private final TokenizationModel model;
    private final Normalizer normalizer;
    private final Splitter splitter;
    private final boolean hasNormalizer;
    private final boolean hasSplitter;

    /**
     * Creates a pipeline. Pass {@code null} for {@code normalizer} or {@code splitter} to skip that
     * stage. The model is required.
     */
    TokenizationPipeline(Normalizer normalizer, Splitter splitter, TokenizationModel model) {
        this.model = Objects.requireNonNull(model, "model is required");
        this.normalizer = normalizer;
        this.splitter = splitter;
        this.hasNormalizer = normalizer != null;
        this.hasSplitter = splitter != null;
    }

    // ---- introspection ----

    public TokenizationModel model() {
        return model;
    }

    public Optional<Normalizer> normalizer() {
        return Optional.ofNullable(normalizer);
    }

    public Optional<Splitter> splitter() {
        return Optional.ofNullable(splitter);
    }

    // ---- encode / countTokens ----

    @Override
    public void encodeInto(
            CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");
        validateRange(text, startInclusive, endExclusive);

        if (!hasNormalizer && !hasSplitter) {
            model.encodeInto(text, startInclusive, endExclusive, out);
            return;
        }
        encodeNormalizedInto(normalizeSlice(text, startInclusive, endExclusive), out);
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
        validateRange(text, startInclusive, endExclusive);
        if (!hasNormalizer && !hasSplitter) {
            return model.countTokens(text, startInclusive, endExclusive);
        }

        CharSequence current = normalizeSlice(text, startInclusive, endExclusive);
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

    // ---- decode (delegated to model) ----

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

    // ---- helpers ----

    private CharSequence normalizeSlice(CharSequence text, int startInclusive, int endExclusive) {
        CharSequence current =
                (startInclusive == 0 && endExclusive == text.length())
                        ? text
                        : text.subSequence(startInclusive, endExclusive);
        return hasNormalizer ? normalizer.apply(current) : current;
    }

    private static void validateRange(CharSequence text, int startInclusive, int endExclusive) {
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for text length "
                            + text.length());
        }
    }

    @Override
    public String toString() {
        return "Pipeline[norm=" + hasNormalizer + ", split=" + hasSplitter + ", " + model + "]";
    }
}
