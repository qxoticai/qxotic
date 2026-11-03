package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.ImplAccessor;
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
    private final boolean scanned; // the tiktoken scanner and its model, which encode in one pass

    /**
     * Creates a pipeline. Pass {@code null} for {@code normalizer} or {@code splitter} to skip that
     * stage. The model is required.
     */
    TokenizationPipeline(Normalizer normalizer, Splitter splitter, TokenizationModel model) {
        this.model = Objects.requireNonNull(model, "model is required");
        this.normalizer = normalizer;
        this.splitter = splitter;
        // the identity changes nothing, and skipping it keeps UTF-8 on the bytes path
        this.hasNormalizer = normalizer != null && normalizer != Normalizer.identity();
        this.hasSplitter = splitter != null;
        this.scanned = hasSplitter && ImplAccessor.isScanned(model, splitter);
    }

    // ---- introspection ----

    /**
     * Returns the underlying {@link TokenizationModel}.
     *
     * @return the tokenization model
     */
    public TokenizationModel model() {
        return model;
    }

    /**
     * Returns the normalizer, if one is configured.
     *
     * @return configured normalizer, or empty if none
     */
    public Optional<Normalizer> normalizer() {
        return Optional.ofNullable(normalizer);
    }

    /**
     * Returns the splitter, if one is configured.
     *
     * @return configured splitter, or empty if none
     */
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
        if (hasNormalizer) {
            CharSequence normalized = normalize(text, startInclusive, endExclusive);
            encodeNormalizedInto(normalized, 0, normalized.length(), out);
        } else {
            encodeNormalizedInto(text, startInclusive, endExclusive, out);
        }
    }

    @Override
    public void encodeUtf8Into(
            byte[] utf8, int startInclusive, int endExclusive, IntSequence.Builder out) {
        Objects.requireNonNull(utf8, "utf8");
        Objects.requireNonNull(out, "out");
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > utf8.length) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for length "
                            + utf8.length);
        }
        // Bytes go straight to the model when nothing normalizes them, a Unicode form included
        // when they are in it already; else a normalizer works on chars.
        boolean asIs =
                !hasNormalizer
                        || normalizer instanceof UnicodeNormalizer
                                && ((UnicodeNormalizer) normalizer)
                                        .leaves(utf8, startInclusive, endExclusive);
        if (scanned && asIs) {
            ImplAccessor.encodeScannedUtf8(
                    model, splitter, utf8, startInclusive, endExclusive, out);
            return;
        }
        Tokenizer.super.encodeUtf8Into(utf8, startInclusive, endExclusive, out);
    }

    /** Encodes {@code text[start, end)}, normalized already or with nothing to normalize. */
    private void encodeNormalizedInto(
            CharSequence text, int start, int end, IntSequence.Builder out) {
        if (!hasSplitter) {
            model.encodeInto(text, start, end, out);
        } else if (scanned) {
            ImplAccessor.encodeScanned(model, splitter, text, start, end, out);
        } else {
            splitter.splitAll(
                    text,
                    start,
                    end,
                    (source, chunkStart, chunkEnd) ->
                            model.encodeInto(source, chunkStart, chunkEnd, out));
        }
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        Objects.requireNonNull(text, "text");
        validateRange(text, startInclusive, endExclusive);
        CharSequence source = text;
        int start = startInclusive;
        int end = endExclusive;
        if (hasNormalizer) {
            source = normalize(text, startInclusive, endExclusive);
            start = 0;
            end = source.length();
        }
        if (!hasSplitter) {
            return model.countTokens(source, start, end);
        }
        if (scanned) {
            IntSequence.Builder counted =
                    IntSequence.newBuilder(
                            Tokenizer.estimateTokenCapacity(end - start, expectedTokensPerChar()));
            ImplAccessor.encodeScanned(model, splitter, source, start, end, counted);
            return counted.size();
        }
        int[] total = {0};
        splitter.splitAll(
                source,
                start,
                end,
                (chunk, chunkStart, chunkEnd) ->
                        total[0] += model.countTokens(chunk, chunkStart, chunkEnd));
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

    private CharSequence normalize(CharSequence text, int startInclusive, int endExclusive) {
        CharSequence slice =
                (startInclusive == 0 && endExclusive == text.length())
                        ? text
                        : text.subSequence(startInclusive, endExclusive);
        return normalizer.apply(slice);
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
