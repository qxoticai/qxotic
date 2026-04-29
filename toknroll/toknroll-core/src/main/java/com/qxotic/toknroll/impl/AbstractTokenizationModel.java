package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Vocabulary;
import java.util.Objects;

/**
 * Abstract base implementation of the {@link TokenizationModel} interface.
 *
 * <p>Provides vocabulary management and a default encoding pipeline. Subclasses implement the core
 * chunk-level encoding logic via {@link #encodeImplInto(CharSequence, IntSequence.Builder)}.
 * Splitting is handled externally by {@link TokenizationPipeline}.
 *
 * <p>Implementations need to provide:
 *
 * <ul>
 *   <li>The core encoding logic via {@link #encodeImpl(CharSequence)}
 *   <li>The decoding logic via {@link #decodeBytesInto(IntSequence, int, java.nio.ByteBuffer)}
 * </ul>
 */
abstract class AbstractTokenizationModel implements TokenizationModel {

    protected final Vocabulary vocabulary;

    private final float expectedTokensPerChar;

    protected AbstractTokenizationModel(Vocabulary vocabulary, float expectedTokensPerChar) {
        this.vocabulary = Objects.requireNonNull(vocabulary, "vocabulary");
        this.expectedTokensPerChar = expectedTokensPerChar;
    }

    @Override
    public float expectedTokensPerChar() {
        return expectedTokensPerChar;
    }

    @Override
    public Vocabulary vocabulary() {
        return this.vocabulary;
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
        encodeImplInto(text.subSequence(startInclusive, endExclusive), out);
    }

    protected abstract IntSequence encodeImpl(CharSequence text);

    /** Appends encoded token IDs into {@code out}. Override to avoid intermediate allocations. */
    protected void encodeImplInto(CharSequence text, IntSequence.Builder out) {
        out.addAll(encodeImpl(text));
    }

    /**
     * Estimates the initial token capacity for a chunk of text.
     *
     * <p>Uses a conservative heuristic (char count × tokens-per-char ratio × 1.15 safety margin +
     * 8) with overflow protection.
     */
    protected int estimateInitialTokenCapacity(int charCount) {
        float ratio = Math.max(1.0e-6f, expectedTokensPerChar());
        double estimated = Math.ceil(charCount * (double) ratio * 1.15d) + 8d;
        if (estimated > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Estimated token capacity exceeds int range: " + estimated);
        }
        int predicted = (int) estimated;
        return Math.max(8, predicted);
    }

    protected static boolean heapLess(int rankA, long nodeA, int rankB, long nodeB) {
        if (rankA != rankB) {
            return rankA < rankB;
        }
        int leftA = (int) (nodeA >>> 32);
        int leftB = (int) (nodeB >>> 32);
        return leftA < leftB;
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        Objects.requireNonNull(text, "text");
        IntSequence.Builder out =
                IntSequence.newBuilder(estimateInitialTokenCapacity(endExclusive - startInclusive));
        encodeInto(text, startInclusive, endExclusive, out);
        return out.size();
    }
}
