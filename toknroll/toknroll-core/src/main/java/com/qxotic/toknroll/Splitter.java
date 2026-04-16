package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.RegexSplitter;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Range-based text partitioner.
 *
 * <p>Contract for {@link #splitAll(CharSequence, int, int, SplitConsumer)}:
 *
 * <ul>
 *   <li>Must emit ranges over the provided {@code text}; no text transformation.
 *   <li>Each emitted range is half-open: {@code [startInclusive, endExclusive)}.
 *   <li>All emitted ranges must stay within the requested parent range.
 *   <li>Emitted ranges must form a full partition of the requested range:
 *       <ul>
 *         <li>No holes (gaps).
 *         <li>No overlaps.
 *         <li>No duplicates / out-of-order fragments.
 *         <li>Coverage must be exact from parent start to parent end.
 *       </ul>
 *   <li>Empty child ranges are never allowed.
 *   <li>For empty parent ranges, splitters must emit no ranges (rather than an empty child range).
 * </ul>
 *
 * <p>Hot paths should prefer {@code splitAll(...)} over eager list materialization.
 */
@FunctionalInterface
public interface Splitter {

    /**
     * Emits split ranges over {@code text} to {@code consumer}.
     *
     * <p>The consumer always receives the same source instance passed as {@code text}; splitters
     * should only partition, never transform. Emitted ranges must be a partition of {@code
     * [startInclusive, endExclusive)} with no holes and no overlaps. Ranges are half-open: {@code
     * [startInclusive, endExclusive)}.
     *
     * <p>Empty emitted ranges are never valid, including when the parent range is empty. For empty
     * parent ranges, emit no ranges.
     */
    void splitAll(CharSequence text, int startInclusive, int endExclusive, SplitConsumer consumer);

    default void splitAll(CharSequence text, SplitConsumer consumer) {
        Objects.requireNonNull(text, "text");
        splitAll(text, 0, text.length(), consumer);
    }

    /** Strict default: single chunk, no rewrite. */
    static Splitter identity() {
        return (text, startInclusive, endExclusive, consumer) -> {
            Objects.requireNonNull(text, "text");
            Objects.requireNonNull(consumer, "consumer");
            validateRange(startInclusive, endExclusive, text.length());
            if (startInclusive == endExclusive) {
                return;
            }
            consumer.accept(text, startInclusive, endExclusive);
        };
    }

    static Splitter regex(Pattern pattern) {
        Objects.requireNonNull(pattern, "pattern");
        return RegexSplitter.create(pattern);
    }

    static Splitter regex(String regexPattern) {
        Objects.requireNonNull(regexPattern, "regexPattern");
        return regex(Pattern.compile(regexPattern));
    }

    static Splitter sequence(Splitter... splitters) {
        Objects.requireNonNull(splitters, "splitters");
        for (Splitter splitter : splitters) {
            Objects.requireNonNull(splitter, "splitter");
        }
        if (splitters.length == 0) {
            return identity();
        }
        if (splitters.length == 1) {
            return splitters[0];
        }
        return (text, startInclusive, endExclusive, consumer) -> {
            Objects.requireNonNull(text, "text");
            Objects.requireNonNull(consumer, "consumer");
            validateRange(startInclusive, endExclusive, text.length());
            applySequence(splitters, 0, text, startInclusive, endExclusive, consumer);
        };
    }

    private static void applySequence(
            Splitter[] splitters,
            int stage,
            CharSequence source,
            int startInclusive,
            int endExclusive,
            SplitConsumer consumer) {
        if (stage == splitters.length) {
            consumer.accept(source, startInclusive, endExclusive);
            return;
        }
        int[] cursor = {startInclusive};
        splitters[stage].splitAll(
                source,
                startInclusive,
                endExclusive,
                (stageSource, chunkStart, chunkEnd) -> {
                    assert stageSource == source
                            : "Splitter.sequence requires range-preserving splitters"
                                    + " that emit the provided source";
                    assert chunkStart >= startInclusive && chunkEnd <= endExclusive
                            : "Child splitter emitted range ["
                                    + chunkStart
                                    + ", "
                                    + chunkEnd
                                    + ") outside parent range ["
                                    + startInclusive
                                    + ", "
                                    + endExclusive
                                    + ")";
                    validateRange(chunkStart, chunkEnd, source.length());
                    assert chunkStart == cursor[0]
                            : (chunkStart < cursor[0]
                                    ? "Child splitter emitted overlapping or out-of-order range ["
                                            + chunkStart
                                            + ", "
                                            + chunkEnd
                                            + ") after cursor "
                                            + cursor[0]
                                    : "Child splitter left a hole before range ["
                                            + chunkStart
                                            + ", "
                                            + chunkEnd
                                            + ") with cursor "
                                            + cursor[0]);
                    assert chunkEnd > chunkStart
                            : "Child splitter emitted empty range ["
                                    + chunkStart
                                    + ", "
                                    + chunkEnd
                                    + ") inside parent range ["
                                    + startInclusive
                                    + ", "
                                    + endExclusive
                                    + ")";
                    cursor[0] = chunkEnd;
                    applySequence(splitters, stage + 1, source, chunkStart, chunkEnd, consumer);
                });
        assert cursor[0] == endExclusive
                : "Child splitter did not cover full parent range ["
                        + startInclusive
                        + ", "
                        + endExclusive
                        + "), stopped at "
                        + cursor[0];
    }

    private static void validateRange(int startInclusive, int endExclusive, int length) {
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > length) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for source length "
                            + length);
        }
    }

    @FunctionalInterface
    interface SplitConsumer {
        /** Receives one emitted split range over {@code source}. Empty ranges are never valid. */
        void accept(CharSequence source, int startInclusive, int endExclusive);
    }
}
