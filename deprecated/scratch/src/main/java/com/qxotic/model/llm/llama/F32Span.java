package com.qxotic.model.llm.llama;

import com.qxotic.span.FloatSpan;
import java.lang.foreign.MemorySegment;

/**
 * A final implementation of {@link MemorySegmentSpan} that handles 32-bit floating-point data. This
 * class provides operations for working with contiguous blocks of float values stored in native
 * memory segments.
 *
 * <p>The class ensures that the underlying memory segment's size is always a multiple of {@link
 * Float#BYTES}, guaranteeing proper alignment for float values.
 *
 * @see MemorySegmentSpan
 * @see FloatSpan
 */
public final class F32Span extends MemorySegmentSpan {
    /**
     * Constructs a new F32Span with the specified memory segment.
     *
     * @param memorySegment the memory segment containing 32-bit float values
     */
    public F32Span(MemorySegment memorySegment) {
        super(memorySegment);
        assert memorySegment.byteSize() % Float.BYTES == 0;
    }

    /**
     * Returns the number of float values in this span.
     *
     * @return the number of float values (not bytes) in the span
     */
    @Override
    public long size() {
        return this.memorySegment.byteSize() / Float.BYTES;
    }

    /**
     * Creates a new span that represents a subsection of this span. If the requested slice matches
     * the entire span, returns this instance instead of creating a new object.
     *
     * @param spanIndex the starting index of the slice (in float values, not bytes)
     * @param spanLength the length of the slice (in float values, not bytes)
     * @return a new F32Span representing the requested slice, or this instance if the slice
     *     encompasses the entire span
     * @throws IllegalArgumentException if the spanIndex is negative or if the slice would extend
     *     beyond the span's bounds
     */
    @Override
    public FloatSpan slice(long spanIndex, long spanLength) {
        if (spanIndex == 0 && spanLength == size()) {
            return this;
        }
        long offsetInBytes = spanIndex * Float.BYTES;
        long lengthInBytes = spanLength * Float.BYTES;
        return new F32Span(this.memorySegment.asSlice(offsetInBytes, lengthInBytes));
    }
}
