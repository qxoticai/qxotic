package com.qxotic.model.llm.llama;

import com.qxotic.span.FloatSpan;
import java.lang.foreign.MemorySegment;

/**
 * An abstract implementation of {@link FloatSpan} that wraps a {@link MemorySegment}. This class
 * provides basic functionality for handling memory segments and calculating their size.
 *
 * <p>The class serves as a base for specific memory segment span implementations that work with
 * floating-point data stored in native memory.
 *
 * @see FloatSpan
 * @see MemorySegment
 */
public abstract class MemorySegmentSpan implements FloatSpan {
    /** The underlying memory segment that contains the data. */
    final MemorySegment memorySegment;

    /**
     * Constructs a new MemorySegmentSpan with the specified memory segment.
     *
     * @param memorySegment the memory segment to wrap
     */
    MemorySegmentSpan(MemorySegment memorySegment) {
        this.memorySegment = memorySegment;
    }

    /**
     * Returns the total size of the memory segment in bytes.
     *
     * @return the size of the memory segment in bytes
     */
    @Override
    public long sizeInBytes() {
        return this.memorySegment.byteSize();
    }
}
