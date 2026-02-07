package ai.qxotic.jota.ir.lir.scratch;

import ai.qxotic.jota.ir.lir.BufferRef;
import java.util.Map;

/**
 * Scratch buffer layout for a kernel. Maps intermediate buffers to their byte offsets within a
 * single contiguous scratch buffer.
 */
public record ScratchLayout(Map<BufferRef, Long> offsets, long totalByteSize) {

    /** Empty layout indicating no scratch needed. */
    public static final ScratchLayout EMPTY = new ScratchLayout(Map.of(), 0L);

    /** Default alignment for scratch allocations (cache line). */
    public static final long ALIGNMENT = 64L;

    public ScratchLayout {
        offsets = Map.copyOf(offsets);
        if (totalByteSize < 0) {
            throw new IllegalArgumentException("totalByteSize must be non-negative");
        }
    }

    /** Returns true if any scratch memory is required. */
    public boolean requiresScratch() {
        return totalByteSize > 0;
    }

    /** Gets the byte offset for a buffer, or -1 if not a scratch buffer. */
    public long getOffset(BufferRef buffer) {
        return offsets.getOrDefault(buffer, -1L);
    }

    /** Returns true if the buffer is a scratch buffer. */
    public boolean isScratchBuffer(BufferRef buffer) {
        return offsets.containsKey(buffer);
    }

    /** Returns aligned total size (rounded up to ALIGNMENT). */
    public long alignedTotalByteSize() {
        return (totalByteSize + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    }
}
