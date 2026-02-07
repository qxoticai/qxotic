package ai.qxotic.jota.ir.lir.scratch;

import ai.qxotic.jota.ir.lir.BufferRef;
import java.util.Objects;

/**
 * Represents the liveness interval of a buffer during kernel execution. Used for memory reuse
 * optimization - non-overlapping intervals can share memory.
 *
 * <p>The interval uses statement indices as "time" - [firstUse, lastUse] inclusive.
 */
public record LivenessInterval(BufferRef buffer, int firstUse, int lastUse) {

    public LivenessInterval {
        Objects.requireNonNull(buffer, "buffer cannot be null");
        if (firstUse < 0) {
            throw new IllegalArgumentException("firstUse must be non-negative");
        }
        if (lastUse < firstUse) {
            throw new IllegalArgumentException("lastUse must be >= firstUse");
        }
    }

    /** Returns true if this interval overlaps with another. */
    public boolean overlaps(LivenessInterval other) {
        return !(this.lastUse < other.firstUse || other.lastUse < this.firstUse);
    }

    /** Returns the duration (number of statements this buffer is live). */
    public int duration() {
        return lastUse - firstUse + 1;
    }
}
