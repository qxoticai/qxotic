package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.RuntimeState;
import java.lang.foreign.MemorySegment;

/**
 * The one model-specific seam of the prompt cache: a pure copier between the state needed to resume
 * decoding at a position boundary and an opaque memory blob. The cache never interprets the bytes.
 *
 * <p>Every block is SELF-CONTAINED: {@code save} writes the span's per-position rows (attention K/V
 * - windowed layers store their rows through their ring slots, so the window rebuilds from rows
 * alone) followed by a small fixed-size RESIDUE trailer for genuinely recurrent state (short-conv
 * FIR history). Restoring a chain of blocks in order from position 0 leaves the state live at the
 * final {@code to} - every block boundary is a resume point, no placement policy, no walk-back.
 *
 * <p>Two contracts: {@code save} is only valid while {@code state.position() == to} (the residue
 * and the live window only exist at that instant - why blocks match completely or not at all), and
 * the residue must be SMALL (KBs, duplicated per block by design). A model whose mutable state is
 * neither per-position rows nor a small residue (large SSM recurrences) does not offer block
 * caching - live-session reuse still applies.
 *
 * <p>Lifecycle stays out of the codec: {@code restore} copies bytes into the state's tensors, and
 * the cache calls {@link RuntimeState#resumeAt} once after the whole chain is applied.
 */
public interface StateCodec<S extends RuntimeState> {

    /** Block-blob size for a span of {@code positions}: rows plus the fixed residue trailer. */
    long blockBytes(int positions);

    /** Serialize the span {@code [from,to)} - rows then residue; {@code state.position() == to}. */
    void save(S state, int from, int to, MemorySegment dst);

    /** Copy the span {@code [from,to)} - rows then residue - from {@code src} into the state. */
    void restore(S state, int from, int to, MemorySegment src);
}
