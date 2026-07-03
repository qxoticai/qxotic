package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.RuntimeState;

import java.lang.foreign.MemorySegment;

/** The one model-specific seam of the prompt cache: a pure copier between the state needed to
 *  resume decoding at a position boundary and an opaque memory blob. The cache never interprets
 *  the bytes; what they are is the model's business, split into two orthogonal sections:
 *
 *  <ul>
 *  <li><b>Rows</b> — per-position state (attention K/V rows). Every block stores its span's rows;
 *      size is proportional to the span.</li>
 *  <li><b>Checkpoint</b> — a whole-state snapshot of position-independent size (short-conv
 *      history, SSM state, a sliding window). A checkpoint only exists at the position the state
 *      is actually at, so the cache writes one only when {@code state.position() == to} — which
 *      is why blocks match completely or not at all. Checkpoints are SPARSE: the cache decides
 *      which blocks carry one (multi-token blocks, and every {@code jinfer.checkpointStride}
 *      positions along single-token runs), so decode blocks cost rows only and a resume lands at
 *      the deepest checkpointed block, the caller re-ingesting the short tail past it. Models
 *      with no recurrent/windowed state leave {@link #checkpointBytes} at 0 and every block is a
 *      resume point.</li>
 *  </ul>
 *
 *  <p>A checkpointed block's blob is {@code [rows][checkpoint]}. Lifecycle stays out of the
 *  codec: restore methods copy bytes into the state's tensors, and the cache calls
 *  {@link RuntimeState#resumeAt} once after the whole chain is applied. Rows restore in chain
 *  order from position 0; the checkpoint is applied once, from the resume block. */
public interface KvCodec<S extends RuntimeState> {

    /** Rows-section size for a span of {@code positions} — exact and deterministic. */
    long rowBytes(int positions);

    /** Checkpoint-section size — fixed, position-independent; 0 = no recurrent/windowed state
     *  (every block is then a resume point at zero extra cost). */
    default long checkpointBytes() {
        return 0;
    }

    /** Serialize the span's per-position rows into {@code dst}. */
    void saveRows(S state, int from, int to, MemorySegment dst);

    /** Copy the span's per-position rows from {@code src} into the state's tensors. */
    void restoreRows(S state, int from, int to, MemorySegment src);

    /** Serialize the whole-state checkpoint as of {@code to} (== {@code state.position()}). */
    default void saveCheckpoint(S state, int to, MemorySegment dst) {
    }

    /** Copy the checkpoint for position {@code to} from {@code src} into the state. */
    default void restoreCheckpoint(S state, int to, MemorySegment src) {
    }
}
