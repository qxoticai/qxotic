package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.RuntimeState;

import java.lang.foreign.MemorySegment;

/** The one model-specific seam of the prompt cache: a pure copier between the state needed to
 *  resume decoding at a position boundary and an opaque memory blob. The cache never interprets
 *  the bytes; what they are is the model's business — per-position KV rows for attention layers,
 *  a fixed-size checkpoint for recurrent/windowed layers (short-conv state, SSM state, a sliding
 *  window).
 *
 *  <p>Checkpoint currency: a recurrent checkpoint only exists at the position the state is
 *  actually at, so the cache invokes {@link #save} only when {@code state.position() == to}
 *  (enforced centrally by {@link PromptCache}). This is why blocks match completely or not at all
 *  — there is no mid-block state to resume from.
 *
 *  <p>Lifecycle stays out of the codec: {@link #restore} copies bytes into the state's tensors,
 *  and the cache calls {@link RuntimeState#resumeAt} once after the whole chain is applied.
 *  Blocks restore in chain order from position 0, so the deepest block's whole-state checkpoint
 *  wins. */
public interface KvCodec<S extends RuntimeState> {

    /** Blob size for a span of {@code positions} — exact and deterministic (row sizes are
     *  position-independent; checkpoints are fixed-size). */
    long bytes(int positions);

    /** Serialize the span's resume-state into {@code dst}. */
    void save(S state, int from, int to, MemorySegment dst);

    /** Copy the span's resume-state from {@code src} into the state's tensors. */
    void restore(S state, int from, int to, MemorySegment src);
}
