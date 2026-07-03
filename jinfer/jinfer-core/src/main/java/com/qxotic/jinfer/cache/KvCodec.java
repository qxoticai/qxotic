package com.qxotic.jinfer.cache;

import java.lang.foreign.MemorySegment;

/** The one model-specific seam of the prompt cache: serialize the state needed to resume decoding
 *  at a position boundary, for a span of positions, to and from an opaque memory blob. The cache
 *  never interprets the bytes; what they are is entirely the model's business — per-position KV
 *  rows for attention layers, a fixed-size checkpoint for recurrent/windowed layers (short-conv
 *  state, SSM state, a sliding window).
 *
 *  <p>Checkpoint currency: a recurrent checkpoint only exists at the position the state is
 *  actually at, so {@link #save} REQUIRES {@code state.position() == to} (implementations throw
 *  otherwise). This is why cache blocks are committed at ingestion boundaries and why a block
 *  matches completely or not at all — there is no mid-block state to resume from.
 *
 *  <p>{@link #restore} leaves the state resumable exactly at {@code to} (position advanced,
 *  recurrent state set). Transient per-batch scratch (logits, activations) is NOT restored — the
 *  caller always ingests something after a restore before reading logits. */
public interface KvCodec<S> {

    /** Blob size for the span — must be exact and deterministic (the cache allocates with it). */
    long bytes(int from, int to);

    /** Serialize the span's resume-state into {@code dst}. Requires {@code state.position() == to}. */
    void save(S state, int from, int to, MemorySegment dst);

    /** Load the span's resume-state from {@code src}; leaves the state resumable at {@code to}.
     *  Blocks are restored in chain order from position 0, so the last restore wins for
     *  whole-state checkpoints (recurrent layers). */
    void restore(S state, int from, int to, MemorySegment src);
}
