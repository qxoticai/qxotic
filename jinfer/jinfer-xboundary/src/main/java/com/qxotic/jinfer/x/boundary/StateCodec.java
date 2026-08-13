package com.qxotic.jinfer.x.boundary;

import java.lang.foreign.MemorySegment;

/**
 * Copies resumable model history between a runtime state and opaque checkpoints.
 *
 * <p>A checkpoint captures everything needed to resume decoding at a position boundary: the span's
 * per-position rows ({@code [from,to)} of attention K/V) plus, for recurrent or hybrid models, a
 * fixed-size endpoint snapshot (convolution history, recurrent matrices) that only exists while the
 * live state is positioned at {@code to} - so a checkpoint must be saved at that instant. Restore
 * checkpoint chains in ascending position order, then call {@link RuntimeState#resumeAt(int)} with
 * the final endpoint. The prompt cache stores checkpoints as its blocks.
 *
 * <p>{@code checkpointBytes(0)} is the endpoint snapshot alone: a zero-span save is a pure endpoint
 * checkpoint, and the number is the per-checkpoint duplication cost that decides {@link
 * #coarseCheckpoints()}.
 *
 * <p>The blob format is model-private. A destination or source must be exactly {@link
 * #checkpointBytes(int) checkpointBytes(to - from)} bytes.
 */
public interface StateCodec<S extends RuntimeState> {

    /** Encoded bytes for a checkpoint spanning {@code positions}, endpoint snapshot included. */
    long checkpointBytes(int positions);

    /** Saves a checkpoint of {@code [from,to)} from a live state positioned at {@code to}. */
    void saveCheckpoint(S state, int from, int to, MemorySegment destination);

    /** Restores a checkpoint of {@code [from,to)} without changing the state's cursor. */
    void restoreCheckpoint(S state, int from, int to, MemorySegment source);

    /**
     * Whether cache users should avoid fine-grained checkpoints because each one duplicates a large
     * endpoint snapshot. Decide from {@code checkpointBytes(0)} against the block budget: a KB
     * scale snapshot (LFM2's ~340KB convolution state) costs noise per turn, so serving writes
     * checkpoints freely; an MB scale one (NemotronH ~50MB, Qwen3.5 ~66MB) would consume the budget
     * in tens of requests, so checkpoints come from define alone and served follow-ups reuse the
     * tail via zero-span endpoint snapshots.
     */
    default boolean coarseCheckpoints() {
        return false;
    }

    /**
     * The one validation of the checkpoint transfer contract, shared by every codec: the span must
     * lie inside the state's context, the blob must be exactly the span's encoding ({@code
     * codec.checkpointBytes(to - from)}), and a SAVE must happen with the state positioned at the
     * span's end - an endpoint snapshot only exists at that instant, and the single law is simpler
     * than one per codec flavor.
     */
    static <S extends RuntimeState> void requireCheckpoint(
            StateCodec<S> codec, S state, int from, int to, MemorySegment blob, boolean save) {
        if (from < 0 || from > to || to > state.contextCapacity()) {
            throw new IllegalArgumentException(
                    "state span ["
                            + from
                            + ","
                            + to
                            + ") outside context capacity "
                            + state.contextCapacity());
        }
        long expectedBytes = codec.checkpointBytes(to - from);
        if (blob.byteSize() != expectedBytes) {
            throw new IllegalArgumentException(
                    "checkpoint bytes " + blob.byteSize() + " != " + expectedBytes);
        }
        if (save && state.position() != to) {
            throw new IllegalStateException(
                    "state position "
                            + state.position()
                            + " != checkpoint endpoint "
                            + to
                            + "; a checkpoint must be captured at its endpoint");
        }
    }
}
