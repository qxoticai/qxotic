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
 * the residue is duplicated per block by design - keep it small (KBs) for fine-grained blocking, or
 * declare {@link #coarseBlocks()} when it is genuinely large (MB-scale recurrences: define-only
 * blocks, tail-snapshot serving; see the flag's decision guide).
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

    /**
     * Coarse blocking hint: a codec whose residue is LARGE (MBs of true recurrence, duplicated per
     * block by design) asks drivers to commit cached prompts as ONE block per prompt - one residue
     * per prompt, prefix sharing limited to whole prompts - instead of one block per batch
     * boundary. Default false: small residue, fine-grained blocking.
     *
     * <p>HOW TO DECIDE FOR A NEW MODEL - the flag trades granularity against footprint, and the
     * deciding number is {@code blockBytes(0)}, the residue duplicated into every block:
     *
     * <ul>
     *   <li>{@code false} (fine): serving WRITES blocks - turn-aligned prompt blocks plus one reply
     *       block per generation, each carrying a full residue copy. Echoes and forks resume at any
     *       turn boundary. Cost per served turn: new rows (~10-35KB/token) + one residue. Sound
     *       while a long conversation's residues stay a small slice of the block budget (2GB
     *       default): LFM2.5's ~340KB conv residue costs ~8MB per 25-turn conversation - noise - so
     *       it is fine-grained.
     *   <li>{@code true} (coarse): serving never writes (a residue per served turn would consume
     *       the budget in tens of requests: NemotronH ~50MB and Qwen3.5 ~66MB residues = ~1.5GB per
     *       25-turn conversation). Blocks come from {@link PromptCache#define define()} alone;
     *       served follow-ups reuse the tail via the facade's per-session residue snapshot instead
     *       (rewind to the last prompt boundary - append-only, no fork points; forks re-prefill).
     * </ul>
     *
     * <p>The shipped anchors sit 147x apart (340KB fine, 50MB coarse), so the middle ground is
     * unmeasured territory: a residue in the low MBs is defensible either way - work the arithmetic
     * above against expected conversation shapes and the budget, and prefer measuring (the block
     * layer's {@code bytes} in {@code /props} over a representative session) to guessing. Both
     * behaviors are correct; only footprint and resume granularity move.
     */
    default boolean coarseBlocks() {
        return false;
    }
}
