// Extension points the engine's prefill/decode loop calls as the token frontier advances.
// Model-agnostic: hooks see only the token stream and stream positions, never model state. The
// server's prompt cache implements these to resume from cached positions, align chunks to page
// boundaries, and commit pages as the frontier passes them.
package com.qxotic.jinfer;

/** Hooks for {@link Engine}'s generation loop; positions are stream indexes (relative to the
 *  generation's start position). The default no-op set is {@link #NONE}. */
interface GenerationHooks {
    GenerationHooks NONE = new GenerationHooks() {};

    /** Called once before ingestion with the effective stream (length = prefill tokens);
     *  returns how many leading positions are already in the state (cache restore). */
    default int resumePosition(int[] stream, int prefillLength) { return 0; }

    /** May shrink (never grow) the next chunk so it ends on a boundary the hook cares about. */
    default int clampChunk(int position, int chunkLength) { return chunkLength; }

    /** The frontier advanced: stream[0, position) is now ingested. */
    default void afterIngest(int[] stream, int position) {}

    /** The prompt is fully ingested and its logits are computed (time-to-first-token boundary). */
    default void afterPrefill() {}
}
