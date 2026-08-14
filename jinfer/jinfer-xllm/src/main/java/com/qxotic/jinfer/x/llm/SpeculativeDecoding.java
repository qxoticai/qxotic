package com.qxotic.jinfer.x.llm;

import com.qxotic.jinfer.x.boundary.RuntimeState;
import com.qxotic.jinfer.x.llm.Generator.Constraints;
import com.qxotic.jinfer.x.llm.Generator.FinishReason;
import com.qxotic.jinfer.x.llm.Generator.GenerationListener;
import com.qxotic.toknroll.IntSequence;
import java.time.Duration;
import java.util.OptionalInt;

/**
 * A model that can decode with its own draft-and-verify loop. The prompt is ingested by the
 * ordinary path first; a caller that finds this capability ready may then decode through {@link
 * #speculate} instead of {@link Generator#generate}. The draft weights may be attached or embedded;
 * that storage detail is deliberately outside this API.
 *
 * <p>Deliberately narrow: one capability check and one decode call, mirroring {@link Generator}'s
 * contract with the SAME vocabulary (Constraints, GenerationListener, FinishReason) - the general
 * multi-implementation machinery stays behind each model; this interface lets callers dispatch on
 * the capability without naming model classes.
 *
 * @param <S> the implementing model's state type
 */
public interface SpeculativeDecoding<S extends RuntimeState> {

    /** Whether this model has usable draft weights, so {@link #speculate} may be called. */
    boolean speculationReady();

    /** As the 6-arg form without an audit tap (the production call). */
    default SpeculationResult speculate(
            S state,
            Sampler sampler,
            Constraints constraints,
            int depth,
            GenerationListener listener) {
        // the 1..8 contract enforced at the door every caller uses, so a port cannot fake it;
        // the 7-arg audit door is guarded inside each port
        if (depth < 1 || depth > 8)
            throw new IllegalArgumentException("speculation depth " + depth + " outside 1..8");
        return speculate(state, sampler, constraints, depth, listener, null);
    }

    /**
     * Decodes from the state's frontier (the prompt is already ingested) until a stop token, the
     * {@code constraints} budget/deadline (checked per verify iteration), a verify block that no
     * longer fits the context, or the listener's abort. {@code sampler} samples the TARGET
     * distribution at every verified row and the draft is kept only while it agrees -
     * distribution-correct without draft probabilities (llama.cpp's sample-and-accept). {@code
     * depth} drafts per iteration, 1..8.
     *
     * <p>Listener contract, as {@link Generator}'s with one honest divergence: every emitted token
     * in order, the trailing stop token included; abort (false) ends the pass at BLOCK granularity
     * - the aborting token is the last emitted, but the state may commit past it (verified rows of
     * the current block are kept, never rolled back). {@code onIngested} fires for every committed
     * token, the stop token included (a speculative pass ingests everything it verifies).
     *
     * <p>Sampled output is not RNG-identical to the plain loop (rejected drafts consume draws), and
     * even greedy output can differ from plain greedy on near-ties: verify rows are computed in a
     * batch, and batched-vs-single-row numerics differ - the same reason prefill already is not
     * bit-equal to decode.
     */
    SpeculationResult speculate(
            S state,
            Sampler sampler,
            Constraints constraints,
            int depth,
            GenerationListener listener,
            SpeculationAudit audit);

    /**
     * What a speculative pass produced. {@code emitted} excludes the stop token; {@code committed}
     * is what the KV actually holds - it can run PAST {@code emitted} (the stop token itself was
     * verified and kept, and a budget cut or abort can land mid-block), and a caller tracking the
     * state's token stream (cache adoption) must use it, never re-derive from {@code emitted}. Both
     * are read-only.
     */
    record SpeculationResult(
            IntSequence emitted,
            IntSequence committed,
            OptionalInt stopToken,
            FinishReason finishReason,
            Duration decodeTime,
            int drafted,
            int accepted,
            int forwards) {}

    /**
     * Test-only tap certifying the defining invariant: every emitted token IS the target model's
     * own greedy choice at its position - {@code token == targetArgmax} must hold for every call,
     * in every engine configuration. Null in production; the full near-tie forensics (top-2 logits)
     * belong to diagnostic probes, not this seam.
     */
    @FunctionalInterface
    interface SpeculationAudit {
        void onEmit(int token, int targetArgmax);
    }
}
