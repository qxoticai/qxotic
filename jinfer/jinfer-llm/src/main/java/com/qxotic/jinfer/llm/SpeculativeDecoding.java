package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.RuntimeState;
import com.qxotic.toknroll.IntSequence;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * A model that can decode with its OWN draft-and-verify loop - self-speculation via an attached
 * draft head (gemma4's MTP sidecar). The prompt is ingested by the ordinary path first; a caller
 * that finds this capability ready may then decode through {@link #speculate} instead of the
 * one-token-per-forward loop.
 *
 * <p>Deliberately narrow: one capability check and one decode call, mirroring the decode half of
 * {@link Generator#generate}'s contract. The general multi-implementation seam (llama.cpp's
 * begin/process/draft/accept with a {@code --spec-type} selector) waits for a second draft family;
 * this interface is what lets a caller that must not name port classes (the CLI holds ports at
 * runtime scope) dispatch on the capability instead.
 *
 * @param <S> the implementing model's state type
 */
public interface SpeculativeDecoding<S extends RuntimeState> {

    /** Whether a draft head is attached, so {@link #speculate} may be called. */
    boolean speculationReady();

    /**
     * Decodes from the state's frontier (the prompt is already ingested) until a stop token, the
     * {@code maxTokens} budget (negative = as much as the context allows), or a verify block that
     * no longer fits the context. {@code sampler} samples the TARGET distribution at every verified
     * row and the draft is kept only while it agrees - distribution-correct without draft
     * probabilities (llama.cpp's sample-and-accept). {@code onToken} fires per EMITTED token, for
     * streaming.
     *
     * <p>Sampled output is not RNG-identical to the plain loop (rejected drafts consume draws), and
     * even greedy output can differ from plain greedy on near-ties: verify rows are computed in a
     * batch, and batched-vs-single-row numerics differ - the same reason prefill already is not
     * bit-equal to decode.
     */
    Speculation speculate(
            S state,
            int maxTokens,
            Set<Integer> stops,
            Sampler sampler,
            int depth,
            IntConsumer onToken);

    /**
     * What a speculative pass produced. {@code emitted} excludes the stop token ({@code stopToken}
     * -1 when none); {@code committed} is what the KV actually holds - it can run PAST {@code
     * emitted} (the stop token itself was verified and kept, and a budget cut can land mid-block),
     * and a caller tracking the state's token stream must use it, never re-derive from {@code
     * emitted}.
     */
    record Speculation(
            IntSequence emitted,
            IntSequence committed,
            int stopToken,
            int drafted,
            int accepted,
            int forwards) {}
}
