package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.telemetry.SpeculationEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.function.IntConsumer;

/**
 * Self-speculative decode over the Gemma 4 MTP draft head (Stage 3). Per iteration: chain {@code
 * depth} greedy drafts through {@link Gemma4MtpDecoder} (no backbone forwards), then verify them in
 * ONE backbone ALL-outputs batch, SAMPLE the target distribution at each verified row, and keep the
 * draft only while it agrees (llama.cpp's sample-and-accept: distribution-correct without draft
 * probabilities); the rejected tail is rolled back with {@link
 * com.qxotic.jinfer.RuntimeState#resumeAt} (stale KV rows are overwritten by the next append, sound
 * under the causal mask).
 *
 * <p>Every emitted token is the backbone's own sample from a verified row - the draft only decides
 * how many backbone forwards it takes. The verify batch always starts with the exact next token
 * (known from the previous row), so each iteration commits at least one token. HONESTY over the old
 * claim: even greedy output is NOT guaranteed token-identical to the plain loop - near-ties flip
 * under batched-verify numerics ({@code BatchVsStepProbe} documents it, the same reason prefill is
 * not bit-equal to decode) - and sampled output is not RNG-identical either (rejected drafts
 * consume draws).
 *
 * <p>Single-threaded, like the decoder it drives. The caller has already ingested the prompt (token
 * batches - the row->token map {@code state.lastTokens} seeds the first draft).
 */
public final class Gemma4Speculative {

    /**
     * Emitted tokens (stop-truncated, exclusive) + all committed tokens (what the KV actually
     * holds, for {@link com.qxotic.jinfer.cache.CachedSession#adopt}), and draft statistics.
     */
    public record Result(
            List<Integer> tokens,
            List<Integer> committed,
            int stopToken,
            int drafted,
            int accepted,
            int forwards) {}

    /**
     * Diagnostic hook: for every emitted token, the top-2 of the verify row that produced it — the
     * spec-side half of a near-tie analysis (the lockstep oracle supplies the other half).
     */
    public interface TopRecorder {
        void onEmit(int token, int top1, float top1Logit, int top2, float top2Logit);
    }

    private Gemma4Speculative() {}

    /**
     * Pre-allocated per-STATE speculation scratch, from the state's own arena: freed exactly when
     * the state closes, never "when GC notices" - a few MB of native memory must not depend on heap
     * pressure. One state runs one generation at a time (the model serializes), so reuse is
     * race-free; RESET is implicit, because every buffer is written before it is read each
     * iteration (the warm-up draft re-seeds the decoder, logitsAll rewrites the verify rows).
     */
    static final class Scratch {
        final Gemma4MtpDecoder decoder;
        final F32FloatTensor vlogits; // verify rows, one head GEMM
        final F32FloatTensor row; // one verify row, sampled
        final F32FloatTensor h; // the draft chain's seed hidden
        final int depth;

        Scratch(Gemma4 model, Gemma4.State s, int depth) {
            this.decoder = model.mtpDecoder(s.arena);
            this.vlogits =
                    F32FloatTensor.allocate(s.arena, depth + 1, model.config().vocabularySize());
            this.row = F32FloatTensor.allocate(s.arena, model.config().vocabularySize());
            this.h = F32FloatTensor.allocate(s.arena, model.config().embeddingLength());
            this.depth = depth;
        }
    }

    public static Result generate(
            Gemma4 model, Gemma4.State s, int maxTokens, Set<Integer> stops, int depth) {
        return generate(model, s, maxTokens, stops, depth, null);
    }

    public static Result generate(
            Gemma4 model,
            Gemma4.State s,
            int maxTokens,
            Set<Integer> stops,
            int depth,
            TopRecorder recorder) {
        return generate(model, s, maxTokens, stops, depth, Sampler.ARGMAX, null, recorder);
    }

    public static Result generate(
            Gemma4 model,
            Gemma4.State s,
            int maxTokens,
            Set<Integer> stops,
            int depth,
            Sampler sampler,
            IntConsumer onEmit,
            TopRecorder recorder) {
        int vocab = model.config().vocabularySize();
        if (!model.speculationReady())
            throw new IllegalStateException(
                    "MTP sidecar not loaded - use loadWithMtp(gguf, mtpSidecar, arena)");
        Scratch scratch = s.specScratch;
        if (scratch == null || scratch.depth < depth) {
            scratch = new Scratch(model, s, depth);
            s.specScratch = scratch; // reused every generation, freed with the state
        }
        Result result =
                generate(
                        model, s, depth, maxTokens, stops, sampler, onEmit, recorder, scratch,
                        vocab);
        // one emission point for three return sites inside the loop
        SpeculationEvent event = new SpeculationEvent();
        if (event.isEnabled()) {
            event.draftedTokens = result.drafted();
            event.acceptedTokens = result.accepted();
            event.forwards = result.forwards();
            event.commit();
        }
        return result;
    }

    private static Result generate(
            Gemma4 model,
            Gemma4.State s,
            int depth,
            int maxTokens,
            Set<Integer> stops,
            Sampler sampler,
            IntConsumer onEmit,
            TopRecorder recorder,
            Scratch scratch,
            int vocab) {
        List<Integer> emitted = new ArrayList<>();
        List<Integer> committed = new ArrayList<>();
        int drafted = 0, acceptedTotal = 0, forwards = 0;
        int dim = model.config().embeddingLength();
        Gemma4MtpDecoder decoder = scratch.decoder;
        F32FloatTensor vlogits = scratch.vlogits;
        F32FloatTensor row = scratch.row;
        F32FloatTensor h = scratch.h;

        // seed from the last ingested row: its token, its hidden, and the exact next token
        int lastRow = s.lastChunkLen - 1;
        int tLast = s.lastTokens[lastRow];
        s.residual.copyTo((long) lastRow * dim, h, 0, dim);
        FloatTensor promptLogits = model.logits(s, s.outputCount - 1);
        int next = sampler.sampleToken(promptLogits);
        Top2 pending =
                recorder != null
                        ? top2(promptLogits, 0, vocab)
                        : null; // stats of the row that produced `next`
        Top2[] rowStats = recorder != null ? new Top2[depth + 1] : null;

        int[] cand = new int[depth + 1];
        while (emitted.size() < maxTokens
                && s.position() + depth + 1 <= s.contextCapacity()) { // a verify block must fit
            // draft chain: warm-up pairs (h, tLast) at tLast's position; heads chain greedily from
            // `next`
            int pos = s.position() - 1;
            decoder.draft(s, h, 0, tLast, pos);
            cand[0] = next;
            int dTok = next;
            for (int i = 1; i <= depth; i++) {
                FloatTensor dl = decoder.draft(s, decoder.chainedHidden(), 0, dTok, pos);
                dTok = dl.argmax(0, vocab);
                cand[i] = dTok;
            }
            drafted += depth;

            // verify all candidates in one backbone forward (ALL outputs), then walk the rows
            // greedily
            int basePos = s.position();
            model.ingest(s, Batch.score(cand));
            forwards++;
            model.logitsAll(s, vlogits); // every verify row's logits in ONE head GEMM
            int accepted = 0; // drafts confirmed beyond cand[0]
            int nextAfter = -1;
            while (accepted < depth) {
                if (rowStats != null)
                    rowStats[accepted] = top2(vlogits, (long) accepted * vocab, vocab);
                // sample the TARGET at this row; the draft survives only by agreeing with it
                vlogits.copyTo((long) accepted * vocab, row, 0, vocab);
                int target = sampler.sampleToken(row);
                if (cand[accepted + 1] == target) accepted++;
                else {
                    nextAfter = target;
                    break;
                }
            }
            if (nextAfter < 0) {
                if (rowStats != null)
                    rowStats[accepted] = top2(vlogits, (long) accepted * vocab, vocab);
                vlogits.copyTo((long) accepted * vocab, row, 0, vocab);
                nextAfter = sampler.sampleToken(row);
            }
            acceptedTotal += accepted;

            // extract the next iteration's seed BEFORE rollback (residual/logits are chunk scratch)
            s.residual.copyTo((long) accepted * dim, h, 0, dim);
            tLast = cand[accepted];
            next = nextAfter;

            // stop handling first, so the KV keep-count and `committed` stay in exact lockstep:
            // keep everything up to and including a stop (it was verified), else the accepted
            // prefix
            int stopIdx = -1;
            for (int i = 0; i <= accepted && stopIdx < 0; i++) {
                if (stops.contains(cand[i])) stopIdx = i;
            }
            int keep = stopIdx >= 0 ? stopIdx + 1 : accepted + 1;
            s.resumeAt(basePos + keep); // keep cand[0..keep), drop the rest
            for (int i = 0; i < keep; i++) committed.add(cand[i]);
            for (int i = 0; i < keep && (stopIdx < 0 || i < stopIdx); i++) {
                emitted.add(cand[i]);
                if (onEmit != null) onEmit.accept(cand[i]);
                if (recorder != null) {
                    Top2 st = i == 0 ? pending : rowStats[i - 1]; // row that produced cand[i]
                    recorder.onEmit(cand[i], st.t1(), st.l1(), st.t2(), st.l2());
                }
                if (emitted.size() >= maxTokens)
                    return new Result(
                            emitted,
                            committed,
                            stopIdx >= 0 ? cand[stopIdx] : -1,
                            drafted,
                            acceptedTotal,
                            forwards);
            }
            if (recorder != null)
                pending = rowStats[accepted]; // the row that produced `next` (= nextAfter)
            if (stopIdx >= 0)
                return new Result(
                        emitted, committed, cand[stopIdx], drafted, acceptedTotal, forwards);
        }
        return new Result(emitted, committed, -1, drafted, acceptedTotal, forwards);
    }

    /** argmax + runner-up of one logits row. */
    record Top2(int t1, float l1, int t2, float l2) {}

    static Top2 top2(FloatTensor logits, long off, int vocab) {
        int i1 = -1, i2 = -1;
        float l1 = Float.NEGATIVE_INFINITY, l2 = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            float v = logits.getFloat(off + i);
            if (v > l1) {
                i2 = i1;
                l2 = l1;
                i1 = i;
                l1 = v;
            } else if (v > l2) {
                i2 = i;
                l2 = v;
            }
        }
        return new Top2(i1, l1, i2, l2);
    }
}
