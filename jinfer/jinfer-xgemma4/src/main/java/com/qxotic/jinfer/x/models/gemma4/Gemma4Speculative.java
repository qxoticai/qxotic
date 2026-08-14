package com.qxotic.jinfer.x.models.gemma4;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.kernels.Convert;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jinfer.x.llm.Generator.FinishReason;
import com.qxotic.jinfer.x.llm.Generator.GenerationListener;
import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jinfer.x.llm.SpeculativeDecoding.SpeculationAudit;
import com.qxotic.jinfer.x.llm.SpeculativeDecoding.SpeculationResult;
import com.qxotic.jinfer.x.telemetry.SpeculationEvent;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.OptionalInt;
import java.util.Set;

/**
 * Self-speculative decode over the Gemma 4 MTP draft head. Per iteration: chain {@code depth}
 * greedy drafts through {@link Gemma4MtpDecoder} (no backbone forwards), then verify them in ONE
 * backbone ALL-outputs batch, SAMPLE the target distribution at each verified row, and keep the
 * draft only while it agrees (llama.cpp's sample-and-accept: distribution-correct without draft
 * probabilities); the rejected tail is rolled back with {@code resumeAt} (stale KV rows are
 * overwritten by the next append, sound under the causal mask).
 *
 * <p>Every emitted token is the backbone's own sample from a verified row - the draft only decides
 * how many backbone forwards it takes. The verify batch always starts with the exact next token
 * (known from the previous row), so each iteration commits at least one token. HONESTY: even greedy
 * output is NOT guaranteed token-identical to the plain loop - near-ties flip under batched-verify
 * numerics (the same reason prefill is not bit-equal to decode) - and sampled output is not
 * RNG-identical either (rejected drafts consume draws).
 *
 * <p>Single-threaded, like the decoder it drives. The caller has already ingested the prompt (the
 * row->token map {@code state.lastTokens} seeds the first draft).
 */
public final class Gemma4Speculative {

    private Gemma4Speculative() {}

    /**
     * Pre-allocated per-STATE speculation scratch, from the state's own arena: freed exactly when
     * the state closes, never "when GC notices" - a few MB of native memory must not depend on heap
     * pressure. One state runs one generation at a time (the claim serializes), so reuse is
     * race-free; RESET is implicit, because every buffer is written before it is read each
     * iteration (the warm-up draft re-seeds the decoder, logitsAll rewrites the verify rows).
     */
    static final class Scratch {
        final Gemma4MtpDecoder decoder;
        final MemoryView<MemorySegment> vlogits; // verify rows [depth+1, vocab], one head GEMM
        final MemoryView<MemorySegment> row; // one verify row, sampled
        final MemoryView<MemorySegment> h; // the draft chain's seed hidden
        final int depth;

        Scratch(Gemma4 model, Gemma4.State s, int depth) {
            var arena = s.specArena();
            this.decoder = model.mtpDecoder(arena);
            this.vlogits =
                    Views.allocateF32(arena, (long) (depth + 1) * model.config().vocabularySize());
            this.row = Views.allocateF32(arena, model.config().vocabularySize());
            this.h = Views.allocateF32(arena, model.config().embeddingLength());
            this.depth = depth;
        }
    }

    public static SpeculationResult generate(
            Gemma4 model,
            Gemma4.State s,
            int maxTokens,
            long timeoutNanos,
            Set<Integer> stops,
            int depth,
            Sampler sampler,
            GenerationListener listener,
            SpeculationAudit audit) {
        if (depth < 1 || depth > 8) {
            throw new IllegalArgumentException("speculation depth " + depth + " outside 1..8");
        }
        int vocab = model.config().vocabularySize();
        if (!model.speculationReady())
            throw new IllegalStateException(
                    "MTP sidecar not loaded - use loadWithMtp(gguf, mtpSidecar, arena)");
        Scratch scratch = s.specScratch;
        if (scratch == null || scratch.depth < depth) {
            scratch = new Scratch(model, s, depth);
            s.specScratch = scratch; // reused every generation, freed with the state
        }
        long startNanos = System.nanoTime();
        SpeculationResult result =
                walk(
                        model,
                        s,
                        depth,
                        maxTokens,
                        timeoutNanos,
                        stops,
                        sampler,
                        listener,
                        audit,
                        scratch,
                        vocab,
                        startNanos);
        // one emission point, resolved once per pass (telemetry must not perturb what it measures)
        SpeculationEvent event = new SpeculationEvent();
        if (event.isEnabled()) {
            event.draftedTokens = result.drafted();
            event.acceptedTokens = result.accepted();
            event.forwards = result.forwards();
            event.commit();
        }
        return result;
    }

    private static SpeculationResult walk(
            Gemma4 model,
            Gemma4.State s,
            int depth,
            int maxTokens,
            long timeoutNanos,
            Set<Integer> stops,
            Sampler sampler,
            GenerationListener listener,
            SpeculationAudit audit,
            Scratch scratch,
            int vocab,
            long startNanos) {
        long deadline =
                timeoutNanos > 0 && timeoutNanos <= Long.MAX_VALUE - startNanos
                        ? startNanos + timeoutNanos
                        : Long.MAX_VALUE;
        IntSequence.Builder emitted = IntSequence.newBuilder();
        IntSequence.Builder committed = IntSequence.newBuilder();
        int drafted = 0, acceptedTotal = 0, forwards = 0;
        int dim = model.config().embeddingLength();
        Gemma4MtpDecoder decoder = scratch.decoder;
        MemoryView<MemorySegment> vlogits = scratch.vlogits;
        MemoryView<MemorySegment> row = scratch.row;
        MemoryView<MemorySegment> h = scratch.h;

        // seed from the last ingested row: its token, its hidden, and the exact next token
        int lastRow = s.lastChunkLen - 1;
        int tLast = s.lastTokens[lastRow];
        Convert.copyF32(s.residual, (long) lastRow * dim, h, 0, dim);
        MemoryView<?> promptLogits = model.logits(s, s.outputCount() - 1);
        int next = sampler.sampleToken(promptLogits);
        int pending =
                audit != null
                        ? Ops.argmax(Views.castToSegmentBacked(promptLogits, "logits"), 0, vocab)
                        : 0; // argmax of the row that produced `next`
        int[] rowArgmax = audit != null ? new int[depth + 1] : null;

        int[] cand = new int[depth + 1];
        FinishReason finish = FinishReason.LENGTH; // budget/context, the while's exits
        while (emitted.size() < maxTokens
                && s.position() + depth + 1 <= s.contextCapacity() // a verify block must fit
                && System.nanoTime() < deadline) {
            // draft chain: warm-up pairs (h, tLast) at tLast's position; heads chain greedily
            // from `next`
            int pos = s.position() - 1;
            decoder.draft(s, h, 0, tLast, pos);
            cand[0] = next;
            int dTok = next;
            for (int i = 1; i <= depth; i++) {
                MemoryView<MemorySegment> dl =
                        decoder.draft(s, decoder.chainedHidden(), 0, dTok, pos);
                dTok = Ops.argmax(dl, 0, vocab);
                cand[i] = dTok;
            }
            drafted += depth;

            // verify all candidates in one backbone forward (ALL outputs), then walk the rows
            int basePos = s.position();
            model.ingest(s, Batch.score(cand));
            forwards++;
            model.logitsAll(s, vlogits); // every verify row's logits in ONE head GEMM
            int accepted = 0; // drafts confirmed beyond cand[0]
            int nextAfter = -1;
            while (accepted < depth) {
                // sample the TARGET at this row; the draft survives only by agreeing with it
                Convert.copyF32(vlogits, (long) accepted * vocab, row, 0, vocab);
                if (rowArgmax != null) rowArgmax[accepted] = Ops.argmax(row, 0, vocab);
                int target = sampler.sampleToken(row);
                if (cand[accepted + 1] == target) accepted++;
                else {
                    nextAfter = target;
                    break;
                }
            }
            if (nextAfter < 0) {
                Convert.copyF32(vlogits, (long) accepted * vocab, row, 0, vocab);
                if (rowArgmax != null) rowArgmax[accepted] = Ops.argmax(row, 0, vocab);
                nextAfter = sampler.sampleToken(row);
            }
            acceptedTotal += accepted;

            // extract the next iteration's seed BEFORE rollback (residual/logits are chunk
            // scratch)
            Convert.copyF32(s.residual, (long) accepted * dim, h, 0, dim);
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
            for (int i = 0; i < keep; i++) {
                committed.add(cand[i]);
                if (listener != null) listener.onIngested(cand[i]);
            }
            boolean aborted = false;
            for (int i = 0; i < keep && (stopIdx < 0 || i < stopIdx); i++) {
                emitted.add(cand[i]);
                if (audit != null) {
                    audit.onEmit(cand[i], i == 0 ? pending : rowArgmax[i - 1]);
                }
                if (listener != null && !listener.onToken(cand[i])) {
                    aborted = true; // the aborting token is the LAST emitted (Generator's law)
                    break;
                }
                if (emitted.size() >= maxTokens) {
                    return result(
                            emitted,
                            committed,
                            OptionalInt.empty(),
                            FinishReason.LENGTH,
                            startNanos,
                            drafted,
                            acceptedTotal,
                            forwards);
                }
            }
            if (audit != null) pending = rowArgmax[accepted]; // the row that produced `next`
            if (aborted) {
                return result(
                        emitted,
                        committed,
                        OptionalInt.empty(),
                        FinishReason.ABORT,
                        startNanos,
                        drafted,
                        acceptedTotal,
                        forwards);
            }
            if (stopIdx >= 0) {
                if (listener != null)
                    listener.onToken(cand[stopIdx]); // the stop is seen, not emitted
                return result(
                        emitted,
                        committed,
                        OptionalInt.of(cand[stopIdx]),
                        FinishReason.STOP,
                        startNanos,
                        drafted,
                        acceptedTotal,
                        forwards);
            }
        }
        if (System.nanoTime() >= deadline && emitted.size() < maxTokens) {
            finish = FinishReason.TIMEOUT;
        }
        return result(
                emitted,
                committed,
                OptionalInt.empty(),
                finish,
                startNanos,
                drafted,
                acceptedTotal,
                forwards);
    }

    private static SpeculationResult result(
            IntSequence.Builder emitted,
            IntSequence.Builder committed,
            OptionalInt stopToken,
            FinishReason finish,
            long startNanos,
            int drafted,
            int accepted,
            int forwards) {
        return new SpeculationResult(
                emitted.build(),
                committed.build(),
                stopToken,
                finish,
                Duration.ofNanos(System.nanoTime() - startNanos),
                drafted,
                accepted,
                forwards);
    }
}
