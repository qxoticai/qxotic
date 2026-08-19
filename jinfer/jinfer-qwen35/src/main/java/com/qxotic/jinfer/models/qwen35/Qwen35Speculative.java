package com.qxotic.jinfer.models.qwen35;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.llm.Generator.FinishReason;
import com.qxotic.jinfer.llm.Generator.GenerationListener;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.SpeculativeDecoding.SpeculationAudit;
import com.qxotic.jinfer.llm.SpeculativeDecoding.SpeculationResult;
import com.qxotic.jinfer.telemetry.SpeculationEvent;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import java.lang.foreign.MemorySegment;
import java.time.Duration;
import java.util.OptionalInt;
import java.util.Set;

/** Draft-and-verify loop for Qwen3.5's embedded single-layer MTP head. */
final class Qwen35Speculative {

    private Qwen35Speculative() {}

    static final class Scratch {
        final MemoryView<MemorySegment> verificationLogits;
        final Qwen35CheckpointCodec codec;
        final MemorySegment checkpoint;
        final int depth;

        Scratch(Qwen35 model, Qwen35.State state, int depth) {
            var arena = state.specArena();
            int vocab = model.configuration().vocabularySize();
            verificationLogits = Views.allocateF32(arena, (long) (depth + 1) * vocab);
            codec = new Qwen35CheckpointCodec(model.configuration());
            checkpoint = arena.allocateMemory(codec.byteSize(0), 64).base();
            this.depth = depth;
        }
    }

    static SpeculationResult generate(
            Qwen35 model,
            Qwen35.State state,
            int maxTokens,
            long timeoutNanos,
            Set<Integer> stops,
            int depth,
            Sampler sampler,
            GenerationListener listener,
            SpeculationAudit audit) {
        if (depth < 1 || depth > 8)
            throw new IllegalArgumentException("speculation depth " + depth + " outside 1..8");
        if (!model.speculationReady())
            throw new IllegalStateException("Qwen3.5 model has no embedded MTP layer");
        depth = Math.min(depth, state.batchCapacity() - 1);
        Scratch scratch = state.specScratch;
        if (scratch == null || scratch.depth < depth) {
            scratch = new Scratch(model, state, depth);
            state.specScratch = scratch;
        }
        long started = System.nanoTime();
        SpeculationResult result =
                walk(
                        model,
                        state,
                        maxTokens,
                        timeoutNanos,
                        stops,
                        depth,
                        sampler,
                        listener,
                        audit,
                        scratch,
                        started);
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
            Qwen35 model,
            Qwen35.State state,
            int maxTokens,
            long timeoutNanos,
            Set<Integer> stops,
            int depth,
            Sampler sampler,
            GenerationListener listener,
            SpeculationAudit audit,
            Scratch scratch,
            long started) {
        long deadline =
                timeoutNanos > 0 && timeoutNanos <= Long.MAX_VALUE - started
                        ? started + timeoutNanos
                        : Long.MAX_VALUE;
        int vocab = model.configuration().vocabularySize();
        Qwen35CheckpointCodec codec = scratch.codec;
        IntSequence.Builder emitted = IntSequence.newBuilder();
        IntSequence.Builder committed = IntSequence.newBuilder();
        int drafted = 0, acceptedTotal = 0, forwards = 0;
        if (maxTokens == 0)
            return result(
                    emitted,
                    committed,
                    OptionalInt.empty(),
                    FinishReason.LENGTH,
                    started,
                    drafted,
                    acceptedTotal,
                    forwards);
        int[][] blocks = new int[depth + 1][];
        for (int drafts = 0; drafts <= depth; drafts++) blocks[drafts] = new int[drafts + 1];
        int[][] prefixes = new int[depth + 1][];
        for (int length = 1; length <= depth; length++) prefixes[length] = new int[length];

        MemoryView<?> promptLogits = model.logits(state, state.outputCount() - 1);
        int next = sample(sampler, promptLogits, vocab);
        int pending =
                audit == null
                        ? 0
                        : Ops.argmax(Views.castToSegmentBacked(promptLogits, "logits"), 0, vocab);
        int[] targetArgmax = audit == null ? null : new int[depth + 1];

        FinishReason finish = FinishReason.LENGTH;
        while (emitted.size() < maxTokens
                && state.position() < state.contextCapacity()
                && System.nanoTime() < deadline) {
            int drafts =
                    Math.min(
                            depth,
                            Math.min(
                                    state.contextCapacity() - state.position() - 1,
                                    maxTokens - emitted.size() - 1));
            int[] candidates = blocks[drafts];
            int base = state.position();
            if (drafts > 0) codec.capture(state, base, base, scratch.checkpoint);

            candidates[0] = next;
            if (drafts > 0) model.draft(state, drafts, candidates);
            drafted += drafts;

            model.ingest(state, Batch.score(candidates));
            forwards++;
            model.logitsAll(state, scratch.verificationLogits);

            int accepted = 0;
            int nextAfter = -1;
            while (accepted < drafts) {
                MemoryView<MemorySegment> logits = row(scratch.verificationLogits, accepted, vocab);
                if (targetArgmax != null) targetArgmax[accepted] = Ops.argmax(logits, 0, vocab);
                int target = sample(sampler, logits, vocab);
                if (candidates[accepted + 1] == target) accepted++;
                else {
                    nextAfter = target;
                    break;
                }
            }
            if (nextAfter < 0) {
                MemoryView<MemorySegment> logits = row(scratch.verificationLogits, accepted, vocab);
                if (targetArgmax != null) targetArgmax[accepted] = Ops.argmax(logits, 0, vocab);
                nextAfter = sample(sampler, logits, vocab);
            }
            acceptedTotal += accepted;

            int stop = -1;
            for (int i = 0; i <= accepted && stop < 0; i++)
                if (stops.contains(candidates[i])) stop = i;
            int keep = stop >= 0 ? stop + 1 : accepted + 1;

            if (keep < drafts + 1) {
                codec.restore(state, base, base, scratch.checkpoint);
                state.resumeAt(base);
                int[] prefix = prefixes[keep];
                System.arraycopy(candidates, 0, prefix, 0, keep);
                model.ingest(state, Batch.score(prefix));
                forwards++;
            }

            for (int i = 0; i < keep; i++) {
                committed.add(candidates[i]);
                if (listener != null) listener.onIngested(candidates[i]);
            }

            boolean aborted = false;
            for (int i = 0; i < keep && (stop < 0 || i < stop); i++) {
                emitted.add(candidates[i]);
                if (audit != null)
                    audit.onEmit(candidates[i], i == 0 ? pending : targetArgmax[i - 1]);
                if (listener != null && !listener.onToken(candidates[i])) {
                    aborted = true;
                    break;
                }
                if (emitted.size() >= maxTokens)
                    return result(
                            emitted,
                            committed,
                            OptionalInt.empty(),
                            FinishReason.LENGTH,
                            started,
                            drafted,
                            acceptedTotal,
                            forwards);
            }
            if (audit != null) pending = targetArgmax[accepted];
            next = nextAfter;

            if (aborted)
                return result(
                        emitted,
                        committed,
                        OptionalInt.empty(),
                        FinishReason.ABORT,
                        started,
                        drafted,
                        acceptedTotal,
                        forwards);
            if (stop >= 0) {
                if (listener != null) listener.onToken(candidates[stop]);
                return result(
                        emitted,
                        committed,
                        OptionalInt.of(candidates[stop]),
                        FinishReason.STOP,
                        started,
                        drafted,
                        acceptedTotal,
                        forwards);
            }
        }
        if (System.nanoTime() >= deadline && emitted.size() < maxTokens)
            finish = FinishReason.TIMEOUT;
        return result(
                emitted,
                committed,
                OptionalInt.empty(),
                finish,
                started,
                drafted,
                acceptedTotal,
                forwards);
    }

    private static int sample(Sampler sampler, MemoryView<?> logits, int vocabularySize) {
        int token = sampler.sampleToken(logits);
        if (token < 0 || token >= vocabularySize)
            throw new IllegalArgumentException(
                    "sampler returned token id " + token + " outside [0," + vocabularySize + ")");
        return token;
    }

    private static MemoryView<MemorySegment> row(
            MemoryView<MemorySegment> matrix, int row, int width) {
        long from = (long) row * width;
        return matrix.slice(0, from, from + width);
    }

    private static SpeculationResult result(
            IntSequence.Builder emitted,
            IntSequence.Builder committed,
            OptionalInt stop,
            FinishReason finish,
            long started,
            int drafted,
            int accepted,
            int forwards) {
        return new SpeculationResult(
                emitted.build(),
                committed.build(),
                stop,
                finish,
                Duration.ofNanos(System.nanoTime() - started),
                drafted,
                accepted,
                forwards);
    }
}
