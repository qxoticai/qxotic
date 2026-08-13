// Shared Mixture-of-Experts dispatch, ported from jinfer-kernels Moe: tensor params become
// FP32-checked views; the gather is a raw row copy and the scatter-add x.Ops.saxpyInPlace.
// The router/gating (softmax/sigmoid, top-k, normalization) is a model's identity and stays
// per-architecture; this owns only the architecture-independent plumbing: the CSR grouping of
// tokens by routed expert, the gather, and the prob-weighted scatter-add. The per-expert FFN math
// (gated/ungated, activation, biases, layout) is supplied as an ExpertKernel closure — called once
// per expert (never per element), so the vector kernels inside stay monomorphic.
package com.qxotic.jinfer.x.kernels;

import static com.qxotic.jinfer.x.Segments.readFloat;
import static com.qxotic.jinfer.x.Segments.writeFloat;

import com.qxotic.jinfer.x.Parallel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

public final class Moe {
    private Moe() {}

    /**
     * Per-route routing produced by a model's gating + top-k + normalize. Wraps the State's
     * existing CSR scratch arrays — no new buffers. {@code rowTopE[s*topK+k]}/{@code rowTopP[...]}
     * are the selected expert and its combine weight for route k of row s; {@code counts[e]} is how
     * many routes landed on expert e (the rest are filled by {@link #dispatch}).
     */
    public static final class Routing {
        final int[] rowTopE, counts, offsets, cursor, rowByExpert;
        final float[] rowTopP, probByExpert;
        public int seqLen, topK, numExperts; // per-call scalars; the scratch arrays are wired once

        /**
         * The caller supplies only the arrays its gating fills ({@code rowTopE}/{@code rowTopP}
         * sized rows*topK, {@code counts} sized numExperts); the dispatch-internal CSR scratch
         * (offsets/cursor/gather order) is allocated here, once per State - {@link #dispatch} needs
         * no per-call allocation.
         */
        public Routing(int[] rowTopE, float[] rowTopP, int[] counts) {
            this(
                    rowTopE,
                    rowTopP,
                    counts,
                    new int[counts.length + 1],
                    new int[counts.length],
                    new int[rowTopE.length],
                    new float[rowTopE.length]);
        }

        private Routing(
                int[] rowTopE,
                float[] rowTopP,
                int[] counts,
                int[] offsets,
                int[] cursor,
                int[] rowByExpert,
                float[] probByExpert) {
            this.rowTopE = rowTopE;
            this.rowTopP = rowTopP;
            this.counts = counts;
            this.offsets = offsets;
            this.cursor = cursor;
            this.rowByExpert = rowByExpert;
            this.probByExpert = probByExpert;
        }
    }

    /**
     * The shared top-k selection over gated logits (every architecture's loop is identical: k times
     * argmax, mask the winner to -Infinity, count the route). Gating (bias/softmax/sigmoid) runs
     * BEFORE this and normalization AFTER, both per-architecture. Fills {@code rowTopE}/{@code
     * rowTopP} ({@code s*topK+k}) and re-zeros then fills {@code counts}; {@code logits} is
     * [rows][experts] and is consumed (masked) in place.
     */
    public static void selectTopK(
            MemoryView<MemorySegment> logits,
            int rows,
            int experts,
            int topK,
            int[] rowTopE,
            float[] rowTopP,
            int[] counts) {
        Raw r = Raw.f32(logits, "logits");
        Arrays.fill(counts, 0);
        for (int s = 0; s < rows; s++) {
            long ro = (long) s * experts;
            for (int ki = 0; ki < topK; ki++) {
                int best = 0;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int ei = 0; ei < experts; ei++) {
                    float v = readFloat(r.vseg(), r.vbase() + (ro + ei) * Float.BYTES);
                    if (v > bestVal) {
                        bestVal = v;
                        best = ei;
                    }
                }
                rowTopE[s * topK + ki] = best;
                rowTopP[s * topK + ki] = bestVal;
                writeFloat(
                        r.vseg(), r.vbase() + (ro + best) * Float.BYTES, Float.NEGATIVE_INFINITY);
                counts[best]++;
            }
        }
    }

    /**
     * Expert {@code e}'s FFN over {@code n} gathered rows ({@code gather}, stride dim) → {@code n}
     * rows in {@code out} (stride dim). Gated/ungated, activation, biases and weight layout live
     * here.
     */
    public interface ExpertKernel {
        void apply(int e, int n, MemoryView<MemorySegment> gather, MemoryView<MemorySegment> out);
    }

    /**
     * CSR-grouped MoE dispatch: build the per-expert row buckets from {@code r}, gather each
     * expert's rows out of {@code input}, run its {@code kernel}, and scatter-add the result into
     * {@code out} weighted by the route's combine weight. {@code expertScale} (nullable) folds a
     * per-expert output scale into the combine weight at build time (e.g. Gemma's per-expert down
     * scale) — byte-identical to applying it at the scatter. {@code expertOut} is the kernel's
     * per-group output scratch.
     */
    public static void dispatch(
            Routing r,
            int dim,
            MemoryView<MemorySegment> input,
            MemoryView<MemorySegment> gather,
            MemoryView<MemorySegment> expertOut,
            MemoryView<MemorySegment> out,
            MemoryView<MemorySegment> expertScale,
            ExpertKernel kernel) {
        Raw scaleRaw = expertScale != null ? Raw.f32(expertScale, "expertScale") : null;
        int[] off = r.offsets;
        off[0] = 0;
        for (int e = 0; e < r.numExperts; e++) off[e + 1] = off[e] + r.counts[e];
        System.arraycopy(off, 0, r.cursor, 0, r.numExperts);
        for (int s = 0; s < r.seqLen; s++) {
            for (int k = 0; k < r.topK; k++) {
                int e = r.rowTopE[s * r.topK + k];
                if (e < 0)
                    continue; // unfilled top-k slot (e.g. Qwen's insertion sort); not counted
                // either
                int pos = r.cursor[e]++;
                r.rowByExpert[pos] = s;
                r.probByExpert[pos] =
                        scaleRaw == null
                                ? r.rowTopP[s * r.topK + k]
                                : r.rowTopP[s * r.topK + k]
                                        * com.qxotic.jinfer.x.Segments.readFloat(
                                                scaleRaw.vseg(),
                                                scaleRaw.vbase() + (long) e * Float.BYTES);
            }
        }

        Ops.fillInPlace(out, 0, r.seqLen * dim, 0f);
        for (int e = 0; e < r.numExperts; e++) {
            int start = off[e], n = off[e + 1] - start;
            if (n == 0) continue;
            Parallel.forRows(
                    n,
                    j ->
                            Convert.copyF32(
                                    input,
                                    (long) r.rowByExpert[start + j] * dim,
                                    gather,
                                    (long) j * dim,
                                    dim));
            kernel.apply(e, n, gather, expertOut);
            Parallel.forRows(
                    n,
                    j ->
                            Ops.saxpyInPlace(
                                    out,
                                    (long) r.rowByExpert[start + j] * dim,
                                    expertOut,
                                    (long) j * dim,
                                    dim,
                                    r.probByExpert[start + j]));
        }
    }
}
