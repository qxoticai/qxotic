// Shared Mixture-of-Experts plumbing over FP32-checked views. Shared here: the top-k selection
// (selectTopK, with an optional separate weights source for llama.cpp's selection-only exp_probs_b
// bias), the softmax+renormalize spine (softmaxSelectTopK), the CSR grouping of tokens by routed
// expert, the gather (a raw row copy), and the prob-weighted scatter-add (Ops.saxpyInPlace).
// Per-architecture - the model's identity - stay the gating flavor (softmax/sigmoid, extra
// scales), the normalization policy, and the per-expert FFN math (gated/ungated, activation,
// biases, layout), the latter supplied as an ExpertKernel closure - called once per expert
// (never per element), so the vector kernels inside stay monomorphic.
package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.readFloat;
import static com.qxotic.jinfer.Segments.writeFloat;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

public final class Moe {
    private Moe() {}

    /**
     * Per-route routing produced by a model's gating + top-k + normalize. Wraps the State's
     * existing CSR scratch arrays - no new buffers. {@code rowTopE[s*topK+k]}/{@code rowTopP[...]}
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
     * BEFORE this and normalization ({@link #normalizeTopP}, when the architecture wants it) AFTER,
     * both per-architecture. Fills {@code rowTopE}/{@code rowTopP} ({@code s*topK+k}) and re-zeros
     * then fills {@code counts}; {@code logits} is [rows][experts] and is consumed (masked) in
     * place.
     */
    public static void selectTopK(
            MemoryView<MemorySegment> logits,
            int rows,
            int experts,
            int topK,
            int[] rowTopE,
            float[] rowTopP,
            int[] counts) {
        selectTopK(logits, logits, rows, experts, topK, rowTopE, rowTopP, counts);
    }

    /**
     * Group-limited top-k used by DeepSeek-style routers: rank groups by the sum of their best two
     * selection scores, then choose experts only from the winning groups. Combine weights come from
     * {@code weights}, so a selection-only expert bias remains selection-only.
     */
    public static void selectTopKGrouped(
            MemoryView<MemorySegment> selection,
            MemoryView<MemorySegment> weights,
            int rows,
            int experts,
            int topK,
            int groups,
            int groupsUsed,
            int[] rowTopE,
            float[] rowTopP,
            int[] counts,
            float[] groupScores,
            boolean[] groupMask) {
        if (groups <= 0 || experts % groups != 0 || groupsUsed <= 0 || groupsUsed > groups)
            throw new IllegalArgumentException("invalid grouped routing dimensions");
        if (groupScores.length < groups || groupMask.length < groups)
            throw new IllegalArgumentException("group routing scratch is too small");
        Raw sel = Raw.f32(selection, "selection");
        Raw w = Raw.f32(weights, "weights");
        int perGroup = experts / groups;
        Arrays.fill(counts, 0);
        for (int row = 0; row < rows; row++) {
            long base = (long) row * experts;
            Arrays.fill(groupMask, 0, groups, false);
            for (int group = 0; group < groups; group++) {
                float first = Float.NEGATIVE_INFINITY, second = Float.NEGATIVE_INFINITY;
                int from = group * perGroup;
                for (int i = 0; i < perGroup; i++) {
                    float value =
                            readFloat(sel.vseg(), sel.vbase() + (base + from + i) * Float.BYTES);
                    if (value > first) {
                        second = first;
                        first = value;
                    } else if (value > second) second = value;
                }
                groupScores[group] = first + second;
            }
            for (int k = 0; k < groupsUsed; k++) {
                int best = -1;
                float value = Float.NEGATIVE_INFINITY;
                for (int group = 0; group < groups; group++) {
                    if (!groupMask[group] && (best < 0 || groupScores[group] > value)) {
                        value = groupScores[group];
                        best = group;
                    }
                }
                groupMask[best] = true;
            }
            int rowBase = row * topK;
            for (int k = 0; k < topK; k++) {
                int best = -1;
                float value = Float.NEGATIVE_INFINITY;
                for (int expert = 0; expert < experts; expert++) {
                    if (!groupMask[expert / perGroup]) continue;
                    boolean taken = false;
                    for (int prior = 0; prior < k; prior++)
                        if (rowTopE[rowBase + prior] == expert) {
                            taken = true;
                            break;
                        }
                    if (taken) continue;
                    float candidate =
                            readFloat(sel.vseg(), sel.vbase() + (base + expert) * Float.BYTES);
                    if (best < 0 || candidate > value) {
                        value = candidate;
                        best = expert;
                    }
                }
                if (best < 0)
                    throw new IllegalStateException("not enough experts in selected groups");
                rowTopE[rowBase + k] = best;
                rowTopP[rowBase + k] = readFloat(w.vseg(), w.vbase() + (base + best) * Float.BYTES);
                counts[best]++;
            }
        }
    }

    /**
     * Two-source variant: the argmax runs over {@code selection} (consumed, masked in place) but
     * the recorded combine weight is read from {@code weights}. This is llama.cpp's {@code
     * exp_probs_b} semantics (build_moe_ffn): the bias steers WHICH experts are picked, not HOW
     * MUCH they contribute - callers pass a bias-added scratch as {@code selection} and the
     * unbiased gating probabilities as {@code weights}.
     */
    public static void selectTopK(
            MemoryView<MemorySegment> selection,
            MemoryView<MemorySegment> weights,
            int rows,
            int experts,
            int topK,
            int[] rowTopE,
            float[] rowTopP,
            int[] counts) {
        Raw sel = Raw.f32(selection, "selection");
        Raw w = Raw.f32(weights, "weights");
        Arrays.fill(counts, 0);
        for (int s = 0; s < rows; s++) {
            long ro = (long) s * experts;
            for (int ki = 0; ki < topK; ki++) {
                int best = -1;
                float bestVal = Float.NEGATIVE_INFINITY;
                for (int ei = 0; ei < experts; ei++) {
                    float v = readFloat(sel.vseg(), sel.vbase() + (ro + ei) * Float.BYTES);
                    if (v > bestVal) {
                        bestVal = v;
                        best = ei;
                    }
                }
                // a row with nothing above -inf (NaN logits) still routes to topK DISTINCT
                // experts: the NaN travels in the weight, the per-expert counts stay <= rows
                if (best < 0) best = firstUnpicked(rowTopE, s * topK, ki, experts);
                rowTopE[s * topK + ki] = best;
                rowTopP[s * topK + ki] = readFloat(w.vseg(), w.vbase() + (ro + best) * Float.BYTES);
                writeFloat(
                        sel.vseg(),
                        sel.vbase() + (ro + best) * Float.BYTES,
                        Float.NEGATIVE_INFINITY);
                counts[best]++;
            }
        }
    }

    private static int firstUnpicked(int[] rowTopE, int rowBase, int picked, int experts) {
        for (int e = 0; e < experts; e++) {
            boolean taken = false;
            for (int k = 0; k < picked && !taken; k++) taken = rowTopE[rowBase + k] == e;
            if (!taken) return e;
        }
        throw new IllegalStateException("topK exceeds the expert count");
    }

    /**
     * The common softmax-routing spine (llama.cpp build_moe_ffn with softmax gating and {@code
     * norm_w=true}): softmax each row of {@code routerLogits} in place, select the top-k, then
     * renormalize the k weights to sum to 1. Architectures with other gating (sigmoid,
     * selection-time bias) compose the pieces themselves.
     */
    public static void softmaxSelectTopK(
            MemoryView<MemorySegment> routerLogits,
            int rows,
            int experts,
            int topK,
            int[] rowTopE,
            float[] rowTopP,
            int[] counts) {
        for (int s = 0; s < rows; s++)
            Ops.softmaxInPlace(routerLogits, (long) s * experts, experts);
        selectTopK(routerLogits, rows, experts, topK, rowTopE, rowTopP, counts);
        normalizeTopP(rowTopP, rows, topK);
    }

    /**
     * Renormalize every row's k selected weights to sum to 1 (llama.cpp build_moe_ffn {@code
     * norm_w}). Runs on {@link #selectTopK}'s {@code rowTopP} output.
     */
    public static void normalizeTopP(float[] rowTopP, int rows, int topK) {
        for (int s = 0; s < rows; s++) {
            float sum = 0f;
            for (int k = 0; k < topK; k++) sum += rowTopP[s * topK + k];
            for (int k = 0; k < topK; k++) rowTopP[s * topK + k] /= sum;
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
     * scale) - equivalent to applying it at the scatter, up to float rounding. {@code expertOut} is
     * the kernel's per-group output scratch.
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
        buildCsr(r, expertScale);
        int[] off = r.offsets;

        Ops.fillInPlace(out, 0, r.seqLen * dim, 0f);
        for (int e = 0; e < r.numExperts; e++) {
            int start = off[e], n = off[e + 1] - start;
            if (n == 0) continue;
            Parallel.forLoop(
                    n,
                    j ->
                            Convert.copyF32(
                                    input,
                                    (long) r.rowByExpert[start + j] * dim,
                                    gather,
                                    (long) j * dim,
                                    dim));
            kernel.apply(e, n, gather, expertOut);
            Parallel.forLoop(
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

    /** CSR grouping for dispatch; folds expertScale into the combine weights. */
    private static void buildCsr(Routing r, MemoryView<MemorySegment> expertScale) {
        Raw scaleRaw = expertScale != null ? Raw.f32(expertScale, "expertScale") : null;
        int[] off = r.offsets;
        off[0] = 0;
        for (int e = 0; e < r.numExperts; e++) off[e + 1] = off[e] + r.counts[e];
        System.arraycopy(off, 0, r.cursor, 0, r.numExperts);
        for (int s = 0; s < r.seqLen; s++) {
            for (int k = 0; k < r.topK; k++) {
                int e = r.rowTopE[s * r.topK + k];
                int pos = r.cursor[e]++;
                r.rowByExpert[pos] = s;
                r.probByExpert[pos] =
                        scaleRaw == null
                                ? r.rowTopP[s * r.topK + k]
                                : r.rowTopP[s * r.topK + k]
                                        * readFloat(
                                                scaleRaw.vseg(),
                                                scaleRaw.vbase() + (long) e * Float.BYTES);
            }
        }
    }
}
