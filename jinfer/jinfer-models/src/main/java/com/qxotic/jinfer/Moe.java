// Shared Mixture-of-Experts dispatch. The router/gating (softmax/sigmoid, top-k, normalization) is a
// model's identity and stays per-architecture; this owns only the architecture-independent plumbing:
// the CSR grouping of tokens by routed expert, the gather, and the prob-weighted scatter-add. The
// per-expert FFN math (gated/ungated, activation, biases, layout) is supplied as an ExpertKernel
// closure — called once per expert (never per element), so the vector kernels inside stay monomorphic.
package com.qxotic.jinfer;

final class Moe {
    private Moe() {}

    /** Per-route routing produced by a model's gating + top-k + normalize. Wraps the State's existing
     *  CSR scratch arrays — no new buffers. {@code rowTopE[s*topK+k]}/{@code rowTopP[...]} are the
     *  selected expert and its combine weight for route k of row s; {@code counts[e]} is how many
     *  routes landed on expert e (the rest are filled by {@link #dispatch}). */
    static final class Routing {
        final int[] rowTopE, counts, offsets, cursor, rowByExpert;
        final float[] rowTopP, probByExpert;
        int seqLen, topK, numExperts;   // per-call scalars; the scratch arrays are wired once

        /** Wraps a State's per-call CSR scratch so {@link #dispatch} needs no per-call allocation. */
        Routing(int[] rowTopE, float[] rowTopP, int[] counts, int[] offsets,
                int[] cursor, int[] rowByExpert, float[] probByExpert) {
            this.rowTopE = rowTopE; this.rowTopP = rowTopP; this.counts = counts;
            this.offsets = offsets; this.cursor = cursor; this.rowByExpert = rowByExpert;
            this.probByExpert = probByExpert;
        }
    }

    /** Expert {@code e}'s FFN over {@code n} gathered rows ({@code gather}, stride dim) → {@code n} rows
     *  in {@code out} (stride dim). Gated/ungated, activation, biases and weight layout live here. */
    interface ExpertKernel {
        void apply(int e, int n, FloatTensor gather, FloatTensor out);
    }

    /**
     * CSR-grouped MoE dispatch: build the per-expert row buckets from {@code r}, gather each expert's
     * rows out of {@code input}, run its {@code kernel}, and scatter-add the result into {@code out}
     * weighted by the route's combine weight. {@code expertScale} (nullable) folds a per-expert output
     * scale into the combine weight at build time (e.g. Gemma's per-expert down scale) — byte-identical
     * to applying it at the scatter. {@code expertOut} is the kernel's per-group output scratch.
     */
    static void dispatch(Routing r, int dim, FloatTensor input, FloatTensor gather,
                         FloatTensor expertOut, FloatTensor out, FloatTensor expertScale, ExpertKernel kernel) {
        int[] off = r.offsets;
        off[0] = 0;
        for (int e = 0; e < r.numExperts; e++) off[e + 1] = off[e] + r.counts[e];
        System.arraycopy(off, 0, r.cursor, 0, r.numExperts);
        for (int s = 0; s < r.seqLen; s++) {
            for (int k = 0; k < r.topK; k++) {
                int e = r.rowTopE[s * r.topK + k];
                if (e < 0) continue;   // unfilled top-k slot (e.g. Qwen's insertion sort); not counted either
                int pos = r.cursor[e]++;
                r.rowByExpert[pos] = s;
                r.probByExpert[pos] = expertScale == null
                        ? r.rowTopP[s * r.topK + k]
                        : r.rowTopP[s * r.topK + k] * expertScale.getFloat(e);
            }
        }

        out.fillInPlace(0, r.seqLen * dim, 0f);
        for (int e = 0; e < r.numExperts; e++) {
            int start = off[e], n = off[e + 1] - start;
            if (n == 0) continue;
            Parallel.forRows(n, j -> input.copyTo(r.rowByExpert[start + j] * dim, gather, j * dim, dim));
            kernel.apply(e, n, gather, expertOut);
            Parallel.forRows(n, j -> out.saxpyInPlace(r.rowByExpert[start + j] * dim, expertOut, j * dim, dim,
                    r.probByExpert[start + j]));
        }
    }
}
