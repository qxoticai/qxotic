package com.qxotic.jinfer.models.gemmaembedding2;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/**
 * Unscaled non-causal attention where query {@code i} attends keys {@code [first[i], end[i])}, both
 * non-decreasing: whole inputs, sequences packed side by side, windows. Sequences attending few
 * keys per query share one region of dot products; the rest go as GEMMs per block of queries and
 * head (the scores, a softmax, the value mix as {@code valuesᵀ · scoresᵀ} with the queries as the
 * GEMM's rows), at jam's GEMM throughput where a flash tile loop is bound by its dot products.
 */
final class GemmAttention {
    private GemmAttention() {}

    private static final int QUERY_BLOCK = 1024;

    /**
     * At most this many keys per query on average and a sequence's GEMMs cost more than they save
     * (three regions per head and block): short inputs, short sequences packed. ponytail: 160 once
     * jam-native fans its packed F32 GEMM over whole 8-row tiles (2-4x faster at 128-448 rows).
     */
    static final int DIRECT_KEYS = 384;

    /** Working buffers for up to {@code rows} queries over values {@code kvWidth} wide. */
    record Scratch(
            MemoryView<MemorySegment> valuesT,
            MemoryView<MemorySegment> scores,
            MemoryView<MemorySegment> outputT,
            int[] directQueries,
            float[][] directScores) {
        static Scratch allocate(
                MemoryAllocator<MemorySegment> allocator, int rows, int kvWidth, int outWidth) {
            int block = Math.min(rows, QUERY_BLOCK);
            return new Scratch(
                    Views.allocateF32(allocator, (long) kvWidth * rows),
                    Views.allocateF32(allocator, (long) block * rows),
                    Views.allocateF32(allocator, (long) outWidth * block),
                    new int[rows],
                    new float[Parallel.threads()][rows]);
        }
    }

    /**
     * {@code out = softmax(q kᵀ over [first, end)) v} for {@code rows} queries, into {@code out}
     * packed {@code [rows, heads * headSize]}. Query rows sit at {@code qStride} with head h at
     * {@code h * headSize}; key and value rows at {@code kvStride}, query head h reading kv head
     * {@code h / (heads / kvHeads)}. A row whose keys start at itself ({@code first[r] == r})
     * begins a packed sequence; that grouping only steers the work, any would compute the same.
     */
    static void prefill(
            MemoryView<MemorySegment> q,
            int qStride,
            MemoryView<MemorySegment> k,
            MemoryView<MemorySegment> v,
            int kvStride,
            MemoryView<MemorySegment> out,
            int rows,
            int heads,
            int kvHeads,
            int headSize,
            int[] first,
            int[] end,
            Scratch scratch) {
        int kvMul = heads / kvHeads, outStride = heads * headSize;
        // blocks never span two sequences, and each goes direct or as GEMMs by its own keys
        int[] direct = scratch.directQueries();
        int directRows = 0;
        boolean transposed = false;
        for (int s = 0, e; s < rows; s = e) {
            long attended = end[s] - first[s];
            for (e = s + 1; e < rows && first[e] != e; e++) attended += end[e] - first[e];
            if (attended <= (long) DIRECT_KEYS * (e - s)) {
                for (int i = s; i < e; i++) direct[directRows++] = i;
                continue;
            }
            if (!transposed) { // each head dimension becomes one contiguous run over the keys
                Ops.transposeCopy(v, rows, kvStride, scratch.valuesT);
                transposed = true;
            }
            for (int q0 = s; q0 < e; q0 += QUERY_BLOCK) {
                int queries = Math.min(QUERY_BLOCK, e - q0);
                int k0 = first[q0], width = end[q0 + queries - 1] - k0, firstQuery = q0;
                for (int h = 0; h < heads; h++) { // sequential: mm runs its own parallel region
                    long kvHead = (long) (h / kvMul) * headSize;
                    MatMul.mm(
                            k,
                            (long) k0 * kvStride + kvHead,
                            kvStride,
                            q,
                            (long) q0 * qStride + (long) h * headSize,
                            qStride,
                            scratch.scores,
                            0,
                            width,
                            width,
                            queries,
                            headSize);
                    Parallel.forLoop(
                            queries,
                            i -> {
                                long row = (long) i * width;
                                int lo = first[firstQuery + i] - k0;
                                int hi = end[firstQuery + i] - k0;
                                Ops.fillInPlace(scratch.scores, row, lo, 0f);
                                Ops.softmaxInPlace(scratch.scores, row + lo, hi - lo);
                                Ops.fillInPlace(scratch.scores, row + hi, width - hi, 0f);
                            });
                    MatMul.mm(
                            scratch.scores,
                            0,
                            width,
                            scratch.valuesT,
                            kvHead * rows + k0,
                            rows,
                            scratch.outputT,
                            (long) h * headSize * queries,
                            queries,
                            queries,
                            headSize,
                            width);
                }
                Ops.transposeCopy(
                        scratch.outputT, outStride, queries, out.slice(0, q0, q0 + queries));
            }
        }
        if (directRows > 0) {
            int[] queryRows = direct;
            int count = directRows;
            float[][] scores = scratch.directScores();
            Parallel.forLoop(
                    heads * count,
                    (job, slot) -> {
                        int h = job / count, i = queryRows[job % count];
                        int lo = first[i], n = end[i] - lo;
                        long kvHead = (long) (h / kvMul) * headSize;
                        long at = (long) i * outStride + (long) h * headSize;
                        float[] p = scores[slot];
                        for (int j = 0; j < n; j++)
                            p[j] =
                                    Ops.dot(
                                            q,
                                            (long) i * qStride + (long) h * headSize,
                                            k,
                                            (long) (lo + j) * kvStride + kvHead,
                                            headSize);
                        float max = Float.NEGATIVE_INFINITY;
                        for (int j = 0; j < n; j++) max = Math.max(max, p[j]);
                        double sum = 0;
                        for (int j = 0; j < n; j++) sum += p[j] = (float) Math.exp(p[j] - max);
                        Ops.fillInPlace(out, at, headSize, 0f);
                        for (int j = 0; j < n; j++)
                            Ops.saxpyInPlace(
                                    out,
                                    at,
                                    v,
                                    (long) (lo + j) * kvStride + kvHead,
                                    headSize,
                                    (float) (p[j] / sum));
                    });
        }
    }
}
