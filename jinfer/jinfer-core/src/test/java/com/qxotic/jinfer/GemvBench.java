package com.qxotic.jinfer;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

/**
 * Measures the Q8_0 decode gemv kernel's peak DRAM bandwidth in isolation: one large weight matrix
 * (sized &gt; L3 so it streams from DRAM) run through {@code vectorGemv512} repeatedly. Compare the GB/s
 * to decode's effective bandwidth (~101 GB/s) and llama.cpp's (~121) to tell whether the gap is the
 * kernel itself or the per-token overhead of ~360 small gemvs + barriers.
 *
 *   java ... com.qxotic.GemvBench [dim0=16384] [dim1=8192] [iters=2000]
 */
public final class GemvBench {
    public static void main(String[] args) {
        int dim0 = args.length > 0 ? Integer.parseInt(args[0]) : 16384;
        int dim1 = args.length > 1 ? Integer.parseInt(args[1]) : 8192;
        int iters = args.length > 2 ? Integer.parseInt(args[2]) : 2000;
        int blockSize = 32;   // Q8_0 elements per block
        int typeSize = 34;    // Q8_0 block bytes (2 scale + 32 int8)
        long wbytes = (long) dim0 * (dim1 / blockSize) * typeSize;

        MemorySegment wseg = Arena.ofAuto().allocate(wbytes, 64);   // zero-filled; reads happen regardless
        Q8_0FloatTensor w = new Q8_0FloatTensor((long) dim0 * dim1, wseg);
        F32FloatTensor x = F32FloatTensor.allocate(dim1);
        F32FloatTensor out = F32FloatTensor.allocate(dim0);
        for (int i = 0; i < dim1; i++) x.setFloat(i, 0.013f * (i % 11));

        for (int it = 0; it < 60; it++) MatMul.INSTANCE.mm(w, 0, dim1, x, 0, dim1, out, 0, dim0, dim0, 1, dim1);
        long t0 = System.nanoTime();
        for (int it = 0; it < iters; it++) MatMul.INSTANCE.mm(w, 0, dim1, x, 0, dim1, out, 0, dim0, dim0, 1, dim1);
        long t1 = System.nanoTime();

        double secs = (t1 - t0) / 1e9;
        double gbps = (double) wbytes * iters / secs / 1e9;
        System.out.printf("gemv %dx%d  %.1f GB/s  (%.3f ms/gemv, %d iters, weights %.0f MB)%n",
                dim0, dim1, gbps, secs / iters * 1e3, iters, wbytes / 1e6);
    }
}
