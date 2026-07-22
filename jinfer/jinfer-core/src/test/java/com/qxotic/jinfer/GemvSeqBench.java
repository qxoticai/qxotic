package com.qxotic.jinfer;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Walks a large (&gt; L3) Q8_0 matrix in {@code rowsPerGemv}-row slices, running each slice through
 * {@code vectorGemv512} — simulating decode's sequence of separate gemvs over cold weights.
 * Sweeping {@code rowsPerGemv} shows whether small/medium gemvs reach the kernel's ~121 GB/s peak
 * or fall short (per-gemv prefetch ramp-up), which decides whether fusing consecutive gemvs into
 * larger streams helps.
 *
 * <p>java ... com.qxotic.GemvSeqBench [totalRows=65536] [dim1=4096] [rowsPerGemv=2048] [iters=200]
 */
public final class GemvSeqBench {
    @Test
    @Tag("bench")
    void run() throws Exception {
        main(testArgs());
    }

    private static String[] testArgs() {
        String argv = System.getProperty("jinfer.args", "");
        return argv.isBlank() ? new String[0] : argv.trim().split("\\s+");
    }

    private static void main(String[] args) {
        int totalRows = args.length > 0 ? Integer.parseInt(args[0]) : 65536;
        int dim1 = args.length > 1 ? Integer.parseInt(args[1]) : 4096;
        int rowsPerGemv = args.length > 2 ? Integer.parseInt(args[2]) : 2048;
        int iters = args.length > 3 ? Integer.parseInt(args[3]) : 200;
        int blockSize = 32, typeSize = 34;
        long rowBytes = (long) (dim1 / blockSize) * typeSize;
        long wbytes = (long) totalRows * rowBytes;

        MemorySegment wseg = Arena.ofAuto().allocate(wbytes, 64);
        Q8_0FloatTensor w = new Q8_0FloatTensor((long) totalRows * dim1, wseg);
        F32FloatTensor x = F32FloatTensor.allocate(dim1);
        F32FloatTensor out = F32FloatTensor.allocate(totalRows);
        for (int i = 0; i < dim1; i++) x.setFloat(i, 0.013f * (i % 11));
        int nGemv = totalRows / rowsPerGemv;

        for (int it = 0; it < 4; it++)
            for (int g = 0; g < nGemv; g++)
                MatMul.instance()
                        .mm(
                                w,
                                (long) g * rowsPerGemv * dim1,
                                dim1,
                                x,
                                0,
                                dim1,
                                out,
                                g * rowsPerGemv,
                                rowsPerGemv,
                                rowsPerGemv,
                                1,
                                dim1);
        long t0 = System.nanoTime();
        for (int it = 0; it < iters; it++)
            for (int g = 0; g < nGemv; g++)
                MatMul.instance()
                        .mm(
                                w,
                                (long) g * rowsPerGemv * dim1,
                                dim1,
                                x,
                                0,
                                dim1,
                                out,
                                g * rowsPerGemv,
                                rowsPerGemv,
                                rowsPerGemv,
                                1,
                                dim1);
        long t1 = System.nanoTime();

        double secs = (t1 - t0) / 1e9;
        double gbps = (double) wbytes * iters / secs / 1e9;
        System.out.printf(
                "rowsPerGemv=%-5d  %.1f GB/s  (%d gemvs/iter, total %.0f MB)%n",
                rowsPerGemv, gbps, nGemv, wbytes / 1e6);
    }
}
