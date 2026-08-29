package com.qxotic.jinfer.bench;

import com.qxotic.jinfer.Parallel;

/**
 * Microbench: the pure cost of one {@link Parallel#forLoop} region with NO work in it - the
 * per-region dispatch and barrier latency a decode token pays ~100 times.
 */
public final class SpinProbe {
    public static void main(String[] args) {
        int iters = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;
        int t = Parallel.threads();
        for (int w = 0; w < 5000; w++) Parallel.forLoop(0, t, i -> {}); // warm
        long t0 = System.nanoTime();
        for (int i = 0; i < iters; i++) Parallel.forLoop(0, t, j -> {});
        long ns = System.nanoTime() - t0;
        System.err.printf(
                "empty forLoop (%d participants): %.3f us/region over %d iters%n",
                t, ns / 1e3 / iters, iters);
    }
}
