package com.qxotic.jinfer;

/**
 * Microbench: the pure cost of a SpinPool dispatch+barrier with NO work in the region. Isolates the
 * per-region fork-join latency that the decode forward pays ~100x/token. Run on the decode pool so it
 * uses the real spin-barrier path.
 */
public final class SpinProbe {
    public static void main(String[] args) {
        int iters = args.length > 0 ? Integer.parseInt(args[0]) : 200_000;
        Parallel.onDecodePool(() -> {
            for (int w = 0; w < 5000; w++) Parallel.parallelFor(0, RuntimeFlags.DECODE_THREADS, i -> {});   // warm
            long t0 = System.nanoTime();
            for (int i = 0; i < iters; i++) Parallel.parallelFor(0, RuntimeFlags.DECODE_THREADS, j -> {});
            long ns = System.nanoTime() - t0;
            System.err.printf("empty parallelFor (%d participants): %.3f us/region  over %d iters%n",
                    RuntimeFlags.DECODE_THREADS, ns / 1e3 / iters, iters);
            return null;
        });
    }
}
