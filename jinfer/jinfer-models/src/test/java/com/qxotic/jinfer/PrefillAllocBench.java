package com.qxotic.jinfer;

import com.sun.management.ThreadMXBean;

import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Prefill throughput AND heap-allocation rate. A nondeterministic JIT "boxing collapse" (a hot Vector API
 * method compiled to boxed FloatVector ops instead of SIMD) shows up as BOTH a big tok/s drop and a big
 * allocation spike (each boxed binaryOp/ternaryOp allocates a FloatVector). Run repeatedly: collapsed
 * processes should pair low tok/s with high MB/token, confirming the mechanism and which runs are boxed.
 *
 *   java ... com.qxotic.PrefillAllocBench <model.gguf> [chunk=512] [warmup=12] [measure=14]
 */
public final class PrefillAllocBench {
    static long totalAllocated() {
        var tb = (ThreadMXBean) ManagementFactory.getThreadMXBean();
        long s = 0;
        for (long id : tb.getAllThreadIds()) { long b = tb.getThreadAllocatedBytes(id); if (b > 0) s += b; }
        return s;
    }

    public static void main(String[] args) throws Exception {
        Path path = Path.of(args[0]);
        int chunk = args.length > 1 ? Integer.parseInt(args[1]) : 512;
        int warmup = args.length > 2 ? Integer.parseInt(args[2]) : 12;
        int measure = args.length > 3 ? Integer.parseInt(args[3]) : 14;

        Model model = ModelLoader.loadModel(path, Math.max(4096, chunk + 16));
        String para = "The history of the Roman empire is long and complex, spanning many centuries of "
                + "conquest, civil strife, cultural achievement and eventual decline across three continents. ";
        StringBuilder sb = new StringBuilder();
        List<Integer> enc = new ArrayList<>();
        while (enc.size() < chunk) { sb.append(para); enc = model.tokenizer().encode(sb.toString()); }
        int[] prompt = new int[chunk];
        for (int i = 0; i < chunk; i++) prompt[i] = enc.get(i);

        InferenceState state = model.createNewState();
        for (int i = 0; i < warmup; i++) model.ingest(state, prompt, 0, 0, chunk);

        long a0 = totalAllocated(), t0 = System.nanoTime();
        for (int i = 0; i < measure; i++) model.ingest(state, prompt, 0, 0, chunk);
        long t1 = System.nanoTime(), a1 = totalAllocated();

        double secs = (t1 - t0) / 1e9, toks = (double) chunk * measure;
        System.out.printf("%-26s prefill %.0f tok/s | alloc %.0f MB/s | %.2f MB/token%n",
                path.getFileName().toString(), toks / secs,
                (a1 - a0) / 1048576.0 / secs, (a1 - a0) / 1048576.0 / toks);
    }
}
