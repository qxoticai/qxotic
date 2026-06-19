package com.llama4j;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Measures prefill throughput (tokens/sec) on a long prompt ingested as one chunk. Prefill exercises the
 * causal-attention path whose per-query work is triangular (query i attends to i keys), so it is the
 * workload where {@code Parallel.parallelFor} load-balancing (over-decomposition + work stealing) can help,
 * unlike the uniform decode path.
 *
 *   java ... com.llama4j.PrefillBench <model.gguf> [promptTokens=1024] [warmup=3] [measure=8]
 */
public final class PrefillBench {
    public static void main(String[] args) throws Exception {
        Path path = Path.of(args[0]);
        int promptLen = args.length > 1 ? Integer.parseInt(args[1]) : 1024;   // one batch chunk (capacity 1024)
        int warmup = args.length > 2 ? Integer.parseInt(args[2]) : 3;
        int measure = args.length > 3 ? Integer.parseInt(args[3]) : 8;

        Model model = ModelLoader.loadModel(path, Math.max(4096, promptLen + 16));

        // Build a prompt of ~promptLen tokens by tokenizing repeated prose and trimming.
        String para = "The history of the Roman empire is long and complex, spanning many centuries of "
                + "conquest, civil strife, cultural achievement and eventual decline across three continents. ";
        StringBuilder sb = new StringBuilder();
        List<Integer> enc = new ArrayList<>();
        while (enc.size() < promptLen) { sb.append(para); enc = model.tokenizer().encode(sb.toString()); }
        int[] prompt = new int[promptLen];
        for (int i = 0; i < promptLen; i++) prompt[i] = enc.get(i);

        InferenceState state = model.createNewState();
        for (int i = 0; i < warmup; i++) model.ingest(state, prompt, 0, 0, promptLen);

        long t0 = System.nanoTime();
        for (int i = 0; i < measure; i++) model.ingest(state, prompt, 0, 0, promptLen);
        long t1 = System.nanoTime();

        double secs = (t1 - t0) / 1e9;
        double toks = (double) promptLen * measure;
        System.out.printf("%-34s prefill %.0f tok/s (%d tok x %d, %.2fs)%n",
                path.getFileName().toString(), toks / secs, promptLen, measure, secs);
        if (System.getProperty("bench.err") != null) System.err.printf("RESULT %.0f%n", toks / secs);
    }
}
